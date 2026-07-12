"""Poker44 miner watchdog: status checks + auto-retrain on new benchmark releases.

Designed to be run every ~30 min from cron. Each run it:
  1. Reads on-chain stats for our hotkey (incentive / rank / stake / active).
  2. Counts validator queries seen in the pm2 miner log since the last run.
  3. If the benchmark API has a release newer than what the live model was
     trained on, downloads it, retrains, and restarts the miner via pm2.
  4. Appends a one-line status record to the monitor log and updates state.

Everything is best-effort: any step that fails is logged and the run continues,
so a transient RPC/API hiccup never leaves the miner in a bad state.

Config via env (all optional; sensible defaults):
  MINER_HOTKEY_SS58   hotkey to track on-chain (default: the registered hk1)
  NETUID              default 126
  NETWORK             default finney
  PM2_NAME            default poker44_miner
  PM2_OUT_LOG         default /root/.pm2/logs/poker44-miner-out.log
  REPO_DIR            default: repo root inferred from this file
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO_DIR = Path(os.environ.get("REPO_DIR", Path(__file__).resolve().parents[2]))
PYTHON = sys.executable  # the venv that launched us (has bittensor + lightgbm)

NETUID = int(os.environ.get("NETUID", "126"))
NETWORK = os.environ.get("NETWORK", "finney")
PM2_NAME = os.environ.get("PM2_NAME", "poker44_miner")
PM2_OUT_LOG = Path(
    os.environ.get("PM2_OUT_LOG", "/root/.pm2/logs/poker44-miner-out.log")
)
HOTKEY_SS58 = os.environ.get(
    "MINER_HOTKEY_SS58", "5CoUkv71YhzgwA5VL5DCC5KmVZYUMgrS536tpfuBQCAwVWsW"
)

DATA_DIR = REPO_DIR / "data"
STATE_PATH = DATA_DIR / "monitor_state.json"
MONITOR_LOG = DATA_DIR / "monitor.log"
LOCK_PATH = DATA_DIR / "monitor.lock"
ARTIFACT_META = REPO_DIR / "poker44" / "model" / "artifacts" / "model_meta.json"
BENCHMARK_API = "https://api.poker44.net/api/v1/benchmark"

QUERY_MARKER = "Scored "  # emitted by miner.forward() per validator query


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def log_line(msg: str) -> None:
    line = f"{_now()} | {msg}"
    print(line)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with MONITOR_LOG.open("a") as handle:
        handle.write(line + "\n")


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}


def save_state(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def trained_through() -> str | None:
    try:
        meta = json.loads(ARTIFACT_META.read_text())
        return max(meta.get("release_dates") or []) or None
    except Exception:  # noqa: BLE001
        return None


def count_queries() -> int:
    """Total validator query events currently visible in the pm2 out log."""
    try:
        text = PM2_OUT_LOG.read_text(errors="ignore")
        return text.count(QUERY_MARKER)
    except Exception:  # noqa: BLE001
        return -1


def chain_stats() -> dict:
    """On-chain stats for our hotkey. Returns {} on any failure."""
    try:
        import bittensor as bt

        st = bt.subtensor(network=NETWORK)
        mg = st.metagraph(netuid=NETUID)
        if HOTKEY_SS58 not in mg.hotkeys:
            return {"error": "hotkey_not_in_metagraph"}
        uid = mg.hotkeys.index(HOTKEY_SS58)
        return {
            "uid": int(uid),
            "incentive": float(mg.I[uid]),
            "rank": float(mg.R[uid]),
            "stake": float(mg.S[uid]),
            "trust": float(mg.T[uid]),
            "active": int(mg.active[uid]) if hasattr(mg, "active") else None,
            "block": int(mg.block.item()),
        }
    except Exception as err:  # noqa: BLE001
        return {"error": repr(err)[:160]}


def latest_release_date() -> str | None:
    try:
        data = requests.get(BENCHMARK_API, timeout=30).json()["data"]
        return data.get("latestSourceDate")
    except Exception:  # noqa: BLE001
        return None


LEADERBOARD_API = "https://api.poker44.net/api/v1/competition/leaderboard"


def leaderboard_standing() -> dict:
    """Our row + the top-10 cutline from the public competition leaderboard.

    This is the metric that actually pays (ranked_top_10 settlement), so we
    track it directly instead of only the benchmark proxy.
    """
    try:
        data = requests.get(LEADERBOARD_API, params={"limit": 300}, timeout=30).json()[
            "data"
        ]
        rows = data.get("rows") or []
        epoch = data.get("epoch") or {}
        cutline = None
        if len(rows) >= 10:
            cutline = rows[9].get("compositeScore")
        mine = next((r for r in rows if str(r.get("uid")) == "186"), None)
        out = {
            "epoch_id": epoch.get("epochId"),
            "epoch_mode": epoch.get("settlementMode"),
            "epoch_ends": epoch.get("endsAt"),
            "top10_cutline": cutline,
        }
        if mine:
            windows = [
                w.get("compositeScore")
                for w in (mine.get("windowCompositeScores") or [])
            ]
            out.update(
                {
                    "rank": mine.get("rank"),
                    "composite": mine.get("compositeScore"),
                    "reporting_validators": mine.get("reportingValidators"),
                    "score_source": mine.get("scoreSource"),
                    "manifest_present": mine.get("manifestPresent"),
                    "compliance": mine.get("complianceStatus"),
                    "incentive": mine.get("incentive"),
                    "windows": windows,
                    "windows_scored": sum(1 for w in windows if w is not None),
                }
            )
        return out
    except Exception as err:  # noqa: BLE001
        return {"error": repr(err)[:120]}


def run(cmd: list[str], timeout: int) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        tail = (proc.stdout or "")[-500:] + (proc.stderr or "")[-200:]
        return proc.returncode == 0, tail.strip()
    except Exception as err:  # noqa: BLE001
        return False, repr(err)[:200]


def maybe_retrain(latest: str | None) -> str:
    current = trained_through()
    if not latest:
        return "retrain: skipped (benchmark API unreachable)"
    if current and latest <= current:
        return f"retrain: up-to-date (model through {current}, latest {latest})"

    log_line(f"retrain: NEW release {latest} > trained {current}; starting")
    ok, tail = run(
        [PYTHON, "scripts/miner/download_benchmark.py"], timeout=1800
    )
    if not ok:
        return f"retrain: download FAILED — {tail}"
    ok, tail = run([PYTHON, "scripts/miner/train_model.py"], timeout=3600)
    if not ok:
        return f"retrain: train FAILED — {tail}"
    ok, tail = run(["pm2", "restart", PM2_NAME], timeout=120)
    if not ok:
        return f"retrain: DONE but pm2 restart FAILED — {tail}"
    return f"retrain: DONE, retrained through {latest}, miner reloaded"


def acquire_lock() -> bool:
    """Prevent overlapping runs (a retrain can outlast the 30-min interval)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        age = time.time() - LOCK_PATH.stat().st_mtime
        if age < 3 * 3600:  # stale after 3h
            return False
    LOCK_PATH.write_text(str(os.getpid()))
    return True


def release_lock() -> None:
    try:
        LOCK_PATH.unlink()
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    if not acquire_lock():
        log_line("skip: another monitor run is in progress (lock held)")
        return
    try:
        state = load_state()
        stats = chain_stats()
        queries = count_queries()
        prev_queries = state.get("query_count", 0)
        delta = queries - prev_queries if queries >= 0 and prev_queries >= 0 else 0

        if "error" in stats:
            status = f"chain error: {stats['error']}"
        else:
            first = "NONE YET" if queries <= 0 else f"{queries} total"
            status = (
                f"UID {stats['uid']} incentive={stats['incentive']:.6f} "
                f"rank={stats['rank']:.6f} stake={stats['stake']:.2f} "
                f"active={stats['active']} block={stats['block']} | "
                f"validator_queries={first} (+{delta} since last check)"
            )

        board = leaderboard_standing()
        if "error" in board:
            board_msg = f"leaderboard error: {board['error']}"
        elif "rank" in board:
            board_msg = (
                f"LEADERBOARD rank={board['rank']} composite={board['composite']} "
                f"incentive={board.get('incentive')} "
                f"windows_scored={board.get('windows_scored')}/5 "
                f"reporting_validators={board['reporting_validators']} | "
                f"top10_cutline={board['top10_cutline']} epoch_ends={board['epoch_ends']}"
            )
        else:
            board_msg = "leaderboard: UID 186 not listed"

        # --- Change detection: alert on new window score, composite move, or
        #     first nonzero incentive. Written prominently to monitor.log +
        #     a dedicated alerts file.
        prev = state.get("leaderboard") or {}
        alerts = []
        if "rank" in board:
            pw = prev.get("windows_scored", 0) or 0
            cw = board.get("windows_scored", 0) or 0
            if cw > pw:
                new_scores = [w for w in (board.get("windows") or []) if w is not None]
                alerts.append(
                    f"NEW WINDOW SCORED ({pw}->{cw}) | windows={new_scores} | "
                    f"composite={board.get('composite')} vs top10_cutline={board.get('top10_cutline')}"
                )
            pc, cc = prev.get("composite"), board.get("composite")
            if isinstance(pc, (int, float)) and isinstance(cc, (int, float)) and abs(cc - pc) >= 0.003:
                direction = "UP" if cc > pc else "DOWN"
                cut = board.get("top10_cutline")
                gap = f"{cc - cut:+.4f} vs top10" if isinstance(cut, (int, float)) else ""
                alerts.append(f"COMPOSITE {direction} {pc:.4f}->{cc:.4f} {gap}")
            pi, ci = prev.get("incentive") or 0, board.get("incentive") or 0
            if ci > 1e-6 and pi <= 1e-6:
                alerts.append(f"INCENTIVE STARTED: {ci:.6f} 🎉")
        for a in alerts:
            log_line(f"🔔 ALERT | {a}")
            try:
                with (DATA_DIR / "alerts.log").open("a") as fh:
                    fh.write(f"{_now()} | {a}\n")
            except Exception:  # noqa: BLE001
                pass

        latest = latest_release_date()
        retrain_msg = maybe_retrain(latest)
        log_line(f"{status} | {board_msg} | {retrain_msg}")

        state.update(
            {
                "last_run": _now(),
                "query_count": queries if queries >= 0 else prev_queries,
                "last_incentive": stats.get("incentive"),
                "trained_through": trained_through(),
                "latest_release": latest,
                "leaderboard": board,
            }
        )
        save_state(state)
    finally:
        release_lock()


if __name__ == "__main__":
    main()
