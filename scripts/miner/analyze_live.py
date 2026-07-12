"""Measure the real benchmark->live distribution shift from logged live chunks.

Once the miner has logged enough real validator queries (poker44/model/live_log),
this compares live chunk features against the benchmark to reveal (a) the actual
chunk-size gap and (b) which features drift most benchmark->live — the features
that inflate our benchmark score but won't generalize. Run:

    python scripts/miner/analyze_live.py [--min-live 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker44.model.features import chunk_features  # noqa: E402
from poker44.model.live_log import DEFAULT_DIR  # noqa: E402
from scripts.miner.train_model import load_dataset, to_miner_view  # noqa: E402


def load_live(path: Path):
    hands_per_chunk, scores = [], []
    feats = []
    if not path.exists():
        return [], np.array([]), np.array([])
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        chunk = rec.get("hands") or []
        # Live chunks arrive already validator-viewed; score directly.
        feats.append(chunk_features(chunk))
        hands_per_chunk.append(rec.get("n_hands", len(chunk)))
        scores.append(rec.get("score", 0.0))
    names = sorted({k for f in feats for k in f})
    X = np.asarray([[f.get(k, 0.0) for k in names] for f in feats], dtype=float)
    return names, X, np.asarray(scores), np.asarray(hands_per_chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-live", type=int, default=200)
    ap.add_argument("--live-path", default=str(DEFAULT_DIR / "live_chunks.jsonl"))
    args = ap.parse_args()

    live_path = Path(args.live_path)
    lnames, Xl, scores, sizes = load_live(live_path)
    if len(Xl) < args.min_live:
        print(
            f"only {len(Xl)} live chunks logged (need >= {args.min_live}). "
            f"Let the miner run longer, then re-run."
        )
        if len(Xl):
            print(f"live chunk sizes so far: min={sizes.min()} max={sizes.max()} "
                  f"mean={sizes.mean():.1f}")
        return

    # Benchmark features (validator-viewed, same as training).
    Xb, yb, _, _, bnames = load_dataset(Path("data/benchmark"))

    print(f"live chunks: {len(Xl)} | sizes min={sizes.min()} max={sizes.max()} "
          f"mean={sizes.mean():.1f}")
    print(f"benchmark chunks: {len(Xb)} | live score mean={scores.mean():.3f} "
          f"(bot-rate@0.5={(scores>=0.5).mean():.3f})")

    common = [n for n in bnames if n in lnames]
    bi = {n: i for i, n in enumerate(bnames)}
    li = {n: i for i, n in enumerate(lnames)}
    mu_b = Xb.mean(0); sd_b = Xb.std(0) + 1e-9
    drift = {}
    for n in common:
        d = abs(Xl[:, li[n]].mean() - mu_b[bi[n]]) / sd_b[bi[n]]
        drift[n] = d
    ranked = sorted(drift.items(), key=lambda kv: -kv[1])
    print(f"\nTOP 20 benchmark->live DRIFTING features (standardized mean shift):")
    for n, d in ranked[:20]:
        print(f"  {n:34s} drift={d:.2f}  bench_mean={mu_b[bi[n]]:.3f} live_mean={Xl[:,li[n]].mean():.3f}")
    high = [n for n, d in ranked if d > 1.0]
    print(f"\n{len(high)} features drift > 1.0 sigma (candidates to drop/down-weight).")
    Path("data/live_drift.json").write_text(json.dumps(drift, indent=2))
    print("saved per-feature drift to data/live_drift.json")


if __name__ == "__main__":
    main()
