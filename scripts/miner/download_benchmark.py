"""Download and cache all Poker44 training benchmark releases.

Usage:
    python scripts/miner/download_benchmark.py [--data-dir data/benchmark]

Caches one JSON file per API chunk, keyed by chunkHash, plus a per-date
manifest. Re-running only fetches what is missing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

BASE_URL = "https://api.poker44.net/api/v1/benchmark"


def fetch_json(url: str, params: dict | None = None, retries: int = 5) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            body = resp.json()
            if not body.get("success", True):
                raise RuntimeError(f"API error: {body}")
            return body["data"]
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def list_release_dates() -> list[str]:
    dates: list[str] = []
    before = None
    while True:
        params = {"limit": 100}
        if before:
            params["before"] = before
        data = fetch_json(f"{BASE_URL}/releases", params)
        releases = data.get("releases", [])
        if not releases:
            break
        dates.extend(r["sourceDate"] for r in releases)
        if len(releases) < 100:
            break
        before = releases[-1]["sourceDate"]
    return sorted(set(dates))


def download_date(source_date: str, data_dir: Path) -> int:
    date_dir = data_dir / source_date
    manifest_path = date_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all((date_dir / f"{h}.json").exists() for h in manifest["chunk_hashes"]):
            return len(manifest["chunk_hashes"])

    date_dir.mkdir(parents=True, exist_ok=True)
    hashes: list[str] = []
    cursor = None
    while True:
        params = {"sourceDate": source_date, "limit": 24}
        if cursor:
            params["cursor"] = cursor
        data = fetch_json(f"{BASE_URL}/chunks", params)
        for chunk in data.get("chunks", []):
            chunk_hash = chunk["chunkHash"]
            hashes.append(chunk_hash)
            out = date_dir / f"{chunk_hash}.json"
            if not out.exists():
                out.write_text(json.dumps(chunk))
        cursor = data.get("nextCursor")
        if not cursor:
            break

    manifest_path.write_text(
        json.dumps({"sourceDate": source_date, "chunk_hashes": hashes})
    )
    return len(hashes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/benchmark")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    status = fetch_json(BASE_URL)
    print(f"latest release: {status['latestSourceDate']} ({status['releaseVersion']})")

    dates = list_release_dates()
    print(f"{len(dates)} release dates")
    total = 0
    for source_date in dates:
        count = download_date(source_date, data_dir)
        total += count
        print(f"  {source_date}: {count} api chunks")
    print(f"done: {total} api chunks cached in {data_dir}")


if __name__ == "__main__":
    main()
