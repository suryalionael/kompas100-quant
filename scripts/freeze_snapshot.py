"""Freezes a timestamped, immutable copy of the current OHLCV + features
data for ablation robustness checking (COMPETITION_PLAN.md §4).

Why this exists: yfinance's auto_adjust=True retroactively revises
historical adjusted-close prices on every refetch. Found 2026-08-31 that
re-running the same horizon (3D) against two different live-refetched
data pulls gave visibly different ablation results — the same code, two
different answers, purely from data revision. That means no "horizon X
beats momentum" claim from a live/latest-file ablation run is trustworthy
on its own.

This script copies data/raw/*.parquet, the latest data/features/*.parquet,
and configs/kompas100_pit.csv into data/snapshots/<tag>/ with a manifest
(per-file SHA-256, row counts, source timestamp) so a later ablation run
against this exact snapshot is reproducible regardless of what live data
looks like by then. backtest/ablation.py --snapshot <tag> reads from here
instead of the live files.

Protocol (COMPETITION_PLAN.md §4): a horizon is only reported as "beats
momentum" once that verdict agrees across >= 2 snapshots frozen on
different calendar days — see scripts/compare_snapshots.py.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FEATURES_DIR = ROOT / "data" / "features"
PIT_CSV = ROOT / "configs" / "kompas100_pit.csv"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def freeze(tag: str | None = None) -> Path:
    now = datetime.now(timezone.utc)
    tag = tag or now.strftime("%Y%m%d_%H%M%SZ")
    dest = SNAPSHOTS_DIR / tag
    if dest.exists():
        raise FileExistsError(f"Snapshot {dest} already exists — pick a different tag")

    raw_files = sorted(RAW_DIR.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No raw parquet files in {RAW_DIR}")
    feature_files = sorted(FEATURES_DIR.glob("*.parquet"))
    if not feature_files:
        raise FileNotFoundError(f"No feature files in {FEATURES_DIR}")
    latest_features = feature_files[-1]

    (dest / "raw").mkdir(parents=True)
    (dest / "features").mkdir(parents=True)

    manifest = {
        "tag": tag,
        "frozen_at_utc": now.isoformat(),
        "source_features_file": latest_features.name,
        "raw_files": {},
    }

    for f in raw_files:
        out = dest / "raw" / f.name
        shutil.copy2(f, out)
        manifest["raw_files"][f.name] = _sha256(out)

    out_features = dest / "features" / latest_features.name
    shutil.copy2(latest_features, out_features)
    manifest["features_sha256"] = _sha256(out_features)

    shutil.copy2(PIT_CSV, dest / "kompas100_pit.csv")
    manifest["pit_sha256"] = _sha256(dest / "kompas100_pit.csv")
    manifest["n_raw_files"] = len(raw_files)

    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"Froze snapshot '{tag}' -> {dest} ({len(raw_files)} tickers, features={latest_features.name})")
    return dest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="Snapshot name (default: UTC timestamp)")
    args = parser.parse_args()
    freeze(args.tag)


if __name__ == "__main__":
    main()
