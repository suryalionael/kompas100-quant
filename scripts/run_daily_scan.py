"""Orchestrator: fetch -> validate -> features -> quality filters.

This session only wires up the data-layer stages (per COMPETITION_PLAN.md
§10, Days 1-2). Ranking, portfolio construction, and daily_brief.json
publishing are TODO for later sessions once ranking/ and portfolio/ exist.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

from data_pipeline import feature_builder, fetch_yfinance, quality_filters, validator

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_CSV = ROOT / "configs" / "kompas100_live.csv"
RAW_DIR = ROOT / "data" / "raw"
FEATURES_DIR = ROOT / "data" / "features"
PUBLISHED_DIR = ROOT / "data" / "published"


def write_scan_meta(fetched: dict, clean: dict, features_df: pd.DataFrame) -> None:
    """Record what this run actually did, for the dashboard's freshness panel.

    market_date is the newest date present in the successfully fetched OHLCV —
    on a non-trading day (weekend/holiday) yfinance returns no new rows, so
    this naturally reports the last real trading day instead of today.
    """
    market_date = None
    if not features_df.empty:
        market_date = pd.to_datetime(features_df["date"]).max().strftime("%Y-%m-%d")

    meta = {
        "scanned_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "market_date": market_date,
        "tickers_total": len(fetch_yfinance.load_universe(UNIVERSE_CSV)),
        "tickers_fetched": len(fetched),
        "tickers_validated": len(clean),
        "features_rows": len(features_df),
    }
    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    (PUBLISHED_DIR / "scan_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Scan metadata written: {meta}")


def main() -> None:
    logger.info("Fetching OHLCV for Kompas100 universe...")
    fetched = fetch_yfinance.fetch_universe(UNIVERSE_CSV, RAW_DIR)

    logger.info("Validating raw OHLCV...")
    clean, reports = validator.validate_batch(fetched)
    failed = [r["ticker"] for r in reports if r["ticker"] not in clean]
    if failed:
        logger.warning(f"{len(failed)} tickers failed validation: {failed}")

    logger.info("Building technical features...")
    features_df = feature_builder.build_features_batch(clean)
    if not features_df.empty:
        feature_builder.save_features(features_df, FEATURES_DIR)

    write_scan_meta(fetched, clean, features_df)

    logger.info(
        f"Done: fetched={len(fetched)} validated={len(clean)} "
        f"features_rows={len(features_df)}"
    )


if __name__ == "__main__":
    main()
