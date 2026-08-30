"""Orchestrator: fetch -> validate -> features -> quality filters.

This session only wires up the data-layer stages (per COMPETITION_PLAN.md
§10, Days 1-2). Ranking, portfolio construction, and daily_brief.json
publishing are TODO for later sessions once ranking/ and portfolio/ exist.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

from data_pipeline import feature_builder, fetch_yfinance, fundamental, quality_filters, validator

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_CSV = ROOT / "configs" / "kompas100_live.csv"
RAW_DIR = ROOT / "data" / "raw"
FEATURES_DIR = ROOT / "data" / "features"


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

    logger.info(
        f"Done: fetched={len(fetched)} validated={len(clean)} "
        f"features_rows={len(features_df)}"
    )


if __name__ == "__main__":
    main()
