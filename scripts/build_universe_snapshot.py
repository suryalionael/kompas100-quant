"""Build the latest-per-ticker snapshot the dashboard reads: technical
features + fundamentals + quality-filter status, for every Kompas100 ticker.

Run after scripts/run_daily_scan.py has populated data/features/.
Fundamentals are fetched live from yfinance (rate-limited) and cached to
data/fundamentals/ so the dashboard itself never hits the network.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

from data_pipeline import fundamental, quality_filters
from data_pipeline.fetch_yfinance import to_yf_symbol

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"
FUNDAMENTALS_DIR = ROOT / "data" / "fundamentals"
PUBLISHED_DIR = ROOT / "data" / "published"


def latest_features_file() -> Path:
    files = sorted(FEATURES_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No feature files in {FEATURES_DIR} — run run_daily_scan.py first")
    return files[-1]


def main() -> None:
    scan_date = date.today().strftime("%Y-%m-%d")
    features_path = latest_features_file()
    logger.info(f"Loading features from {features_path}")
    features_df = pd.read_parquet(features_path)

    latest = features_df.sort_values("date").groupby("ticker", as_index=False).tail(1).reset_index(drop=True)

    fund_input = latest[["ticker"]].copy()
    fund_input["ticker"] = fund_input["ticker"].map(to_yf_symbol)

    logger.info(f"Fetching fundamentals for {len(fund_input)} tickers...")
    enriched = fundamental.enrich_with_fundamentals(
        fund_input,
        fundamentals_dir=FUNDAMENTALS_DIR,
        save=True,
        scan_date=scan_date,
    )
    enriched["ticker"] = enriched["ticker"].str.replace(".JK", "", regex=False)

    fundamental_cols = [c for c in fundamental.FUNDAMENTAL_COLS if c in enriched.columns]
    merged = latest.merge(enriched[["ticker", *fundamental_cols]], on="ticker", how="left")

    risk_overrides = quality_filters.load_risk_overrides(ROOT / "data" / "risk")
    snapshot = quality_filters.enrich_df_with_quality_filters(merged, risk_overrides=risk_overrides)

    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PUBLISHED_DIR / f"universe_snapshot_{scan_date}.parquet"
    snapshot.to_parquet(out_path, index=False)
    latest_link = PUBLISHED_DIR / "universe_snapshot_latest.parquet"
    snapshot.to_parquet(latest_link, index=False)
    logger.info(f"Universe snapshot saved → {out_path} ({len(snapshot)} tickers)")


if __name__ == "__main__":
    main()
