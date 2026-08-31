"""Orchestrator: fetch -> validate -> features -> quality filters -> daily brief.

This session wires up the data-layer stages (per COMPETITION_PLAN.md §10,
Days 1-2) plus daily_brief.json publishing (§7) via portfolio/daily_brief.py.
Ranking model training/backtesting and full portfolio construction (§6)
are still TODO — see portfolio/daily_brief.py's module docstring for why
it deliberately never emits a "validated" shortlist yet.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

from data_pipeline import feature_builder, fetch_yfinance, quality_filters, validator
from portfolio import daily_brief, rationale_log

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_CSV = ROOT / "configs" / "kompas100_live.csv"
RAW_DIR = ROOT / "data" / "raw"
FEATURES_DIR = ROOT / "data" / "features"
PUBLISHED_DIR = ROOT / "data" / "published"

# How many business days a ticker's own last-trade date can lag the
# market's (IHSG's) before it's flagged stale in the freshness audit —
# 1 tolerates IDX-wide holidays that only affect that name's own gap
# calculation slightly differently than IHSG's; anything more likely means
# a fetch problem specific to that ticker (suspension, delisting, a
# yfinance hiccup), not a market-wide non-trading day.
STALE_TICKER_TOLERANCE_DAYS = 1


def audit_ticker_freshness(fetched: dict[str, pd.DataFrame], ihsg: pd.DataFrame) -> dict:
    """Per-ticker actual last-trade date — from the fetched data itself,
    not assumed from 'last row in the parquet file'. Flags any ticker
    whose own last trade lags the market's last trading day by more than
    STALE_TICKER_TOLERANCE_DAYS, which the market-wide gap (weekends,
    IDX holidays) doesn't explain.
    """
    market_last_date = pd.to_datetime(ihsg["date"]).max() if not ihsg.empty else None
    per_ticker = {}
    stale = []
    for ticker, df in fetched.items():
        if df.empty:
            per_ticker[ticker] = None
            stale.append(ticker)
            continue
        last_date = pd.to_datetime(df["date"]).max()
        per_ticker[ticker] = str(last_date.date())
        if market_last_date is not None:
            lag_days = len(pd.bdate_range(last_date + pd.Timedelta(days=1), market_last_date)) \
                if last_date < market_last_date else 0
            if lag_days > STALE_TICKER_TOLERANCE_DAYS:
                stale.append(ticker)

    return {
        "market_last_trade_date": str(market_last_date.date()) if market_last_date is not None else None,
        "per_ticker_last_trade_date": per_ticker,
        "stale_tickers": sorted(stale),
    }


def write_scan_meta(fetched: dict, clean: dict, features_df: pd.DataFrame, freshness_audit: dict) -> None:
    """Record what this run actually did, for the dashboard's freshness panel.

    market_date is the newest date present in the successfully fetched OHLCV —
    on a non-trading day (weekend/holiday) yfinance returns no new rows, so
    this naturally reports the last real trading day instead of today.

    trigger comes from GITHUB_EVENT_NAME (set by GitHub Actions —
    "schedule" for the automated daily cron, "workflow_dispatch" for a
    manual run) or "local" outside CI. Kept distinct from a separate
    scheduled_run_state.json (updated only on trigger=="schedule") so the
    dashboard can tell "did the automation actually run on its own
    schedule" apart from "someone ran it by hand," which a single
    latest-run snapshot can't distinguish once a manual run overwrites it.
    """
    market_date = None
    if not features_df.empty:
        market_date = pd.to_datetime(features_df["date"]).max().strftime("%Y-%m-%d")

    trigger = os.environ.get("GITHUB_EVENT_NAME", "local")
    scanned_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    meta = {
        "scanned_at_utc": scanned_at_utc,
        "trigger": trigger,
        "market_date": market_date,
        "tickers_total": len(fetch_yfinance.load_universe(UNIVERSE_CSV)),
        "tickers_fetched": len(fetched),
        "tickers_validated": len(clean),
        "features_rows": len(features_df),
        "stale_tickers": freshness_audit["stale_tickers"],
    }
    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    (PUBLISHED_DIR / "scan_meta.json").write_text(json.dumps(meta, indent=2))
    (PUBLISHED_DIR / "data_freshness_audit.json").write_text(json.dumps(freshness_audit, indent=2))
    logger.info(f"Scan metadata written: {meta}")
    if freshness_audit["stale_tickers"]:
        logger.warning(
            f"{len(freshness_audit['stale_tickers'])} ticker(s) lag the market's last "
            f"trade date by more than {STALE_TICKER_TOLERANCE_DAYS}d: {freshness_audit['stale_tickers']}"
        )

    if trigger == "schedule":
        (PUBLISHED_DIR / "scheduled_run_state.json").write_text(
            json.dumps({"last_scheduled_run_utc": scanned_at_utc}, indent=2)
        )
        logger.info("Recorded successful scheduled run.")


def main() -> None:
    logger.info("Fetching OHLCV for Kompas100 universe...")
    fetched = fetch_yfinance.fetch_universe(UNIVERSE_CSV, RAW_DIR)

    logger.info("Fetching IHSG (^JKSE) for relative-strength features...")
    ihsg = fetch_yfinance.fetch_ihsg(RAW_DIR)

    logger.info("Validating raw OHLCV...")
    clean, reports = validator.validate_batch(fetched)
    failed = [r["ticker"] for r in reports if r["ticker"] not in clean]
    if failed:
        logger.warning(f"{len(failed)} tickers failed validation: {failed}")

    logger.info("Building technical features...")
    features_df = feature_builder.build_features_batch(clean, ihsg=ihsg if not ihsg.empty else None)
    if not features_df.empty:
        feature_builder.save_features(features_df, FEATURES_DIR)

    freshness_audit = audit_ticker_freshness(fetched, ihsg)
    write_scan_meta(fetched, clean, features_df, freshness_audit)

    logger.info("Building daily brief (shortlist + price levels)...")
    universe = fetch_yfinance.load_universe(UNIVERSE_CSV)
    brief = daily_brief.build_daily_brief(features_df, universe, raw_dir=RAW_DIR)
    daily_brief.save_daily_brief(brief)

    logger.info("Building rationale log (open/hold/close + technical/fundamental/macro/risk notes)...")
    rationale_entries = rationale_log.build_rationale_log(brief, features_df, raw_dir=RAW_DIR)
    if rationale_entries:
        entry_date = rationale_entries[0]["date"]
        rationale_log.save_rationale_log(entry_date, rationale_entries)
        rationale_log.save_manual_flow_template(entry_date, [e["ticker"] for e in rationale_entries])

    logger.info(
        f"Done: fetched={len(fetched)} validated={len(clean)} "
        f"features_rows={len(features_df)}"
    )


if __name__ == "__main__":
    main()
