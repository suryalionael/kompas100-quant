"""Builds configs/kompas100_pit.csv, the point-in-time membership calendar
used by backtest/engine.py. configs/kompas100_live.csv is hand-maintained
and NOT touched here.

Verified periods (independently confirmed via BEI evaluation coverage,
2026-08-30 — see COMPETITION_PLAN.md §2):
    2026-08-03 -> 2027-01-29  current list (kompas100_live.csv)
    2026-02-02 -> 2026-08-02  current list with the Aug-2026 rebalance
                              reversed (9 out -> back in, 9 in -> back out)

Everything before 2026-02-02 falls back to the current list, flagged
source_verified=False — Kompas100 rebalances roughly every 6 months and
we have no sourced constituent data further back yet. Backtest folds in
that range are testing the ranking model's mechanics, not a faithful
historical universe; do not treat their results as production-grade
until real historical snapshots replace the fallback (see COMPETITION_PLAN.md
§8's IDX factsheet cross-check).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
LIVE_CSV = ROOT / "configs" / "kompas100_live.csv"
PIT_CSV = ROOT / "configs" / "kompas100_pit.csv"

# 2026-08-03 rebalance, independently confirmed against money.kompas.com's
# coverage of BEI's evaluation announcement (2026-07-28).
AUG_2026_IN = {"BFIN", "BIPI", "BNBR", "COIN", "EMAS", "GGRM", "LSIP", "MINA", "RMKE"}
AUG_2026_OUT = {"BREN", "BTPS", "DSSA", "FILM", "HMSP", "INTP", "MTEL", "SIDO", "TCPI"}

# Earliest date our raw OHLCV history reaches back to (3yr yfinance lookback).
HISTORY_START = "2023-08-31"


def main() -> None:
    current = pd.read_csv(LIVE_CSV)
    current_tickers = set(current[current["is_active"].astype(str).str.lower() == "true"]["ticker"])

    missing_in = AUG_2026_IN - current_tickers
    if missing_in:
        raise ValueError(f"Expected Aug-2026 'in' tickers missing from kompas100_live.csv: {missing_in}")

    prior_tickers = (current_tickers - AUG_2026_IN) | AUG_2026_OUT

    periods = [
        (HISTORY_START, "2026-02-01", sorted(current_tickers), False),
        ("2026-02-02", "2026-08-02", sorted(prior_tickers), True),
        ("2026-08-03", "2027-01-29", sorted(current_tickers), True),
    ]

    rows = []
    for period_start, period_end, tickers, verified in periods:
        for ticker in tickers:
            rows.append({
                "period_start": period_start,
                "period_end": period_end,
                "ticker": ticker,
                "is_active": True,
                "source_verified": verified,
            })

    pit_df = pd.DataFrame(rows)
    pit_df.to_csv(PIT_CSV, index=False)
    logger.info(
        f"Wrote {PIT_CSV} — {len(periods)} periods, "
        f"{sum(1 for p in periods if p[3])}/{len(periods)} verified"
    )


if __name__ == "__main__":
    main()
