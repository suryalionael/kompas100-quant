"""Builds data/published/daily_brief.json — the shortlist Cowork's daily
report will read (COMPETITION_PLAN.md §7). Quant is the only source of
truth for these numbers (CLAUDE.md); this module picks the shortlist and
attaches deterministic price levels (level_calculator.py), nothing more.

strategy_status tells any downstream reader (dashboard, Cowork) which
state actually produced the shortlist — never emit a shortlist, empty or
otherwise, without saying which of these three it is:

    "validated"              — an ablation-gated ranking model (§4) has
                                cleared its bar and its live output is the
                                real ranked shortlist.
    "naive_momentum_interim" — no horizon has cleared the gate yet
                                (current real state as of 2026-08-30, see
                                COMPETITION_PLAN.md §4/§9) — shortlist
                                falls back to the same trailing-momentum
                                rule the dashboard's "Current Best
                                Strategy" panel shows, clearly labeled
                                as interim, not the validated model.
    "no_picks"                — neither is available (e.g. a data-missing
                                day) — shortlist is empty and this field
                                says so explicitly.

Deliberately does not import ranking.ranking_model or backtest.engine:
ranking_model.py's functions build a full walk-forward training dataset
from a PricePanel, meant for backtesting, not a cheap "score today" call.
Wiring a real live-inference path for "validated" is a follow-up task for
whenever a horizon actually clears the gate — not something to bolt on
here as a side effect of a fallback-labeling task. Until that exists,
build_daily_brief() can never return "validated", by design.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

from portfolio import level_calculator

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_DIR = ROOT / "data" / "published"
ABLATION_RESULTS_PATH = PUBLISHED_DIR / "ablation_results.json"
UNIVERSE_SNAPSHOT_PATH = PUBLISHED_DIR / "universe_snapshot_latest.parquet"

DEFAULT_SHORTLIST_SIZE = 10

STATUS_VALIDATED = "validated"
STATUS_NAIVE_MOMENTUM_INTERIM = "naive_momentum_interim"
STATUS_NO_PICKS = "no_picks"

# ISTC 2026 rules: one virtual account, Rp100,000,000 starting capital,
# Kompas100 stocks only. Naive equal-weight sizing here is a placeholder —
# portfolio/portfolio_optimizer.py's real, ablation-tested sizing (§6:
# expected-return-weighted, per-name/sector caps) doesn't exist yet.
BASE_CAPITAL_IDR = 100_000_000
MAX_POSITION_PCT = 25.0


class Kompas100ViolationError(ValueError):
    """Raised if a non-Kompas100 ticker would be published — ISTC 2026
    disqualifies for trading outside Kompas100, so this must be a hard
    failure, never a silently-dropped row."""


def assert_kompas100_only(tickers: list[str], universe: list[str]) -> None:
    bad = sorted(set(tickers) - set(universe))
    if bad:
        raise Kompas100ViolationError(
            f"Refusing to publish — ticker(s) outside the live Kompas100 universe: {bad}"
        )


def _ablation_gate_cleared() -> bool:
    """True if at least one horizon's ranking model beats naive momentum
    on the non-overlapping (honest) fold set — the same check
    scripts/dashboard.py's Rankings tab uses. This tells you whether the
    *model* has cleared the bar, not whether a live-inference path exists
    for it (it doesn't yet — see module docstring)."""
    if not ABLATION_RESULTS_PATH.exists():
        return False
    results = json.loads(ABLATION_RESULTS_PATH.read_text())
    horizons = results.get("horizons", {})
    return any(
        h["ranking_model"]["non_overlapping"]["mean_return"] > h["momentum"]["non_overlapping"]["mean_return"]
        for h in horizons.values()
    )


def _naive_momentum_shortlist(features_df: pd.DataFrame, universe: list[str], top_n: int) -> pd.DataFrame:
    """Rank by trailing 20-day return — the exact Level 2 baseline
    backtest/ablation.py compares the ranking model against."""
    latest = (
        features_df[features_df["ticker"].isin(universe)]
        .dropna(subset=["roc20"])
        .sort_values("date")
        .groupby("ticker")
        .tail(1)
        .sort_values("roc20", ascending=False)
        .head(top_n)
    )
    return latest[["ticker", "roc20"]].rename(columns={"roc20": "score"})


def _empty_brief(status: str) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy_status": status,
        "base_capital_idr": BASE_CAPITAL_IDR,
        "shortlist": [],
    }


def _naive_position_size(n_positions: int) -> dict:
    """Equal-weight across the shortlist, capped at MAX_POSITION_PCT per
    name — a placeholder pending the real optimizer (§6), not itself
    ablation-tested."""
    if n_positions == 0:
        return {"position_pct": 0.0, "position_idr": 0.0}
    position_pct = min(100.0 / n_positions, MAX_POSITION_PCT)
    return {
        "position_pct": round(position_pct, 2),
        "position_idr": round(BASE_CAPITAL_IDR * position_pct / 100, 0),
    }


def _sector_lookup(universe: list[str]) -> dict[str, str]:
    if not UNIVERSE_SNAPSHOT_PATH.exists():
        return {}
    snapshot = pd.read_parquet(UNIVERSE_SNAPSHOT_PATH, columns=["ticker", "sector"])
    snapshot = snapshot[snapshot["ticker"].isin(universe)]
    return dict(zip(snapshot["ticker"], snapshot["sector"]))


def build_daily_brief(
    features_df: pd.DataFrame,
    universe: list[str],
    raw_dir: Path | str = level_calculator.RAW_DIR,
    top_n: int = DEFAULT_SHORTLIST_SIZE,
) -> dict:
    """Builds the daily_brief.json payload. Always sets strategy_status —
    never falls through to an empty or zero-conviction shortlist silently.
    """
    if _ablation_gate_cleared():
        logger.warning(
            "Ablation results show a horizon beating momentum, but no live "
            "ranking-model inference path is wired up yet (see "
            "portfolio/daily_brief.py's module docstring) — falling back to "
            f"'{STATUS_NAIVE_MOMENTUM_INTERIM}' rather than fabricating a "
            "live-model shortlist from an untested code path."
        )

    if features_df.empty or not universe:
        logger.warning("No features or universe available — daily_brief.json will report no_picks.")
        return _empty_brief(STATUS_NO_PICKS)

    momentum_df = _naive_momentum_shortlist(features_df, universe, top_n)
    if momentum_df.empty:
        logger.warning("Momentum ranking produced no candidates — daily_brief.json will report no_picks.")
        return _empty_brief(STATUS_NO_PICKS)

    tickers = momentum_df["ticker"].tolist()
    assert_kompas100_only(tickers, universe)  # hard guard — ISTC 2026 disqualifies for this

    sector_by_ticker = _sector_lookup(universe)
    sizing = _naive_position_size(len(tickers))

    shortlist = []
    for _, row in momentum_df.iterrows():
        ticker = row["ticker"]
        levels = level_calculator.compute_levels(ticker, raw_dir=raw_dir)
        shortlist.append({
            "ticker": ticker,
            "sector": sector_by_ticker.get(ticker, ""),
            "score": round(float(row["score"]), 2),
            "levels": levels,  # None if no clear breakout/pullback setup today
            "position_pct": sizing["position_pct"],
            "position_idr": sizing["position_idr"],
        })

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy_status": STATUS_NAIVE_MOMENTUM_INTERIM,
        "base_capital_idr": BASE_CAPITAL_IDR,
        "shortlist": shortlist,
    }


def save_daily_brief(brief: dict) -> None:
    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    path = PUBLISHED_DIR / "daily_brief.json"
    path.write_text(json.dumps(brief, indent=2, default=str))
    logger.info(
        f"Daily brief saved → {path} "
        f"(strategy_status={brief['strategy_status']}, {len(brief['shortlist'])} tickers)"
    )
