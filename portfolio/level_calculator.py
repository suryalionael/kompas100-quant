"""Deterministic buy/sell/stop price levels — pure OHLCV arithmetic, no ML,
no LLM call. This is what an LLM will later narrate to a real user with
real money implied (COMPETITION_PLAN.md §7's Cowork layer); it needs to be
boringly correct, not clever.

This project is long-only (COMPETITION_PLAN.md §0), so every setup below
is a long entry — the setup types just differ in which technical
condition qualifies it, not in direction:

    breakout  — close is within `proximity_pct` of the rolling
                `lookback_window`-day high (about to break resistance, or
                just broke it)
    pullback  — close is within `proximity_pct` of the rolling
                `lookback_window`-day low (bounced/holding at support)
    atr_band  — fallback when close is in neither zone (found 2026-08-31:
                a momentum-ranked shortlist is mostly names that have
                already run, so requiring fresh proximity to a 20-day
                extreme left 8 of 10 shortlisted names with no level at
                all — read by a teammate as "no risk here," which is
                false, not "no setup here"). Same entry/stop/target
                arithmetic, just without the breakout/pullback proximity
                gate. Always distinguishable from a real technical setup
                via the `setup` field — never silently presented as one.

compute_levels() only returns None when there's genuinely not enough
price history to compute ATR/rolling levels at all — never for "no clean
setup," which now falls through to atr_band instead.

ATR here is a simple rolling mean of True Range, not Wilder's smoothing
(which is what data_pipeline/feature_builder.py's atr14 column uses) —
deliberately, so a hand-computed fixture can verify it exactly in the unit
tests. Don't expect this module's atr to equal feature_builder's atr14 for
the same ticker/date; they're different, both legitimate, definitions.

proximity_pct=3.0 is a placeholder, like the backtest engine's cost
assumptions — pick a real value once there's a reason to prefer one number
over another, not before.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

DEFAULT_LOOKBACK_WINDOW = 20
DEFAULT_PROXIMITY_PCT = 3.0
DEFAULT_ATR_WINDOW = 14
DEFAULT_STOP_ATR_MULT = 1.5
DEFAULT_REWARD_RISK = 2.0


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def compute_levels(
    ticker: str,
    as_of_date: str | pd.Timestamp | None = None,
    raw_dir: Path | str = RAW_DIR,
    lookback_window: int = DEFAULT_LOOKBACK_WINDOW,
    proximity_pct: float = DEFAULT_PROXIMITY_PCT,
    atr_window: int = DEFAULT_ATR_WINDOW,
    stop_atr_mult: float = DEFAULT_STOP_ATR_MULT,
    reward_risk_ratio: float = DEFAULT_REWARD_RISK,
) -> dict | None:
    """Loads {raw_dir}/{ticker}.parquet and delegates to compute_levels_from_df."""
    path = Path(raw_dir) / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return compute_levels_from_df(
        df, ticker, as_of_date, lookback_window, proximity_pct,
        atr_window, stop_atr_mult, reward_risk_ratio,
    )


def compute_levels_from_df(
    df: pd.DataFrame,
    ticker: str,
    as_of_date: str | pd.Timestamp | None = None,
    lookback_window: int = DEFAULT_LOOKBACK_WINDOW,
    proximity_pct: float = DEFAULT_PROXIMITY_PCT,
    atr_window: int = DEFAULT_ATR_WINDOW,
    stop_atr_mult: float = DEFAULT_STOP_ATR_MULT,
    reward_risk_ratio: float = DEFAULT_REWARD_RISK,
) -> dict | None:
    """Same as compute_levels() but takes an already-loaded OHLCV DataFrame
    (needs date, high, low, close) — this is what the unit tests call
    directly with a fixed, hand-computable fixture.

    Returns None only when there's not enough price history to compute
    ATR/rolling levels at all. If close isn't near a rolling high or low,
    setup falls back to "atr_band" rather than returning no level.
    """
    df = df.sort_values("date").reset_index(drop=True)
    if as_of_date is not None:
        df = df[df["date"] <= pd.Timestamp(as_of_date)]

    if len(df) < max(lookback_window, atr_window) + 1:
        return None

    tr = _true_range(df)
    atr = tr.rolling(atr_window).mean().iloc[-1]
    if pd.isna(atr):
        return None

    rolling_high = df["high"].rolling(lookback_window).max().iloc[-1]
    rolling_low = df["low"].rolling(lookback_window).min().iloc[-1]
    close = float(df["close"].iloc[-1])
    resolved_date = df["date"].iloc[-1]

    setup = "atr_band"
    if rolling_high and abs(close - rolling_high) / rolling_high * 100 <= proximity_pct:
        setup = "breakout"
    elif rolling_low and abs(close - rolling_low) / rolling_low * 100 <= proximity_pct:
        setup = "pullback"

    entry = close
    stop = entry - stop_atr_mult * atr
    risk = entry - stop
    target = entry + reward_risk_ratio * risk
    reward = target - entry
    rr_ratio = reward / risk if risk else float("nan")

    return {
        "ticker": ticker,
        "as_of_date": str(pd.Timestamp(resolved_date).date()),
        "setup": setup,
        "entry": round(entry, 2),
        "atr": round(float(atr), 2),
        "stop": round(float(stop), 2),
        "target": round(float(target), 2),
        "rr_ratio": round(rr_ratio, 2),
    }
