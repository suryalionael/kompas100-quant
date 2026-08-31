"""Deterministic market regime classifier — no ML, no LLM (COMPETITION_PLAN.md
§6). Five states from Strong Bear to Strong Bull, built from four IHSG/
Kompas100-universe signals: trend, momentum, breadth, volume regime.
Historical volatility is reported alongside the state as a diagnostic, not
folded into the score — deciding how vol should bend the classification is
an empirical question for the ablation this module hasn't had yet (see
below), not something to guess at.

**Not yet ablation-tested** for whether conditioning ranking/sizing on this
state actually improves realized return, per COMPETITION_PLAN.md §6's own
requirement ("Ablation-test: does conditioning ranking/sizing on regime
improve realized return?"). This module exists right now to produce a real,
loggable macro state for the rationale log (COMPETITION_PLAN.md's ISTC 2026
prompt, Priority 2/5) — it is not wired into ranking_model.py or
backtest/engine.py, and shouldn't be until that ablation is actually run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

STRONG_BULL = "Strong Bull"
BULL = "Bull"
NEUTRAL = "Neutral"
BEAR = "Bear"
STRONG_BEAR = "Strong Bear"


def _trend_component(ihsg: pd.DataFrame) -> int:
    """+2 full bullish alignment (close>MA50>MA200) .. -2 full bearish."""
    close = ihsg["close"]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
    last = close.iloc[-1]

    score = 0
    if pd.notna(ma50):
        score += 1 if last > ma50 else -1
    if pd.notna(ma50) and pd.notna(ma200):
        score += 1 if ma50 > ma200 else -1
    return score


def _momentum_component(ihsg: pd.DataFrame, window: int = 20, threshold_pct: float = 3.0) -> int:
    close = ihsg["close"]
    if len(close) <= window:
        return 0
    ret = (close.iloc[-1] - close.iloc[-1 - window]) / close.iloc[-1 - window] * 100
    if ret > threshold_pct:
        return 1
    if ret < -threshold_pct:
        return -1
    return 0


def _breadth_component(features_df: pd.DataFrame, as_of_date: pd.Timestamp, bull_pct: float = 60.0, bear_pct: float = 40.0) -> int:
    """% of the universe trading above its own 20-day MA, as of the latest
    date on/before as_of_date."""
    latest = (
        features_df[features_df["date"] <= as_of_date]
        .sort_values("date")
        .groupby("ticker")
        .tail(1)
    )
    valid = latest.dropna(subset=["close", "ma20"])
    if valid.empty:
        return 0
    pct_above = (valid["close"] > valid["ma20"]).mean() * 100
    if pct_above >= bull_pct:
        return 1
    if pct_above <= bear_pct:
        return -1
    return 0


def _volume_component(ihsg: pd.DataFrame, window: int = 20, spike_mult: float = 1.5) -> int:
    """+1 = volume-confirmed up move, -1 = volume-confirmed down move."""
    if len(ihsg) <= window:
        return 0
    vol = ihsg["volume"]
    vol_avg = vol.rolling(window).mean().iloc[-1]
    if pd.isna(vol_avg) or vol_avg == 0:
        return 0
    last_vol = vol.iloc[-1]
    daily_ret = ihsg["close"].iloc[-1] - ihsg["close"].iloc[-2]
    if last_vol > spike_mult * vol_avg:
        return 1 if daily_ret > 0 else -1
    return 0


def _annualized_volatility(ihsg: pd.DataFrame, window: int = 20) -> float:
    close = ihsg["close"]
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(window).std().iloc[-1]
    return float(vol * np.sqrt(252) * 100) if pd.notna(vol) else float("nan")


def _score_to_state(score: int) -> str:
    if score >= 3:
        return STRONG_BULL
    if score >= 1:
        return BULL
    if score == 0:
        return NEUTRAL
    if score >= -2:
        return BEAR
    return STRONG_BEAR


def classify_regime(
    ihsg: pd.DataFrame,
    features_df: pd.DataFrame | None = None,
    as_of_date: pd.Timestamp | None = None,
) -> dict:
    """Returns the regime state plus every component that fed it — always
    return the components, not just the label, so the rationale log's
    macro_notes can say *why*, not just *what*.
    """
    ihsg = ihsg.sort_values("date").reset_index(drop=True)
    if as_of_date is not None:
        ihsg = ihsg[ihsg["date"] <= pd.Timestamp(as_of_date)]
    if ihsg.empty:
        return {"state": NEUTRAL, "score": 0, "reason": "no IHSG data available"}

    resolved_date = ihsg["date"].iloc[-1]

    trend = _trend_component(ihsg)
    momentum = _momentum_component(ihsg)
    volume = _volume_component(ihsg)
    breadth = 0
    if features_df is not None and not features_df.empty:
        breadth = _breadth_component(features_df, resolved_date)

    score = trend + momentum + breadth + volume
    state = _score_to_state(score)
    volatility_pct = _annualized_volatility(ihsg)

    return {
        "date": str(pd.Timestamp(resolved_date).date()),
        "state": state,
        "score": score,
        "trend_component": trend,
        "momentum_component": momentum,
        "breadth_component": breadth,
        "volume_component": volume,
        "annualized_volatility_pct": round(volatility_pct, 2) if pd.notna(volatility_pct) else None,
        "ihsg_close": round(float(ihsg["close"].iloc[-1]), 2),
    }


def classify_regime_from_files(
    raw_dir: Path | str = RAW_DIR,
    features_df: pd.DataFrame | None = None,
    as_of_date: pd.Timestamp | None = None,
) -> dict:
    path = Path(raw_dir) / "IHSG.parquet"
    if not path.exists():
        return {"state": NEUTRAL, "score": 0, "reason": "IHSG.parquet not found"}
    ihsg = pd.read_parquet(path)
    return classify_regime(ihsg, features_df, as_of_date)
