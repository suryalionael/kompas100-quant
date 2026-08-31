"""Cheap OHLCV-derived proxies for possible money-flow behavior — volume/
price divergence patterns a trader would call "absorption" or "weak hands"
in bandarmology terms, computed here as plain arithmetic.

**These are proxies, not real flow data.** No free, reliable, automatable
source for actual foreign/institutional flow exists (COMPETITION_PLAN.md
§1's audit of `foreign_flow.py`/`broker_analytics.py` already established
this, and that conclusion doesn't change here). These signals are talking
points for the Final Stage pitch's Money Flow Analysis (20% of that score)
— never fed into ranking/ranking_model.py as a feature, never presented as
if they were real broker/foreign-flow data.

Deliberately kept out of data_pipeline/feature_builder.py: that module's
output feeds ranking/ranking_model.py's RANKING_FEATURES directly, and
these proxies must never end up there by accident.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

DEFAULT_VOLUME_WINDOW = 20
DEFAULT_SPIKE_ZSCORE = 2.0


def compute_money_flow_proxies(df: pd.DataFrame, window: int = DEFAULT_VOLUME_WINDOW) -> dict | None:
    """df: OHLCV sorted or unsorted, needs date/close/volume, at least
    `window`+2 rows. Returns None if there's not enough history.

    volume_spike_no_followthrough: today's volume is an outlier vs. its own
        trailing distribution (z-score >= threshold) but price barely moved
        — reads as "someone moved a lot of size without pushing price,"
        the classic absorption/distribution pattern.
    price_up_declining_volume: price rose today while volume fell below its
        trailing average — reads as "the move isn't backed by participation,"
        a common "weak hands" / low-conviction-rally read.
    volume_zscore: raw diagnostic — how unusual today's volume is vs. the
        stock's own trailing `window`-day distribution.
    """
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < window + 2:
        return None

    vol = df["volume"]
    vol_window = vol.iloc[-(window + 1):-1]  # trailing window, excludes today
    vol_mean = vol_window.mean()
    vol_std = vol_window.std(ddof=0)

    today_vol = float(vol.iloc[-1])
    zscore = (today_vol - vol_mean) / vol_std if vol_std else 0.0

    close = df["close"]
    today_ret_pct = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100 if close.iloc[-2] else 0.0

    volume_spike_no_followthrough = bool(zscore >= DEFAULT_SPIKE_ZSCORE and abs(today_ret_pct) < 1.0)
    price_up_declining_volume = bool(today_ret_pct > 0 and vol_mean and today_vol < vol_mean)

    return {
        "date": str(pd.Timestamp(df["date"].iloc[-1]).date()),
        "volume_zscore": round(float(zscore), 2),
        "today_return_pct": round(float(today_ret_pct), 2),
        "volume_spike_no_followthrough": volume_spike_no_followthrough,
        "price_up_declining_volume": price_up_declining_volume,
    }
