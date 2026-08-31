"""Hand-computed fixtures for portfolio/level_calculator.py.

Every expected number here is worked out by hand in the comments, not
copied from the function's own output — this module feeds numbers an LLM
will narrate to a real user with real money implied, so "the code agrees
with itself" isn't good enough.

Run: python -m unittest tests.test_level_calculator -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from portfolio import level_calculator as lc


def _make_df(highs: list[float], start_date: str = "2026-01-01") -> pd.DataFrame:
    """Builds a minimal OHLCV frame from a list of daily highs, with
    low = high - 2 and close = high - 1 for every row — a constant
    2-point daily range with no gaps between days, chosen so True Range
    (and therefore ATR) comes out to exactly 2.0 by hand:

        TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)

    With high_t - low_t = 2 always, and close_{t-1} = high_{t-1} - 1:
      - trend UP by 1/day:   high_t - close_{t-1} = (high_{t-1}+1) - (high_{t-1}-1) = 2
                             low_t  - close_{t-1} = (high_t-2) - (high_{t-1}-1) = 0
                             -> TR = max(2, 2, 0) = 2
      - trend DOWN by 1/day: high_t - close_{t-1} = (high_{t-1}-1) - (high_{t-1}-1) = 0
                             low_t  - close_{t-1} = (high_t-2) - (high_{t-1}-1) = -2
                             -> TR = max(2, 0, 2) = 2
    Either way ATR14 (a plain rolling mean of a constant-2 series) = 2.0.
    """
    dates = pd.bdate_range(start=start_date, periods=len(highs))
    highs = pd.Series(highs, dtype=float)
    return pd.DataFrame({
        "date": dates,
        "open": highs - 1,
        "high": highs,
        "low": highs - 2,
        "close": highs - 1,
        "volume": [1_000_000] * len(highs),
    })


class TestBreakoutSetup(unittest.TestCase):
    """21 rows, high_t = 100 + t for t=0..20 (a steady uptrend).

    Rolling 20-day high/low is taken over the trailing 20 rows (index
    1..20): high ranges 101..120 so rolling_high = 120; low = high-2
    ranges 99..118 so rolling_low = 99.
    Last close (t=20) = high_20 - 1 = 120 - 1 = 119.

    Proximity to rolling_high: |119 - 120| / 120 * 100 = 0.833% <= 3%
    -> breakout. (Proximity to rolling_low would be far outside 3%, so
    there's no ambiguity between the two setup types here.)

    entry = 119, atr = 2.0
    stop  = 119 - 1.5*2.0 = 116.0
    risk  = 119 - 116.0 = 3.0
    target = 119 + 2.0*3.0 = 125.0
    rr_ratio = (125.0 - 119) / 3.0 = 2.0
    """

    def setUp(self):
        self.df = _make_df([100 + t for t in range(21)])

    def test_breakout_levels_match_hand_computation(self):
        result = lc.compute_levels_from_df(self.df, "TEST")
        self.assertIsNotNone(result)
        self.assertEqual(result["setup"], "breakout")
        self.assertEqual(result["entry"], 119.0)
        self.assertEqual(result["atr"], 2.0)
        self.assertEqual(result["stop"], 116.0)
        self.assertEqual(result["target"], 125.0)
        self.assertEqual(result["rr_ratio"], 2.0)

    def test_as_of_date_ignores_future_rows(self):
        """25 rows total, but as_of_date pins the calculation to the 21st
        row (t=20) — must reproduce the exact 21-row breakout result above,
        proving rows after as_of_date are never used (no look-ahead)."""
        df25 = _make_df([100 + t for t in range(25)])
        cutoff = df25["date"].iloc[20]
        result = lc.compute_levels_from_df(df25, "TEST", as_of_date=cutoff)
        self.assertIsNotNone(result)
        self.assertEqual(result["entry"], 119.0)
        self.assertEqual(result["stop"], 116.0)
        self.assertEqual(result["target"], 125.0)


class TestPullbackSetup(unittest.TestCase):
    """21 rows, high_t = 200 - t for t=0..20 (a steady downtrend).

    Rolling window (index 1..20): high ranges 180..199 (decreasing with t)
    so rolling_high = 199 (at t=1); low = high-2 ranges 178..197 so
    rolling_low = 178 (at t=20).
    Last close (t=20) = high_20 - 1 = 180 - 1 = 179.

    Proximity to rolling_low: |179 - 178| / 178 * 100 = 0.56% <= 3%
    -> pullback. (Proximity to rolling_high is 179 vs 199 = 10.05%, well
    outside 3%, so no ambiguity.)

    entry = 179, atr = 2.0
    stop  = 179 - 1.5*2.0 = 176.0
    risk  = 179 - 176.0 = 3.0
    target = 179 + 2.0*3.0 = 185.0
    rr_ratio = 2.0
    """

    def test_pullback_levels_match_hand_computation(self):
        df = _make_df([200 - t for t in range(21)])
        result = lc.compute_levels_from_df(df, "TEST")
        self.assertIsNotNone(result)
        self.assertEqual(result["setup"], "pullback")
        self.assertEqual(result["entry"], 179.0)
        self.assertEqual(result["atr"], 2.0)
        self.assertEqual(result["stop"], 176.0)
        self.assertEqual(result["target"], 185.0)
        self.assertEqual(result["rr_ratio"], 2.0)


class TestNoSetup(unittest.TestCase):
    def test_flat_price_in_middle_of_range_returns_none(self):
        """Flat market: high=105, low=95, close=100 for every row. Close
        sits dead in the middle — 4.76% from the rolling high and 5.26%
        from the rolling low, both outside the 3% proximity band — so
        there's no breakout or pullback setup and the function must not
        invent one."""
        dates = pd.bdate_range(start="2026-01-01", periods=21)
        df = pd.DataFrame({
            "date": dates, "open": 100.0, "high": 105.0, "low": 95.0,
            "close": 100.0, "volume": 1_000_000,
        })
        result = lc.compute_levels_from_df(df, "TEST")
        self.assertIsNone(result)

    def test_insufficient_history_returns_none(self):
        df = _make_df([100 + t for t in range(10)])  # needs 21 rows minimum
        result = lc.compute_levels_from_df(df, "TEST")
        self.assertIsNone(result)


class TestFileLoading(unittest.TestCase):
    def test_missing_ticker_file_returns_none(self):
        result = lc.compute_levels("DOES_NOT_EXIST", raw_dir="/tmp/nonexistent-dir-for-test")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
