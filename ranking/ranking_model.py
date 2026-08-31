"""Cross-sectional relative-return ranking model (COMPETITION_PLAN.md §4).

Target: forward `horizon`-day return, z-scored against the Kompas100
cross-section on that date. Features: feature_builder's technical set +
IHSG relative strength — all ratios/percentages/booleans, never raw price
levels (not cross-sectionally comparable) and never a composite rule-based
score (CLAUDE.md's circular-features ban).

Model is a from-scratch numpy Ridge regression, not scikit-learn — closed
form, no new heavy dependency, easy to audit. Walk-forward: at each
decision date, only training rows whose target was already resolvable by
that date are used to fit — see build_training_dataset()'s `resolved_date`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from loguru import logger

from backtest.engine import PricePanel, get_pit_universe

# Ratios/percentages/booleans only — no raw price levels (ma5/ma20/close/...),
# no composite rule-based score.
RANKING_FEATURES = [
    "rsi14", "stoch_rsi_k", "stoch_rsi_d",
    "adx", "adx_pos", "adx_neg",
    "roc5", "roc20",
    "pct_from_52w_high",
    "atr_pct", "bb_width", "hist_vol_20d",
    "vol_ratio_20d",
    "price_vs_ma200", "price_vs_vwap",
    "rel_strength_5d", "rel_strength_20d",
    "ma_full_alignment", "ma_partial_alignment", "golden_cross",
    "supertrend_bullish", "squeeze_on", "squeeze_release",
    "atr_breakout", "vol_spike", "obv_trend",
]

# One plain-English sentence per feature — a non-quant teammate should be
# able to read this and recognize a real market behavior, not just a
# column name. Surfaced in the dashboard's Rankings tab (COMPETITION_PLAN.md
# §0's Final Stage note: a model nobody on the team can explain is a
# liability even if it backtests well). Keep every RANKING_FEATURES entry
# covered here — the dashboard flags any that drift out of sync.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "rsi14": "14-day Relative Strength Index — how overbought or oversold the stock is versus its own recent swings",
    "stoch_rsi_k": "Stochastic RSI (%K) — how extreme RSI itself is versus its own recent range; faster and more sensitive than RSI",
    "stoch_rsi_d": "Stochastic RSI (%D) — a smoothed version of Stochastic RSI %K",
    "adx": "Average Directional Index — how strong the current trend is, regardless of direction",
    "adx_pos": "+DI — strength of upward price movement (the bullish half of ADX)",
    "adx_neg": "-DI — strength of downward price movement (the bearish half of ADX)",
    "roc3": "3-day price momentum — % return over the last 3 trading days",
    "roc5": "5-day price momentum — % return over the last 5 trading days",
    "roc20": "20-day price momentum — % return over the last 20 trading days",
    "sharpe_mom_20d": "20-day return per unit of 20-day volatility — an efficient trend, not just a noisy/volatile one",
    "mom_vol_confirmed_20d": "20-day momentum weighted by volume activity — a move backed by real trading interest, not thin volume",
    "pct_from_52w_high": "% below the 52-week high — how far the stock has pulled back from its own recent peak",
    "atr_pct": "Average True Range as a % of price — typical daily move size, scaled so it's comparable across stocks",
    "bb_width": "Bollinger Band width — how compressed or expanded the stock's recent volatility band is",
    "hist_vol_20d": "20-day historical volatility (annualized) — how choppy the stock has been recently",
    "vol_ratio_20d": "Today's volume vs. its own 20-day average — is trading activity unusually high or low",
    "price_vs_ma200": "% above/below the 200-day moving average — long-term trend position",
    "price_vs_vwap": "% above/below the 20-day volume-weighted average price — short-term fair-value position",
    "rel_strength_5d": "5-day return relative to IHSG — outperforming or underperforming the broader market",
    "rel_strength_20d": "20-day return relative to IHSG — same comparison over a longer window",
    "sector_rel_strength_20d": "20-day return relative to same-sector peers — outperforming its own industry, not just the whole market",
    "ma_full_alignment": "MA20 > MA50 > MA200 all stacked bullishly — a textbook uptrend structure",
    "ma_partial_alignment": "MA20 > MA50 — a shorter-term uptrend signal, weaker than full alignment",
    "golden_cross": "MA50 just crossed above MA200 — a classic, if lagging, bullish trend-change signal",
    "supertrend_bullish": "Supertrend indicator currently reads bullish (price above its trailing stop line)",
    "squeeze_on": "Bollinger Bands are inside the Keltner Channel — volatility is compressed, often precedes a big move",
    "squeeze_release": "The volatility squeeze just ended — a big move may be starting",
    "atr_breakout": "Today's move exceeded 1.5x yesterday's ATR — an unusually large single-day price swing",
    "vol_spike": "Volume is more than 2.5x its 20-day average — unusually heavy trading interest",
    "obv_trend": "On-Balance Volume has trended up over the last 10 days — buying pressure accumulating",
}

RIDGE_LAMBDA = 1.0
MIN_TRAIN_ROWS = 200
MIN_CROSS_SECTION = 10


def _to_numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].astype(float).values


def build_training_dataset(features_df: pd.DataFrame, panel: PricePanel, horizon: int, pit_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, ticker) with RANKING_FEATURES + a resolved,
    cross-sectionally z-scored forward-`horizon`-day-return target.

    resolved_date is when that target became knowable (entry + horizon
    trading days later) — the walk-forward trainer only ever uses rows
    with resolved_date <= the current decision date.

    pit_df is required, not optional: found 2026-08-31 that this function
    was z-scoring the target against *every ticker with a feature row on
    that date* — including several fetched only because they were in the
    current Kompas100 list, not because they belonged in that historical
    date's cross-section — rather than backtest.engine.get_pit_universe()'s
    actual point-in-time membership. That's the exact thing
    CLAUDE.md's non-negotiables ban ("kompas100_pit.csv, never
    kompas100_live.csv applied retroactively") and it's a real train/
    inference mismatch: make_ranking_score_fn() scores against the correct
    PIT universe at inference time, so training against a different,
    wrong cross-section adds pure noise to the learned target.
    """
    calendar = panel.calendar
    date_to_idx = {d: i for i, d in enumerate(calendar)}
    feature_cols = [c for c in RANKING_FEATURES if c in features_df.columns]

    rows = []
    for date, group in features_df.groupby("date"):
        if date not in date_to_idx:
            continue
        i = date_to_idx[date]
        if i + 1 + horizon >= len(calendar):
            continue
        entry_date = calendar[i + 1]
        exit_date = calendar[i + 1 + horizon]

        pit_universe = set(get_pit_universe(pit_df, date))
        group = group[group["ticker"].isin(pit_universe)]
        group = group.dropna(subset=feature_cols)
        tickers = group["ticker"].tolist()
        tradeable = panel.tradeable(tickers, entry_date, exit_date)
        if len(tradeable) < MIN_CROSS_SECTION:
            continue

        fwd_ret = panel.forward_returns(tradeable, entry_date, exit_date)
        std = fwd_ret.std(ddof=0)
        if not std or pd.isna(std):
            continue
        z = (fwd_ret - fwd_ret.mean()) / std

        sub = group[group["ticker"].isin(tradeable)].set_index("ticker")
        block = sub.loc[tradeable, feature_cols].copy()
        block["date"] = date
        block["ticker"] = tradeable
        block["target"] = z.reindex(tradeable).values
        block["resolved_date"] = exit_date
        rows.append(block.reset_index(drop=True))

    if not rows:
        return pd.DataFrame(columns=[*feature_cols, "date", "ticker", "target", "resolved_date"])
    return pd.concat(rows, ignore_index=True)


def _fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    n, p = X.shape
    X_aug = np.hstack([np.ones((n, 1)), X])
    reg = np.eye(p + 1) * lam
    reg[0, 0] = 0.0  # don't regularize the intercept
    beta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
    return beta


def _predict_ridge(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_aug @ beta


def make_ranking_score_fn(
    training_dataset: pd.DataFrame,
    panel: PricePanel,
    retrain_every: int = 10,
    lam: float = RIDGE_LAMBDA,
):
    """Returns a score_fn(decision_date, universe, features_asof) for
    backtest.engine.run_backtest — refits a fresh Ridge model periodically
    (every `retrain_every` trading days) using only training rows resolved
    by that point, so no future information ever leaks into a score.
    """
    feature_cols = [c for c in RANKING_FEATURES if c in training_dataset.columns]
    calendar = panel.calendar
    date_to_idx = {d: i for i, d in enumerate(calendar)}
    cache: dict[int, tuple | None] = {}

    def _checkpoint(decision_date: pd.Timestamp) -> int:
        idx = date_to_idx.get(decision_date)
        if idx is None:
            idx = int(np.searchsorted(calendar.values, decision_date.to_datetime64()))
        return idx // retrain_every

    def _get_model(decision_date: pd.Timestamp):
        key = _checkpoint(decision_date)
        if key in cache:
            return cache[key]

        train = training_dataset[training_dataset["resolved_date"] <= decision_date]
        if len(train) < MIN_TRAIN_ROWS:
            cache[key] = None
            return None

        X = _to_numeric_matrix(train, feature_cols)
        y = train["target"].astype(float).values
        mean, std = X.mean(axis=0), X.std(axis=0)
        std_safe = np.where(std == 0, 1.0, std)
        Xz = (X - mean) / std_safe
        beta = _fit_ridge(Xz, y, lam=lam)
        cache[key] = (beta, mean, std_safe)
        return cache[key]

    def score_fn(decision_date: pd.Timestamp, universe: list[str], features_asof: pd.DataFrame) -> dict[str, float]:
        model = _get_model(decision_date)
        if model is None:
            return {}
        beta, mean, std_safe = model

        latest = (
            features_asof[features_asof["ticker"].isin(universe)]
            .dropna(subset=feature_cols)
            .sort_values("date")
            .groupby("ticker")
            .tail(1)
            .set_index("ticker")
        )
        if latest.empty:
            return {}

        X_pred = _to_numeric_matrix(latest, feature_cols)
        Xz_pred = (X_pred - mean) / std_safe
        preds = _predict_ridge(Xz_pred, beta)
        return dict(zip(latest.index, preds))

    return score_fn
