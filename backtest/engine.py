"""Walk-forward backtest simulator over the point-in-time Kompas100 universe.

Core design choices (see COMPETITION_PLAN.md §3):
  - Point-in-time universe only (configs/kompas100_pit.csv) — never
    kompas100_live.csv applied retroactively.
  - Entry at T+1 open, exit at close after `horizon` trading days.
  - IDX-typical costs applied multiplicatively on both legs.
  - Both overlapping (every trading day) and non-overlapping
    (every `horizon` days, independent) fold sets are reported.
  - A scoring function is only ever handed data up to and including the
    decision date — the engine slices it that way itself, so a caller
    cannot accidentally leak future data into a score.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PIT_CSV = ROOT / "configs" / "kompas100_pit.csv"

# IDX-typical placeholders per COMPETITION_PLAN.md §3, pending real
# platform figures from Phase 0.
DEFAULT_COST_BUY_BPS = 20.0
DEFAULT_COST_SELL_BPS = 30.0
DEFAULT_SLIPPAGE_BPS = 15.0
DEFAULT_TOP_K = 8

ScoreFn = Callable[[pd.Timestamp, list[str], pd.DataFrame], dict[str, float]]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_price_panel(raw_dir: Path | str, tickers: list[str]) -> pd.DataFrame:
    """Long-format date/ticker/open/close/volume panel for the given tickers."""
    raw_dir = Path(raw_dir)
    frames = []
    for ticker in tickers:
        path = raw_dir / f"{ticker}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)[["date", "ticker", "open", "close", "volume"]]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "close", "volume"])
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_pit_universe(pit_csv: Path | str = PIT_CSV) -> pd.DataFrame:
    df = pd.read_csv(pit_csv)
    df["period_start"] = pd.to_datetime(df["period_start"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    return df


def get_pit_universe(pit_df: pd.DataFrame, as_of_date: pd.Timestamp) -> list[str]:
    """Tickers active in the Kompas100 as of a given date, per the PIT calendar."""
    active = pit_df[
        (pit_df["period_start"] <= as_of_date)
        & (pit_df["period_end"] >= as_of_date)
        & (pit_df["is_active"])
    ]
    return sorted(active["ticker"].unique().tolist())


class PricePanel:
    """Wraps the long-format panel as wide date x ticker open/close frames —
    simple, unambiguous .loc[date, tickers] lookups instead of fragile
    MultiIndex partial indexing."""

    def __init__(self, long_panel: pd.DataFrame):
        self.open = long_panel.pivot(index="date", columns="ticker", values="open")
        self.close = long_panel.pivot(index="date", columns="ticker", values="close")
        self.calendar = pd.DatetimeIndex(sorted(long_panel["date"].unique()))

    def tradeable(self, tickers: list[str], entry_date: pd.Timestamp, exit_date: pd.Timestamp) -> list[str]:
        if entry_date not in self.open.index or exit_date not in self.close.index:
            return []
        entry_row = self.open.loc[entry_date]
        exit_row = self.close.loc[exit_date]
        return [t for t in tickers if t in entry_row.index and t in exit_row.index
                and pd.notna(entry_row[t]) and pd.notna(exit_row[t])]

    def forward_returns(self, tickers: list[str], entry_date: pd.Timestamp, exit_date: pd.Timestamp) -> pd.Series:
        entry_prices = self.open.loc[entry_date, tickers]
        exit_prices = self.close.loc[exit_date, tickers]
        return (exit_prices - entry_prices) / entry_prices


# ---------------------------------------------------------------------------
# Fold construction and simulation
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    decision_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    universe: list[str]


def build_folds(
    panel: PricePanel,
    pit_df: pd.DataFrame,
    horizon: int,
    spacing: int,
) -> list[Fold]:
    """decision_date -> entry at T+1 open -> exit at close after `horizon` days.

    spacing=1 gives overlapping folds (a decision every trading day);
    spacing=horizon gives independent, non-overlapping folds.
    """
    calendar = panel.calendar
    folds = []
    for i in range(0, len(calendar) - horizon - 1, spacing):
        decision_date = calendar[i]
        entry_date = calendar[i + 1]
        exit_date = calendar[i + 1 + horizon]
        universe = get_pit_universe(pit_df, decision_date)
        folds.append(Fold(decision_date, entry_date, exit_date, universe))
    return folds


def _cost_adjusted_return(
    raw_return: pd.Series,
    cost_buy_bps: float,
    cost_sell_bps: float,
    slippage_bps: float,
) -> pd.Series:
    entry_frac = (cost_buy_bps + slippage_bps) / 10_000
    exit_frac = (cost_sell_bps + slippage_bps) / 10_000
    return (1 + raw_return) * (1 - entry_frac) * (1 - exit_frac) - 1


def run_backtest(
    score_fn: ScoreFn | None,
    panel: PricePanel,
    features_df: pd.DataFrame,
    pit_df: pd.DataFrame,
    horizon: int,
    top_k: int = DEFAULT_TOP_K,
    spacing: int = 1,
    mode: str = "top_k",
    cost_buy_bps: float = DEFAULT_COST_BUY_BPS,
    cost_sell_bps: float = DEFAULT_COST_SELL_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    compute_ic: bool = True,
) -> pd.DataFrame:
    """Runs one fold set through one strategy. Returns a per-fold DataFrame.

    mode: "top_k" picks the top `top_k` scores from score_fn; "equal_weight_all"
    ignores score_fn and equal-weights the whole eligible universe, rebalanced
    every fold at the same cadence as an active strategy — useful for isolating
    stock-picking skill from turnover cost, but NOT the buy-and-hold benchmark
    (see run_index_buy_and_hold(), which rebalances only at real PIT boundaries).

    compute_ic=False skips the Spearman-correlation diagnostic entirely —
    set it for callers (e.g. the random benchmark's many trials) that never
    look at the ic column, since it isn't free at hundreds of folds x trials.
    """
    folds = build_folds(panel, pit_df, horizon, spacing)
    rows = []

    for fold in folds:
        tradeable = panel.tradeable(fold.universe, fold.entry_date, fold.exit_date)
        if not tradeable:
            continue

        if mode == "equal_weight_all":
            selected = tradeable
            ic = float("nan")
        else:
            features_asof = features_df[features_df["date"] <= fold.decision_date] if not features_df.empty else features_df
            scores = score_fn(fold.decision_date, tradeable, features_asof)
            scored = {t: s for t, s in scores.items() if t in tradeable and pd.notna(s)}
            if not scored:
                continue
            selected = [t for t, _ in sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
            ic = _fold_ic(panel, scored, fold.entry_date, fold.exit_date) if compute_ic else float("nan")

        net_returns = _cost_adjusted_return(
            panel.forward_returns(selected, fold.entry_date, fold.exit_date),
            cost_buy_bps, cost_sell_bps, slippage_bps,
        )

        rows.append({
            "decision_date": fold.decision_date,
            "entry_date": fold.entry_date,
            "exit_date": fold.exit_date,
            "n_universe": len(tradeable),
            "n_selected": len(selected),
            "portfolio_return": float(net_returns.mean()),
            "ic": ic,
        })

    return pd.DataFrame(rows)


def _fold_ic(panel: PricePanel, scored: dict[str, float], entry_date: pd.Timestamp, exit_date: pd.Timestamp) -> float:
    """Spearman rank correlation between scores and realized forward returns,
    diagnostic only per COMPETITION_PLAN.md §3 — never the pass/fail metric."""
    if len(scored) < 5:
        return float("nan")
    tickers = list(scored.keys())
    fwd_ret = panel.forward_returns(tickers, entry_date, exit_date)
    score_series = pd.Series(scored)
    ic = score_series.corr(fwd_ret.reindex(score_series.index), method="spearman")
    return float(ic) if pd.notna(ic) else float("nan")


def run_index_buy_and_hold(
    panel: PricePanel,
    pit_df: pd.DataFrame,
    cost_buy_bps: float = DEFAULT_COST_BUY_BPS,
    cost_sell_bps: float = DEFAULT_COST_SELL_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> pd.DataFrame:
    """True Kompas100 equal-weight buy-and-hold: rebalances only at actual
    PIT rebalance boundaries (~twice a year), not every fold. The `mode=
    "equal_weight_all"` path in run_backtest() rebalances every `horizon`
    days instead — useful for isolating stock-picking skill from turnover
    cost at the *same* cadence as an active strategy, but that is a
    different benchmark and must not be reported as "buy-and-hold": paying
    round-trip costs every 10 days is not what holding an index means.
    """
    calendar = panel.calendar
    periods = pit_df[["period_start", "period_end"]].drop_duplicates().sort_values("period_start")

    rows = []
    for _, period in periods.iterrows():
        after_start = calendar[calendar >= period["period_start"]]
        if len(after_start) < 2:
            continue
        decision_date = after_start[0]
        entry_idx = calendar.get_loc(decision_date) + 1
        if entry_idx >= len(calendar):
            continue
        entry_date = calendar[entry_idx]

        before_end = calendar[calendar <= period["period_end"]]
        if before_end.empty or before_end[-1] <= entry_date:
            continue
        exit_date = before_end[-1]

        universe = get_pit_universe(pit_df, decision_date)
        tradeable = panel.tradeable(universe, entry_date, exit_date)
        if not tradeable:
            continue

        net_returns = _cost_adjusted_return(
            panel.forward_returns(tradeable, entry_date, exit_date),
            cost_buy_bps, cost_sell_bps, slippage_bps,
        )
        rows.append({
            "decision_date": decision_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "n_universe": len(tradeable),
            "n_selected": len(tradeable),
            "portfolio_return": float(net_returns.mean()),
            "ic": float("nan"),
        })

    return pd.DataFrame(rows)


def run_random_benchmark(
    panel: PricePanel,
    pit_df: pd.DataFrame,
    horizon: int,
    top_k: int = DEFAULT_TOP_K,
    spacing: int = 1,
    n_trials: int = 100,
    seed: int = 42,
    cost_buy_bps: float = DEFAULT_COST_BUY_BPS,
    cost_sell_bps: float = DEFAULT_COST_SELL_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> pd.DataFrame:
    """Random-portfolio baseline: average of n_trials random top_k draws per fold.

    Builds the fold structure once — it's identical across trials, since it
    doesn't depend on any score — and for each fold computes every ticker's
    net return once, then just draws random top_k subsets and averages.
    (An earlier version re-ran the entire backtest machinery, including the
    unused IC calculation, once per trial — ~50x more work than necessary.)
    """
    folds = build_folds(panel, pit_df, horizon, spacing)
    rng = np.random.default_rng(seed)

    rows = []
    for fold in folds:
        tradeable = panel.tradeable(fold.universe, fold.entry_date, fold.exit_date)
        if len(tradeable) < top_k:
            continue

        net_returns = _cost_adjusted_return(
            panel.forward_returns(tradeable, fold.entry_date, fold.exit_date),
            cost_buy_bps, cost_sell_bps, slippage_bps,
        )

        trial_means = [
            float(net_returns.loc[rng.choice(tradeable, size=top_k, replace=False)].mean())
            for _ in range(n_trials)
        ]

        rows.append({
            "decision_date": fold.decision_date,
            "entry_date": fold.entry_date,
            "exit_date": fold.exit_date,
            "n_universe": len(tradeable),
            "n_selected": top_k,
            "portfolio_return": float(np.mean(trial_means)),
            "ic": float("nan"),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown of the compounded equity curve from a non-overlapping
    (independent) return sequence — meaningless on overlapping folds."""
    if returns.empty:
        return float("nan")
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def summarize(fold_returns: pd.DataFrame, sequential: bool, confidence: float = 0.95) -> dict:
    """sequential must be True only for a non-overlapping fold set — chaining
    overlapping folds' returns with cumprod() double-counts capital that's
    still tied up in the previous, not-yet-closed trade and produces a
    meaningless (wildly inflated) compounded return / drawdown."""
    if fold_returns.empty:
        return {"n_folds": 0}

    returns = fold_returns["portfolio_return"].dropna()
    n = len(returns)
    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if n > 1 else float("nan")
    se = std / np.sqrt(n) if n > 1 else float("nan")
    # Normal approximation for the CI — fine at n>=30; ablation folds target 50-70.
    z = 1.96 if confidence == 0.95 else 1.645
    ci_low, ci_high = (mean - z * se, mean + z * se) if n > 1 else (float("nan"), float("nan"))

    return {
        "n_folds": n,
        "mean_return": mean,
        "std_return": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean_ic": float(fold_returns["ic"].mean()) if "ic" in fold_returns else float("nan"),
        "compounded_return": float((1 + returns).prod() - 1) if sequential else float("nan"),
        "max_drawdown": max_drawdown(returns) if sequential else float("nan"),
    }


def summarize_strategy(
    panel: PricePanel, features_df: pd.DataFrame, pit_df: pd.DataFrame,
    score_fn: ScoreFn, horizon: int, top_k: int = DEFAULT_TOP_K, mode: str = "top_k",
) -> dict:
    """Convenience: run both overlapping and non-overlapping fold sets, summarize both.

    Only the non-overlapping set gets compounded_return/max_drawdown — see
    summarize()'s `sequential` note.
    """
    overlapping = run_backtest(score_fn, panel, features_df, pit_df, horizon, top_k, spacing=1, mode=mode)
    non_overlapping = run_backtest(score_fn, panel, features_df, pit_df, horizon, top_k, spacing=horizon, mode=mode)
    return {
        "overlapping": summarize(overlapping, sequential=False),
        "non_overlapping": summarize(non_overlapping, sequential=True),
    }
