"""Runs the ranking-model horizon ablation (COMPETITION_PLAN.md §4) plus the
baseline levels of the ablation matrix (§9, Levels 0-2) that don't need a
trained model: random portfolio, Kompas100 buy-and-hold, naive momentum.

Level 3 (old scanner's rule-based signal) is intentionally NOT included —
signal_engine.py is "reference only" in COMPETITION_PLAN.md §1, its
fixed-weight formula is meant to be rewritten as stock_character.py (§5),
not ported and benchmarked as-is. Levels 5-7 (character features, research
layer, full portfolio construction) need modules that don't exist yet.

A level only ships if it beats the level below it on realized, cost-adjusted
portfolio return — not IC — per COMPETITION_PLAN.md's non-negotiables.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

from backtest import engine as bt
from ranking import ranking_model as rm

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "features"
PUBLISHED_DIR = ROOT / "data" / "published"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"

HORIZONS = [3, 5, 7, 10, 15]
TOP_K = 8
RETRAIN_EVERY = 10


def latest_features_path() -> Path:
    """Found as a real bug 2026-08-31: this used to be a hardcoded dated
    filename (data/features/2026-08-30.parquet), which silently went
    stale the moment a newer features file existed — every ablation run
    since was training on yesterday's features regardless of what
    scripts/run_daily_scan.py had actually produced since. Same
    latest-file pattern as scripts/dashboard.py's load_features_latest().
    """
    files = sorted(FEATURES_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No feature files in {FEATURES_DIR} — run scripts/run_daily_scan.py first")
    return files[-1]


def snapshot_paths(tag: str) -> tuple[Path, Path, Path]:
    """Returns (raw_dir, features_path, pit_csv) for a frozen snapshot —
    see scripts/freeze_snapshot.py. Used by --snapshot to make an ablation
    run reproducible against a fixed data pull, instead of whatever
    data/raw + data/features happen to contain right now (which yfinance's
    auto_adjust=True can silently revise between runs — see
    COMPETITION_PLAN.md §4's robustness-protocol note)."""
    snap_dir = SNAPSHOTS_DIR / tag
    manifest_path = snap_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No snapshot '{tag}' at {snap_dir} — run scripts/freeze_snapshot.py --tag {tag} first")
    manifest = json.loads(manifest_path.read_text())
    features_path = snap_dir / "features" / manifest["source_features_file"]
    return snap_dir / "raw", features_path, snap_dir / "kompas100_pit.csv"


def _fmt_pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x * 100:+.2f}%"


def print_report(results: dict) -> None:
    bh = results.get("buy_and_hold")
    if bh:
        print("\nLevel 1 — Kompas100 buy-and-hold (full window, real rebalance dates):")
        print(f"  compounded return: {_fmt_pct(bh.get('compounded_return', float('nan')))}, "
              f"max drawdown: {_fmt_pct(bh.get('max_drawdown', float('nan')))}, n periods: {bh.get('n_folds')}")

    print(f"\n{'Horizon':<8}{'Random':<12}{'Momentum':<12}{'Model':<12}{'Model IC':<10}{'Model beats momentum?':<22}")
    for horizon in sorted(results.get("horizons", {}), key=int):
        h = results["horizons"][horizon]
        r = h["random"]["overlapping"]["mean_return"]
        m = h["momentum"]["overlapping"]["mean_return"]
        rk = h["ranking_model"]["overlapping"]["mean_return"]
        ic = h["ranking_model"]["overlapping"]["mean_ic"]
        beats = "YES" if rk > m else "no"
        print(f"{horizon + 'D':<8}{_fmt_pct(r):<12}{_fmt_pct(m):<12}{_fmt_pct(rk):<12}{ic:<10.3f}{beats:<22}")


def run_one_horizon(horizon: int, panel: bt.PricePanel, features_df: pd.DataFrame, pit_df: pd.DataFrame) -> dict | None:
    """Runs a single horizon's comparison — split out so each can be invoked
    as its own short-lived process (see __main__ below): this environment
    kills long-running background processes unpredictably, so incremental,
    per-horizon checkpointing to disk is more reliable than one long run."""
    logger.info(f"=== Horizon {horizon}D ===")

    def naive_momentum_score(decision_date, universe, features_asof):
        latest = (
            features_asof[features_asof["ticker"].isin(universe)]
            .dropna(subset=["roc20"])
            .sort_values("date").groupby("ticker").tail(1)
        )
        return dict(zip(latest["ticker"], latest["roc20"]))

    train_ds = rm.build_training_dataset(features_df, panel, horizon, pit_df)
    score_fn = rm.make_ranking_score_fn(train_ds, panel, retrain_every=RETRAIN_EVERY)

    model_ov = bt.run_backtest(score_fn, panel, features_df, pit_df, horizon, TOP_K, spacing=1)
    model_no = bt.run_backtest(score_fn, panel, features_df, pit_df, horizon, TOP_K, spacing=horizon)

    if model_ov.empty:
        logger.warning(f"Horizon {horizon}D: ranking model produced no valid folds — skipping")
        return None
    valid_start = model_ov["decision_date"].min()

    random_ov = bt.run_random_benchmark(panel, pit_df, horizon, TOP_K, spacing=1)
    random_ov = random_ov[random_ov["decision_date"] >= valid_start]
    random_no = bt.run_random_benchmark(panel, pit_df, horizon, TOP_K, spacing=horizon)
    random_no = random_no[random_no["decision_date"] >= valid_start]

    momentum_ov = bt.run_backtest(naive_momentum_score, panel, features_df, pit_df, horizon, TOP_K, spacing=1)
    momentum_ov = momentum_ov[momentum_ov["decision_date"] >= valid_start]
    momentum_no = bt.run_backtest(naive_momentum_score, panel, features_df, pit_df, horizon, TOP_K, spacing=horizon)
    momentum_no = momentum_no[momentum_no["decision_date"] >= valid_start]

    horizon_result = {
        "valid_start": str(valid_start.date()),
        "n_folds_overlapping": len(model_ov),
        "n_folds_non_overlapping": len(model_no),
        "random": {"overlapping": bt.summarize(random_ov, False), "non_overlapping": bt.summarize(random_no, True)},
        "momentum": {"overlapping": bt.summarize(momentum_ov, False), "non_overlapping": bt.summarize(momentum_no, True)},
        "ranking_model": {"overlapping": bt.summarize(model_ov, False), "non_overlapping": bt.summarize(model_no, True)},
    }

    logger.info(
        f"{horizon}D — random: {_fmt_pct(horizon_result['random']['overlapping']['mean_return'])}/fold, "
        f"momentum: {_fmt_pct(horizon_result['momentum']['overlapping']['mean_return'])}/fold, "
        f"model: {_fmt_pct(horizon_result['ranking_model']['overlapping']['mean_return'])}/fold "
        f"(IC={horizon_result['ranking_model']['overlapping']['mean_ic']:.3f})"
    )
    return horizon_result


def _results_path(snapshot_tag: str | None) -> Path:
    if snapshot_tag:
        return PUBLISHED_DIR / f"ablation_results__{snapshot_tag}.json"
    return PUBLISHED_DIR / "ablation_results.json"


def _load_results(snapshot_tag: str | None = None) -> dict:
    out_path = _results_path(snapshot_tag)
    if out_path.exists():
        return json.loads(out_path.read_text())
    return {"buy_and_hold": None, "horizons": {}}


def _save_results(results: dict, snapshot_tag: str | None = None) -> None:
    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _results_path(snapshot_tag)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Results checkpointed → {out_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=None, help="Run only this horizon and checkpoint to disk")
    parser.add_argument("--buy-hold-only", action="store_true", help="Run only the Level 1 buy-and-hold benchmark")
    parser.add_argument("--report", action="store_true", help="Just print the report from whatever's checkpointed so far")
    parser.add_argument(
        "--snapshot", default=None,
        help="Run against a frozen snapshot (data/snapshots/<tag>, see scripts/freeze_snapshot.py) "
             "instead of live data/raw + data/features. Results go to a separate "
             "ablation_results__<tag>.json so they don't clobber the live-data results — "
             "compare two snapshot runs with scripts/compare_snapshots.py before trusting "
             "a 'beats momentum' verdict (COMPETITION_PLAN.md §4).",
    )
    args = parser.parse_args()

    results = _load_results(args.snapshot)

    if args.report:
        print_report(results)
        return

    if args.snapshot:
        raw_dir, features_path, pit_csv = snapshot_paths(args.snapshot)
        pit_df = bt.load_pit_universe(pit_csv)
        logger.info(f"Using snapshot '{args.snapshot}': raw={raw_dir}, features={features_path.name}")
    else:
        raw_dir = bt.RAW_DIR
        pit_df = bt.load_pit_universe()
        features_path = latest_features_path()
        logger.info(f"Using features file: {features_path}")

    all_tickers = sorted(pit_df["ticker"].unique())
    panel = bt.PricePanel(bt.load_price_panel(raw_dir, all_tickers))
    features_df = pd.read_parquet(features_path)

    if args.buy_hold_only or results.get("buy_and_hold") is None:
        logger.info("Running Level 1: Kompas100 buy-and-hold (full window, real PIT boundaries)...")
        buy_hold = bt.run_index_buy_and_hold(panel, pit_df)
        results["buy_and_hold"] = bt.summarize(buy_hold, sequential=True)
        _save_results(results, args.snapshot)
        if args.buy_hold_only:
            return

    if args.horizon is not None:
        horizon_result = run_one_horizon(args.horizon, panel, features_df, pit_df)
        if horizon_result is not None:
            results["horizons"][str(args.horizon)] = horizon_result
            _save_results(results, args.snapshot)
        return

    for horizon in HORIZONS:
        horizon_result = run_one_horizon(horizon, panel, features_df, pit_df)
        if horizon_result is not None:
            results["horizons"][str(horizon)] = horizon_result
            _save_results(results, args.snapshot)

    print_report(results)


if __name__ == "__main__":
    main()
