# CLAUDE.md — kompas100-quant

## Role

Python coding assistant for a competition-specific stock ranking and
portfolio system for the Kompas100 (Indonesian large-cap index) trading
competition, mid-September 2026. This is a **new, standalone project** —
not a fork or in-place upgrade of `idx-stock-scanner-agent`. That repo is
a reference and a parts bin: a handful of its modules get ported in
because they're already solid, everything else here is built fresh,
scoped specifically to this competition.

Full build spec, phase checklist, and ablation gates live in
`COMPETITION_PLAN.md` — read that before starting any task. This file is
the standing behavior/style contract.

## Relationship to idx-stock-scanner-agent

Treat the old repo (`https://github.com/suryalionael/idx-stock-scanner-agent`)
as a **source to port specific files from, not a codebase to build on top
of.** Never `import` from it live or add it as a dependency/submodule —
copy the specific module in, adapt it to this project's needs, and own it
here. See `COMPETITION_PLAN.md` §1 for exactly which files get ported
as-is, which get ported-and-adapted, and which are deliberately left
behind.

## Project Layout

```
configs/
  kompas100_live.csv       → current 100 constituents (live/paper trading)
  kompas100_pit.csv        → point-in-time membership calendar (backtest only)
  model_config.yaml        → ranking model hyperparams + feature list
data_pipeline/
  fetch_yfinance.py        → PORTED from old repo, trimmed to Kompas100 scope
  validator.py             → PORTED as-is
  feature_builder.py       → PORTED as-is, extended with IHSG relative strength
  news_sentiment.py        → PORTED as-is
  fundamental.py           → PORTED as-is, extended to capture sector/industry from the
                             same yfinance .info call (full 100/100 coverage, replacing
                             the old repo's issuers.csv which only covered 52/100)
  money_flow_proxies.py    → NEW: OHLCV-derived volume/price divergence proxies — Final
                             Stage pitch talking points only, never a ranking feature
ranking/
  ranking_model.py         → NEW: cross-sectional relative-return regressor, per horizon
  stock_character.py       → NEW: rolling per-stock behavioral features + personalized screener
  regime_classifier.py     → NEW: deterministic market regime, no LLM. Not yet wired into
                             ranking/sizing or ablation-tested — exists to produce a
                             loggable macro state for the rationale log
portfolio/
  portfolio_optimizer.py   → NEW: expected-return-weighted sizing, sector/name caps
  competition_strategist.py → NEW: chase/defend/neutral as a function of leaderboard state
  level_calculator.py      → NEW: deterministic buy/sell/stop levels (ATR stop, R:R target),
                             modeled on old repo's alerts/level_calculator.py — feeds
                             daily_brief.json; the Cowork report explains these numbers,
                             never invents or adjusts them
  daily_brief.py           → NEW: builds daily_brief.json — strategy_status
                             (validated/naive_momentum_interim/no_picks), real Rp100M-based
                             position sizing, hard Kompas100-only guard (raises, never
                             silently drops a bad ticker)
  rationale_log.py         → NEW: data/published/rationale_log/{date}.json — day-over-day
                             open/hold/close diff with auto-populated technical/
                             fundamental/macro/risk notes; money_flow_notes is the one
                             field left for the team's own manual research
backtest/
  engine.py                → NEW: walk-forward simulator, point-in-time universe, benchmarks, costs
  ablation.py               → NEW: runs the Level 0-7 experiment matrix, logs results
scripts/
  run_daily_scan.py         → NEW orchestrator: fetch → features → daily brief → rationale log
  build_universe.py         → NEW: builds/updates kompas100_live.csv and kompas100_pit.csv
  build_pitch_deck_source.py → NEW: concatenates rationale_log/ into one Markdown timeline
                             for the ISTC 2026 Final Stage pitch — raw material, not slides
data/
  raw/                      → per-ticker OHLCV parquet (own copy, not shared with old repo)
  published/                → daily_brief.json, rationale_log/ — what the Cowork
                             research/report layer and (if top 7) the pitch deck read
```

Adjust freely as the build progresses — this is a starting map, not a
contract. What's fixed is the separation: ported data-layer code lives in
`data_pipeline/`, everything competition-specific is new code in
`ranking/`, `portfolio/`, `backtest/`.

## Non-negotiables

- **Quant is the only source of truth for numbers.** Any LLM involvement
  (the Cowork-based research/catalyst layer — see `COMPETITION_PLAN.md`
  §7) may only ever apply a small, capped additive adjustment to a score
  this codebase already produced. It never re-ranks or overrides.
- **Nothing ships without a walk-forward backtest** (`backtest/engine.py`)
  proving it beats, on the same historical folds: a random portfolio,
  Kompas100 buy-and-hold, a naive momentum rule, and the previous version
  of this pipeline. No exceptions for "it seems obviously better."
- **Point-in-time universe only for backtests** — `kompas100_pit.csv`,
  never `kompas100_live.csv` applied retroactively.
- **Horizon is selected empirically.** 3D/5D/7D/10D/15D are candidates
  compared on realized portfolio return, stability, drawdown, and cost —
  never hardcode a default without a logged ablation result.
- **Model is frozen once the competition window opens.** No retraining
  against live competition-period data.
- **Circular features are forbidden.** Don't feed a rule-based composite
  score into the ranking model as an input — use the underlying raw
  indicators.
- **Do not port `ai_lab/`, `challenger_score.py`/`promote_challenger.py`,
  or `foreign_flow.py`** from the old repo into this project at all — not
  even as disabled/dormant code. They solve problems this project doesn't
  have, on a timeline that doesn't allow debugging someone else's
  half-finished machinery.

## Coding Standards (carried over from the old repo, still correct here)

```python
# Dates: always pd.Timestamp, tz-naive, normalized to midnight
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

# Logging: loguru, not print/stdlib logging
from loguru import logger
logger.info(f"{ticker}: 3 rows skipped for NaN")

# Config-driven: thresholds and hyperparams in YAML, never hardcoded
```

- Robustness over completeness: skip a ticker rather than crash the whole
  daily scan.
- Idempotent feature building: same input → same output, always.
- Simplicity over cleverness — this is a 2-person, 3-week build.
- No heavy new dependencies (PyTorch/TensorFlow, a new DB engine) without
  discussing it first — see `COMPETITION_PLAN.md` for the free-data /
  free-infra constraint.

## Typical Tasks

- Port a module from the old repo → copy the file into `data_pipeline/`,
  strip anything referencing `ai_lab`/`challenger`/broker-flow, adapt
  ticker universe references to `kompas100_live.csv`.
- Add a new character feature → `ranking/stock_character.py`, then run
  `backtest/ablation.py` Level 5 before assuming it helps.
- Test a new horizon → `ranking/ranking_model.py`, train, run through
  `backtest/engine.py`, log the result in `COMPETITION_PLAN.md` §4 —
  don't change "the" default horizon without that logged comparison.
- Adjust portfolio concentration → `portfolio/portfolio_optimizer.py`,
  re-run the relevant ablation level.
- Add/adjust price-level logic → `portfolio/level_calculator.py`. Output
  must land in `daily_brief.json` as plain numbers (entry/stop/target);
  never let the Cowork research layer compute or override these.