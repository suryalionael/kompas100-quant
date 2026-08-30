# Kompas100 Competition Build Plan

Working spec for a **new, standalone project** (`kompas100-quant` or
whatever you name it) built for the Kompas100 trading competition,
mid-September 2026. This is inspired by, and reuses select parts of,
`idx-stock-scanner-agent` — but it is its own repository. Do not build
this inside the old repo or treat the old repo as a dependency.

Check items off as they're done; log ablation results inline rather than
in a separate untracked spreadsheet, so a fresh Claude Code session can
see exactly what's been tried.

## 0. Hard constraints (do not relitigate these mid-build)

- Competition window: mid-September 2026, ~10 trading days.
- Universe: Kompas100, current list effective **3 Aug 2026 – 29 Jan 2027**.
- Budget: $0 incremental cost. Claude Pro/Cowork already available and does
  not count. No paid data providers.
- Team: 2 people, ~3 weeks from 28 Aug 2026.
- Quant is the source of truth for every number; LLM calls only ever apply
  a small, capped, additive adjustment — never a re-rank or override.
- **This is a separate project from `idx-stock-scanner-agent`.** That repo
  is a source to port specific modules from, not a base to build on top of.

## 1. What to take from idx-stock-scanner-agent — port, adapt, or leave

| Old repo file | Action | Notes |
|---|---|---|
| `stock_scanner/pipeline/fetch_yfinance.py` | **Port, trim** | Copy in, scope it to the ~100 Kompas100 tickers instead of ~830 IDX names. |
| `stock_scanner/pipeline/validator.py` | **Port as-is** | No changes needed — clean, correct data-quality logic. |
| `stock_scanner/pipeline/feature_builder.py` | **Port, extend** | Copy in as the technical-feature base; add IHSG/sector relative strength (missing in the original, still needed here). |
| `stock_scanner/pipeline/news_sentiment.py` | **Port as-is** | Real, free, already works. |
| `stock_scanner/pipeline/fundamental.py` | **Port as-is** | Real, free, use as a static quality filter. |
| `stock_scanner/pipeline/quality_filters.py` | **Port as-is** | Reusable exclusion logic (DER/PBV/float thresholds). |
| `stock_scanner/pipeline/signal_engine.py` | **Reference only** | Don't port the file. Its fixed-weight scoring formula and liquidity-gate *pattern* are worth reading before writing `stock_character.py`'s personalized-weight version, but the code itself gets rewritten, not copied. |
| `stock_scanner/pipeline/ml_ranker.py` / `train_ranker_from_history.py` | **Reference only** | Wrong target/horizon/universe for this project (see audit) — read for the training-loop scaffolding idea, write `ranking/ranking_model.py` fresh. |
| `stock_scanner/reference/issuers.py` | **Port, verify coverage** | Check it covers all current Kompas100 names, especially the Aug 2026 rebalance entrants (COIN, EMAS, MINA). |
| `ai_lab/*`, `challenger_score.py`, `promote_challenger.py` | **Leave behind entirely** | Non-functional, structurally can't feed back into production, out of scope. Do not port even as disabled code. |
| `foreign_flow.py`, `broker_analytics.py`, `broker_intelligence.py` | **Leave behind** | No viable free data source found (see §8 in the free-data research) — porting the code without real data behind it just recreates the fake-signal problem. |
| Streamlit dashboard (`dashboard/*`) | **Leave behind, rebuild minimal if needed** | The old dashboard is built around modules this project isn't using (broker intelligence, ai_lab views). If a dashboard is wanted here, build a small one against this project's own `data/published/daily_brief.json`, don't port the old one. |

## 2. Universe

- [ ] Build `configs/kompas100_live.csv` — the 100 current constituents
      (post 3-Aug-2026 rebalance). Reconstructed from Kompas/Kontan/Emiten
      News coverage of BEI's own evaluation — **cross-check once against
      an official IDX factsheet before trusting it for real money.**
      Rebalance: **out** BREN, BTPS, DSSA, FILM, HMSP, INTP, MTEL, SIDO,
      TCPI · **in** BFIN, BIPI, BNBR, COIN, EMAS, GGRM, LSIP, MINA, RMKE.
- [x] Build `configs/kompas100_pit.csv` — point-in-time membership
      calendar (6-month step function per historical rebalance cycle).
      Flag which periods have a verified snapshot vs. fall back to
      current list. **Done 2026-08-30** via `scripts/build_universe.py`:
      3 periods — 2026-08-03→2027-01-29 and 2026-02-02→2026-08-02 are
      **verified** (the Aug-2026 in/out list cross-checked against
      money.kompas.com's coverage of BEI's evaluation announcement);
      everything before 2026-02-02 falls back to the current list,
      flagged `source_verified=False`. Backtest folds before 2026-02
      are testing the engine's mechanics, not a faithful historical
      universe — upgrade before trusting them for real money.
- [ ] Confirm with the competition organizer: is scoring actually against
      this exact Kompas100 list, or an organizer-curated subset? (Phase 0
      — see §8; blocks nothing technical but blocks trusting the result.)

## 3. Backtest engine (build before trusting any model)

- [x] `backtest/engine.py` — walk-forward simulator over the point-in-time
      universe. Many historical start dates, ~10-day hold, realistic entry
      (T+1 open), IDX-typical costs (~0.15–0.25% buy / 0.25–0.35% sell,
      10–20bps slippage) as placeholders pending real platform figures.
      **Done 2026-08-30.**
- [x] Benchmarks wired in for every run: random portfolio (many draws),
      Kompas100 equal-weight buy-and-hold, naive momentum. Old scanner's
      rule-based signal (Level 3) intentionally **not** included —
      `signal_engine.py` is reference-only per §1, not meant to be ported
      and benchmarked as-is; would need to be rewritten first, which is
      §5's job (stock_character.py), not this one.
      **Pitfall caught while building this:** a "buy-and-hold" that
      rebalances every fold (same cadence as an active strategy) pays
      round-trip costs dozens of times over and is not buy-and-hold at
      all — `run_index_buy_and_hold()` rebalances only at real PIT
      boundaries (~twice a year) instead.
- [x] Report both overlapping and non-overlapping window results with
      confidence intervals — ~50–70 independent folds is the honest
      sample size across ~3 years. **Turned out to matter a lot** — see
      §4's result table; the overlapping view alone would have given a
      false positive.
- [x] IC (Spearman rank correlation) logged per fold as a diagnostic only.
      **Realized portfolio return is the pass/fail metric, not IC.**

## 4. Ranking model — horizon ablation (empirical, not assumed)

- [x] Train 5 independent cross-sectional relative-return regressors:
      **3D / 5D / 7D / 10D / 15D**. Target = forward return z-scored
      against the Kompas100 cross-section that date. **Done 2026-08-30**
      — `ranking/ranking_model.py`, a from-scratch numpy Ridge regression
      (no scikit-learn — closed-form, easy to audit, no new heavy dep).
      Retrained every 10 trading days, walk-forward: only training rows
      whose target was already resolvable by the decision date are used.
- [x] Run all 5 through `backtest/engine.py`, score on: realized portfolio
      performance, stability across regimes, IC (diagnostic), drawdown,
      transaction costs. **Regime stability not yet broken out** —
      `ranking/regime_classifier.py` (§6) doesn't exist yet; this run
      reports the aggregate across the full backtest window only.
- [x] Log the result table once run (`backtest/ablation.py`, results in
      `data/published/ablation_results.json`; overlapping folds = every
      trading day, ~500-700 per horizon; non-overlapping = every
      `horizon` days, independent, 32-169 per horizon — **the
      non-overlapping numbers are the ones that matter**, per §3):

  | Horizon | Model return/fold (non-overlap) | 95% CI | Momentum return/fold (non-overlap) | IC (non-overlap) | Model beats momentum (non-overlap)? |
  |---|---|---|---|---|---|
  | 3D | +0.15% | [-0.74%, +1.05%] | +0.30% | 0.004 | **no** |
  | 5D | +0.32% | [-1.39%, +2.04%] | +0.74% | -0.003 | **no** |
  | 7D | +1.31% | [-1.18%, +3.80%] | +1.74% | 0.007 | **no** |
  | 10D | +1.89% | [-1.58%, +5.35%] | +2.56% | 0.012 | **no** |
  | 15D | +4.87% | [-0.35%, +10.09%] | +5.57% | 0.024 | **no** |

  (The overlapping-fold view's IC — 0.005/0.011/0.016/0.022/0.028 — rises
  more smoothly with horizon and looks marginally better; that's the same
  autocorrelation-inflation effect as the return comparison above, not a
  different or more favorable result.)

  Kompas100 buy-and-hold over the same full window (real PIT rebalance
  dates, cost-adjusted): **+130.91%** compounded, -14.47% max drawdown,
  3 periods (2023-08-31→2026-08-28).

  **Verdict: no horizon beats naive momentum on the honest
  (non-overlapping) fold set — the model does not clear the ablation
  gate as built.** Every horizon's overlapping-fold view looked closer
  (10D/15D even appeared to win there), which is exactly the false
  positive §3's dual-reporting requirement exists to catch: overlapping
  folds sharing most of their trading days inflate apparent skill.
  Non-overlap IC is small, roughly rising with horizon, and negative at
  5D — not a clean signal yet, though it's not obviously nothing either
  (7D/10D/15D all positive). Not enough to beat a one-line momentum rule
  after costs. Per COMPETITION_PLAN's own
  ablation-gate rule, **this model is not wired into the dashboard's
  Rankings tab** — that stays honest about not having a shippable model
  yet rather than showing picks from something that lost the ablation.
- [ ] Select winning horizon(s) — **blocked**: none qualify yet. Next
      moves before re-running this ablation: (a) §5's character features
      might give the model something momentum doesn't have, (b) sector
      relative strength (needs `issuers.py` port, §1) is still missing
      from the feature set, (c) Ridge's lambda=1.0 and the 10-day retrain
      cadence were reasonable defaults, not tuned — a small hyperparameter
      sweep is cheap to try before concluding the approach itself is wrong.
- [x] Features: ported `feature_builder` set + IHSG relative strength
      (new — `rel_strength_5d`/`rel_strength_20d`, added 2026-08-30).
      Sector relative strength and Tier 1/2 character features (§5) still
      TODO. No circular rule-score features.

## 5. Stock character — personalized screener (ablation-gated)

- [ ] Build rolling (60–120d) character features per stock: breakout
      continuation rate, momentum autocorrelation, volume-price
      correlation, rolling beta to IHSG (regime-split), liquidity profile.
      **Recompute at every historical decision date from trailing data
      only** — never compute once over full history (look-ahead leak).
- [ ] Tier 1 — stock-specific normalization (z-score signals against each
      stock's own trailing history). Always on, cheap.
- [ ] Tier 2 — explicit personalized screener: per-stock signal weights as
      a function of character (breakout weight up for reliable breakers,
      momentum weight down/flipped for mean-reverters). Conceptually
      modeled on the old repo's `signal_engine` weighted-sum pattern, but
      with per-stock weights instead of fixed ones — new code.
- [ ] Tier 3 — same character features fed as plain columns to the
      ranking model, let it find interactions implicitly.
- [ ] Run Level 5 ablation: Tier 2 vs. Tier 3 vs. non-personalized
      baseline. Keep the winner, cut the rest — including cutting the
      whole system if neither beats baseline.
- [ ] Tier 4 (archetype clustering) — only build if 1–3 demonstrably leave
      value on the table.

## 6. Regime, portfolio, strategist

- [ ] `ranking/regime_classifier.py` — deterministic, 5 states (Strong
      Bull → Strong Bear); inputs: IHSG trend/momentum, breadth,
      volatility, volume regime. Ablation-test: does conditioning
      ranking/sizing on regime improve realized return?
- [ ] `portfolio/portfolio_optimizer.py` — expected-return-weighted, top
      6–8, per-name cap (~25–30%), per-sector cap. Test against
      equal-weight, top-3, top-10, half-Kelly in the ablation matrix (§9,
      Level 7).
- [ ] `portfolio/competition_strategist.py` — function of `gap`,
      `days_remaining`, `portfolio_volatility` → concentration/cash
      policy. **Only build the live-leaderboard-reactive version if
      Phase 0 confirms leaderboard visibility exists during the contest.**

## 7. Research/Catalyst layer — Cowork-based, not custom code

Implemented as a **scheduled Cowork task**, not a custom Python + LLM API
integration in this repo. This project's job is to publish clean data for
Cowork to read — not to build its own LLM client.

- [ ] Daily scan (`scripts/run_daily_scan.py`) commits
      `data/published/daily_brief.json` — ranked shortlist + scores +
      suggested position sizes.
- [ ] Confirm an email-sending connector (Gmail/Outlook) is available and
      connected before relying on this for delivery.
- [ ] Set up the scheduled task (`create_trigger`, weekday cron, timed
      after the scan completes and before IDX market open — ~1hr buffer).
      The task reads this project's repo/output, not the old one.
- [ ] Cowork's job is explain + research + deliver, **never** decide — the
      prompt hands it an already-ranked shortlist with numbers, never raw
      data to rank itself.
- [ ] Log Cowork's structured catalyst tags (direction/strength/
      confidence), not just the prose email, so the shadow test below has
      something to grade.
- [ ] Shadow test: run Quant-Only vs. Quant+Research in parallel for 5–10
      trading days during paper trading (Phase 6). Only enable for the
      live competition if it wins that comparison.

## 8. Phase 0 — open questions to resolve with the organizer (do this first)

- [ ] Exact competition start/end dates.
- [ ] Confirmed scoring universe (Kompas100 as reconstructed above, or
      something else).
- [ ] Transaction cost / slippage assumptions the platform actually uses.
- [ ] Whether live/near-live leaderboard visibility exists during the
      contest (gates §6's Competition Strategist).
- [ ] Position limits, rebalancing rules, whether shorting is allowed,
      capital amount.

## 9. Ablation matrix — the actual gate for what ships

Run every level through the identical backtest harness. A level that
doesn't beat the one below it gets cut, even if already built.

| # | Level | Result (fill in once run) |
|---|---|---|
| 0 | Random portfolio | 2026-08-30: ~flat to slightly negative per non-overlapping fold across horizons (e.g. 10D: +0.45%/fold) — see `data/published/ablation_results.json` |
| 1 | Kompas100 benchmark (buy-and-hold) | 2026-08-30: **+130.91%** compounded, -14.47% max drawdown, full window 2023-08-31→2026-08-28, real PIT rebalance dates, cost-adjusted |
| 2 | Simple momentum | 2026-08-30: beats Level 0 and Level 1 (per-fold) at every horizon on the non-overlapping set — current best baseline, see §4's table |
| 3 | Old scanner's rule-based signal, for reference | Not run — `signal_engine.py` is reference-only (§1), deliberately not ported/benchmarked as-is |
| 4 | This project's quant model (best horizon from §4) | 2026-08-30: **does not beat Level 2** at any of the 5 horizons on the non-overlapping set — **cut, not shipped** (§4 has the full table + IC + next moves) |
| 5 | + Stock character (winning variant from §5) | Not built |
| 6 | + Research/Catalyst (shadow-tested, §7) | Not built |
| 7 | + Full portfolio construction + strategist (§6) | Not built |

## 10. Roadmap (3 weeks, 28 Aug → mid-Sept)

- [ ] Days 1–2 — Phase 0 (§8) + repo scaffolding + port the §1 files
- [ ] Days 2–5 — Universe + data foundation (§2)
- [ ] Days 4–9 — Backtest engine (§3)
- [ ] Days 7–12 — Ranking model + horizon ablation + character (§4, §5)
- [ ] Days 11–14 — Regime, portfolio, strategist (§6)
- [ ] Days 13–17 — Research layer + shadow test (§7)
- [ ] Days 15–19 — Paper trading, full system dry run
- [ ] Mid-Sept — Competition. Model frozen, no mid-contest changes.

## 11. Explicitly out of scope for this project

`ai_lab/*`, `challenger_score.py`/`promote_challenger.py`, foreign/broker
flow (no viable free source found), bull/bear adversarial LLM agents, any
new database beyond SQLite/Parquet, any paid data provider, any
dependency on the old repo at runtime, any hard-coded horizon assumption
that skipped the §4 ablation.
