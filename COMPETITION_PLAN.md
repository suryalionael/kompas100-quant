# Kompas100 Competition Build Plan

Working spec for a **new, standalone project** (`kompas100-quant` or
whatever you name it) built for **ISTC 2026** (the Kompas100 trading
competition this project targets). This is inspired by, and reuses select
parts of, `idx-stock-scanner-agent` — but it is its own repository. Do not
build this inside the old repo or treat the old repo as a dependency.

Check items off as they're done; log ablation results inline rather than
in a separate untracked spreadsheet, so a fresh Claude Code session can
see exactly what's been tried.

## 0. Hard constraints (do not relitigate these mid-build)

**Official ISTC 2026 rules** (corrected 2026-08-31 — this is not a "most
money wins" contest, and it isn't "mid-September":

- **Live trading window: 21 Sept – 8 Oct 2026, exactly 14 working days.**
  One virtual account, **Rp100,000,000** starting capital, **Kompas100
  stocks only** — trading anything else is disqualification, not a
  penalty. All positions **manually closed by the team on 8 Oct** — this
  is a platform rule, not a strategy choice (see §12).
- **Winner = 60% Preliminary Stage + 40% Final Stage:**
  - Preliminary (60%): pure asset balance (cash + realized + unrealized
    P&L) at the end of the 14 days. Ranks everyone; only the **top 7**
    advance.
  - Final Stage (40%, top-7 only): a **live pitch**, not more trading —
    graded on Strategy & Analysis 40% (Fundamental 10% / Technical 10% /
    **Money Flow Analysis 20%**), Risk Management 20% (diversification
    10% / risk-reward 10%), Macro linkage 10%, Presentation & QnA 30%.
  - Implication that changes priorities: **a great ranking model that
    can't be explained is a losing model.** Money Flow Analysis alone
    outweighs technical or fundamental analysis individually. The system
    has to leave a paper trail good enough to win a judged pitch two
    months after the trading window closes, not just pick good stocks —
    see §12/§13, built alongside the trading engine from day one, not
    bolted on after 8 Oct.
- **Model frozen by ~18–19 Sept 2026**, before the 21 Sept open. No
  exceptions, no "just one more tweak" after that date (see §4's
  ablation table for what's actually validated so far).
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
- [ ] Build `configs/kompas100_pit.csv` — point-in-time membership
      calendar (6-month step function per historical rebalance cycle).
      Flag which periods have a verified snapshot vs. fall back to
      current list.
- [ ] Confirm with the competition organizer: is scoring actually against
      this exact Kompas100 list, or an organizer-curated subset? (Phase 0
      — see §8; blocks nothing technical but blocks trusting the result.)

## 3. Backtest engine (build before trusting any model)

- [x] `backtest/engine.py` — walk-forward simulator over the point-in-time
      universe. Many historical start dates, ~10-day hold, realistic entry
      (T+1 open), IDX-typical costs (~0.15–0.25% buy / 0.25–0.35% sell,
      10–20bps slippage) as placeholders pending real platform figures.
- [x] Benchmarks wired in for every run: random portfolio (many draws),
      Kompas100 equal-weight buy-and-hold, naive momentum. Old scanner's
      rule-based signal (Level 3) intentionally not included —
      `signal_engine.py` is reference-only per §1.
- [x] Report both overlapping and non-overlapping window results with
      confidence intervals — ~50–70 independent folds is the honest
      sample size across ~3 years. **This distinction mattered in
      practice** — see §4's 2026-08-31 entry: the overlapping view
      showed an apparent edge that the non-overlapping view didn't
      confirm, exactly the false-positive this dual reporting exists to
      catch.
- [x] IC (Spearman rank correlation) logged per fold as a diagnostic only.
      **Realized portfolio return is the pass/fail metric, not IC.**

## 4. Ranking model — horizon ablation (empirical, not assumed)

- [x] Train 5 independent cross-sectional relative-return regressors:
      **3D / 5D / 7D / 10D / 15D**. Target = forward return z-scored
      against the Kompas100 cross-section that date. `ranking/
      ranking_model.py` — a from-scratch numpy Ridge regression, not
      scikit-learn (closed-form, no new heavy dep, easy to audit).
- [x] Run all 5 through `backtest/engine.py`, score on: realized portfolio
      performance, IC (diagnostic), drawdown, transaction costs.
      Stability-across-regimes is not yet broken out —
      `ranking/regime_classifier.py` (§6/§12) isn't wired into the
      ablation, only used for narrative logging so far.

### 2026-08-31 — ground-truth audit + real bugs found and fixed, before any model changes

Three real, confirmed bugs, all fixed and verified independently before
touching the model at all (see git history for each):

1. **`data_pipeline/fetch_yfinance.py`'s `default_end_date()` returned
   today's date, but yfinance's `end` parameter is EXCLUSIVE** —
   confirmed directly (`yf.download(..., end="2026-08-31")` returns
   nothing for 2026-08-31 itself; `end="2026-09-01"` does). The daily
   scan could never pick up the same day's close no matter when it ran
   after market close — permanently one trading day behind. Fixed:
   `default_end_date()` now returns tomorrow.
2. **`ranking/ranking_model.py`'s `build_training_dataset()` z-scored the
   forward-return target against every ticker with a feature row on a
   date — not `backtest.engine.get_pit_universe()`'s actual point-in-time
   Kompas100 membership.** A direct violation of CLAUDE.md's non-
   negotiable ("kompas100_pit.csv, never kompas100_live.csv applied
   retroactively") and a real train/inference mismatch: the model scores
   against the correct PIT universe at inference time
   (`make_ranking_score_fn`) but was trained against a different, wrong
   cross-section. Fixed: `build_training_dataset()` now requires a
   `pit_df` argument and filters to `get_pit_universe(pit_df, date)`
   before computing the target.
3. **`backtest/ablation.py` had a hardcoded, dated features-file path**
   (`data/features/2026-08-30.parquet`) that silently went stale the
   moment a newer file existed — every ablation run was training on
   yesterday's features regardless of what `run_daily_scan.py` had
   actually produced since. Fixed: reads the latest file in
   `data/features/` (same pattern as `scripts/dashboard.py`'s
   `load_features_latest()`).

Also verified and found correct (no bug): forward-return z-scoring's
entry/exit prices (`backtest.engine.PricePanel.forward_returns` — T+1
open entry, close exit) exactly match what `run_backtest()`'s own trade
simulation uses, so the training target and the backtest's realized
returns are computed the same way. Walk-forward training correctly only
uses rows with `resolved_date <= decision_date` (`make_ranking_score_fn`).

**Isolated effect of the PIT-universe fix alone** (before any feature
changes, non-overlapping/honest folds): every horizon's IC improved
(3D: 0.005→0.010, 5D: -0.003→0.005, 7D: 0.007→0.017, 10D: 0.012→0.023,
15D: 0.024→0.035) and 3D/5D briefly cleared the "beats momentum" bar on
one data snapshot — but see the robustness caveat below before reading
that as a real edge.

### 2026-08-31 — feature widening, tested additively, both reverted

- **Refined momentum** (`roc3`, `sharpe_mom_20d` = roc20/hist_vol_20d,
  `mom_vol_confirmed_20d` = roc20 × vol_ratio_20d) — added to
  `data_pipeline/feature_builder.py` (kept, still computed) and briefly
  to `RANKING_FEATURES`. Result: **every horizon got worse** — lower
  model return AND lower IC across the board; 5D lost its "beats
  momentum" result entirely. Likely cause: these are multiplicative
  recombinations of features already in the model (roc20, hist_vol_20d,
  vol_ratio_20d), adding collinearity without new information, diluting
  the Ridge fit. **Reverted from `RANKING_FEATURES`.**
- **Sector-relative strength** (`sector_rel_strength_20d` — 20D return
  vs. same-sector peers, not just vs. IHSG; added to `feature_builder.py`
  as `add_sector_relative_strength()`, kept, still computed) — a priori
  the more promising addition (genuinely different comparison group from
  the market-relative feature already in the model). Result: roughly a
  wash to slightly worse — 5D again lost its "beats momentum" result,
  IC flat-to-down on most horizons. **Reverted from `RANKING_FEATURES`.**

### 2026-08-31 — final honest result (current `RANKING_FEATURES`, freshest data)

  | Horizon | Model return/fold (non-overlap) | 95% CI | Momentum return/fold | IC | Beats momentum? |
  |---|---|---|---|---|---|
  | 3D | +0.37% | [-0.48%, +1.21%] | +0.30% | 0.003 | barely, not robust (see below) |
  | 5D | +0.41% | [-1.23%, +2.04%] | +0.74% | -0.003 | no |
  | 7D | +1.22% | [-1.08%, +3.52%] | +1.74% | 0.005 | no |
  | 10D | +1.73% | [-1.64%, +5.10%] | +2.56% | 0.012 | no |
  | 15D | +3.93% | [-0.99%, +8.85%] | +5.57% | 0.024 | no |

  Overlapping view (reference only): **nothing beats momentum on any
  horizon** — 3D's overlapping model return (+0.14%) is below momentum's
  (+0.25%).

  **Robustness check, not a cherry-pick:** re-running 3D twice — once
  right after the PIT fix (still against a slightly older, cached
  features file) and once against a freshly re-fetched file (yfinance's
  `auto_adjust=True` retroactively revises historical adjusted-close
  values on every refetch, e.g. for dividends) — gave visibly different
  results (+0.34%/IC 0.011 vs. the +0.37%/IC 0.003 logged above,
  overlapping went from "wins" to "loses"). **A signal that flips
  between beating and losing to momentum from routine dividend-
  adjustment noise on 3-year-old prices is not a robust, demonstrated
  edge — it's noise-level.** Per this project's own gate rule, honest
  read: 3D does not clear it either, despite the point estimate.

  **Verdict: no horizon has a demonstrated, robust edge over naive
  momentum.** `portfolio/daily_brief.py` cannot emit `"validated"`
  regardless (no live inference path exists — deliberate, see §7), so
  this doesn't change anything about what ships; the honest
  `naive_momentum_interim` state (already dashboard-visible, already
  gated by `quality_filters.py`'s eligible set, already has a real level
  and sizing on every name — see §13/`daily_brief.py`'s 2026-08-31 bug
  fixes) remains correct.

- [ ] Select winning horizon(s) — **still blocked**: none robustly clear
      the bar after real effort (leakage/misalignment fix + two
      feature-widening attempts, this session). Candidate next moves, not
      yet tried: (a) §5's character features, (b) a rank-based objective
      (top-quartile classification) instead of exact z-score regression —
      allowed to change model type per this session's brief, but not
      attempted due to time; a genuinely separate build, not a quick
      addition, (c) regularization/robustness tuning given the fragility
      finding above, (d) more training data (currently ~3yr lookback).
- [x] Features: ported `feature_builder` set + IHSG relative strength.
      Sector-relative strength and refined-momentum interactions
      **tried and reverted** (see above — both computed and available in
      `feature_builder.py` output, just not in `RANKING_FEATURES`). No
      circular rule-score features. Tier 1/2 character features (§5)
      still not built.

### Robustness protocol — frozen-snapshot verification (added 2026-08-31)

The 3D flip above proved a single live/latest-file ablation run is not
trustworthy evidence on its own. **New rule: no horizon may be reported
as "beats momentum" until that verdict agrees across >= 2 data snapshots
frozen on different calendar days.** One live run is a data point, not a
result.

Tooling: `scripts/freeze_snapshot.py --tag <name>` copies the current
`data/raw/*.parquet` + latest `data/features/*.parquet` + the PIT CSV into
an immutable `data/snapshots/<tag>/` with a manifest (per-file SHA-256).
`backtest/ablation.py --snapshot <tag>` runs against that frozen copy
instead of live data, writing to a separate
`data/published/ablation_results__<tag>.json` so runs don't clobber each
other. `scripts/compare_snapshots.py <tag1> <tag2> ...` diffs verdicts
across snapshots and flags any horizon that flips as "not robust."

**Snapshot 1 of >= 2 — `20260831_1719Z`** (frozen 2026-08-31, features
file `2026-08-31.parquet`, 110 tickers):

| Horizon | Model (non-ov) | Momentum (non-ov) | Both views agree? | Verdict |
|---|---|---|---|---|
| 3D | +0.63% | +0.30% | yes (both YES) | marginal win — **not yet confirmed** |
| 5D | +0.82% | +0.81% | yes (both YES) | razor-thin (0.01pp) — **not yet confirmed** |
| 7D | +1.44% | +1.86% | yes (both no) | no |
| 10D | +2.43% | +2.68% | **no** (overlap YES, non-overlap no) | not robust |
| 15D | +4.32% | +5.59% | **no** (overlap YES, non-overlap no) | not robust |

Buy-and-hold this snapshot: +132.36% compounded, -14.47% max drawdown
(drifted slightly from the +130.91% logged 2026-08-30 above — normal
data-revision movement, not a bug, and itself more evidence for why this
protocol exists).

**These numbers are already different from the 2026-08-30 table above**
(e.g. 3D momentum was +0.30% then too, but the model side moved from
+0.37% to +0.63%; 5D model moved from +0.41% to +0.82%) despite no code
change — same conclusion as the earlier 3D flip, now visible on every
horizon, not just one. **Do not report 3D or 5D as a real edge yet** —
freeze a second snapshot on a different day (after the next scheduled or
manual data refresh actually lands new rows) and run
`scripts/compare_snapshots.py 20260831_1719Z <new_tag>` before believing
either one. If they still both say YES after that, that's one real data
point toward "robust"; the protocol calls for >= 2 agreeing snapshots
minimum, ideally 3.

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
- [x] `portfolio/level_calculator.py` — deterministic buy/sell/stop price
      levels for every shortlisted name. Modeled on the old repo's
      `alerts/level_calculator.py`: entry near breakout/support level,
      stop-loss from ATR (e.g. 1.5–2x ATR below entry), target from a
      fixed reward:risk multiple (e.g. 2:1) off the stop distance. Output
      feeds straight into `daily_brief.json` — **this is what the Cowork
      daily report explains, never what it invents.** No LLM call may
      produce a price number; it may only narrate one already computed
      here. Needed before the daily-report Cowork task (§7) can show real
      levels instead of "not yet available." **Done 2026-08-30**: uses
      1.5x ATR (simple rolling-mean True Range, not Wilder's — chosen
      specifically so unit-test fixtures are hand-verifiable) for the
      stop and a 2:1 reward:risk target; a setup only exists (breakout or
      pullback) when close is within 3% (placeholder, like the backtest
      engine's cost assumptions) of the rolling 20-day high/low — no
      setup, no level, rather than forcing one on every ticker.

## 7. Research/Catalyst layer — Cowork-based, not custom code

Implemented as a **scheduled Cowork task**, not a custom Python + LLM API
integration in this repo. This project's job is to publish clean data for
Cowork to read — not to build its own LLM client. Cowork's own scheduled
task (below) reads `daily_brief.json`, so it's only as fresh as the daily
data refresh that produces it — see the fallback procedure right below.

### Manual fallback — daily data refresh

**2026-08-31 finding:** the `daily_data_refresh.yml` cron (10:30 UTC /
17:30 WIB) had not fired on its own schedule as of its first scheduled
slot after being added to `main`. Checked: workflow `state` is `active`
(not disabled), repo Actions permissions allow all actions, the job
declares its own `permissions: contents: write` (so the repo-level
default of "read" doesn't block the commit step), and githubstatus.com
showed no Actions incidents in that window. No misconfiguration found —
most likely a one-off missed/delayed first scheduled run, a known GitHub
Actions behavior.

**Resolved same day, 2026-08-31 17:40 UTC:** the very next slot fired on
its own (`gh run list` shows `event: "schedule"`, `conclusion: "success"`)
and committed `922ff76`, including the first-ever `scheduled_run_state.json`.
Confirms the earlier miss was the one-off first-run delay suspected above,
not a real misconfiguration. The Data Health tab's "Hours Since Last
Scheduled Success" panel will still flag red (>30h) if a slot is ever
missed again — kept as the standing safety net, not removed now that
it's working.

**Until the schedule is confirmed reliable (or any time it silently
misses a day), the fallback is a one-tap manual run — no local setup, no
laptop required:**

1. GitHub mobile app (or github.com) → this repo → **Actions** tab.
2. Select **"Daily data refresh"** in the left sidebar.
3. Tap **"Run workflow"** → confirm. No inputs to fill in
   (`workflow_dispatch: {}` takes none) — it's one tap plus one confirm.
4. Takes 2-4 minutes; commits new data to `main` automatically if the
   fetch produced changes. Check the Data Health tab afterward to confirm
   `Latest Run Trigger` shows the new run and data looks current.

This is deliberately **not** backed by a second independent scheduler
(e.g. cron on another free host) — for a 14-day competition window, a
documented one-tap human fallback plus the red-flag staleness panel is
enough insurance. Revisit only if the schedule proves fundamentally
unreliable (multiple missed days, not just the one slot above).

- [x] Daily scan (`scripts/run_daily_scan.py`) commits
      `data/published/daily_brief.json` — ranked shortlist + scores +
      suggested position sizes + buy/sell/stop levels from
      `portfolio/level_calculator.py` (§6). If the ranking model's
      ablation gate isn't cleared yet, publish an honestly-labeled
      fallback (e.g. `"strategy": "naive_momentum_interim"`) instead of
      an empty or fabricated shortlist — the Cowork report (below) must
      be able to say what it's actually showing. **Done 2026-08-30.**
      `portfolio/level_calculator.py` computes breakout/pullback entry +
      ATR-based stop + fixed-2:1-R:R target, pure OHLCV arithmetic, no
      ML/LLM — 6 hand-computed unit tests in `tests/test_level_calculator.py`.
      `portfolio/daily_brief.py` sets `strategy_status` to one of
      `validated` / `naive_momentum_interim` / `no_picks`; since no live
      ranking-model inference path exists yet (only the backtest-oriented
      training/scoring functions in `ranking/ranking_model.py` do), it
      structurally can never emit `validated` today, by design — current
      real output is `naive_momentum_interim` (10-ticker momentum
      shortlist), matching the dashboard Rankings tab's own verdict.
- [x] Email delivery: **Gmail connector** — connect via claude.ai →
      Settings → Connectors before the scheduled task can send mail.
      Not yet connected as of 31 Aug 2026.
- [ ] Set up the scheduled task (`create_trigger`, daily ~07:00–07:30 WIB,
      after the previous evening's scan and before IDX market open).
      Initial recipient: user's own email only, for the first few days;
      add the partner's email once output quality is confirmed. The task
      reads this project's `daily_brief.json`, never the old repo and
      never the live dashboard UI directly.
- [ ] Cowork's job is explain + research + deliver, **never** decide — the
      prompt hands it an already-ranked shortlist with numbers and
      pipeline-calculated price levels, never raw data to rank or price
      itself. The AI may narrate *why* a level is where it is (e.g. "stop
      sits below the recent swing low"), never invent or adjust the
      number.
- [ ] Report format: PDF, per stock in the shortlist — chart, key
      fundamentals, quant score explanation, catalyst/news research, and
      the pipeline-calculated entry/stop/target — plus a short market
      overview (IHSG, regime, breadth) at the top.
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
| 0 | Random portfolio | 2026-08-31: ~flat to slightly negative per non-overlapping fold across horizons (e.g. 10D: +0.17%/fold) |
| 1 | Kompas100 benchmark (buy-and-hold) | 2026-08-31: **+130.91%** compounded, -14.47% max drawdown, full window 2023-08-31→2026-08-28, real PIT rebalance dates, cost-adjusted |
| 2 | Simple momentum | 2026-08-31: beats Level 0 and Level 1 (per-fold) at every horizon on the non-overlapping set — current best baseline, see §4 |
| 3 | Old scanner's rule-based signal, for reference | Not run — `signal_engine.py` is reference-only (§1), deliberately not ported/benchmarked as-is |
| 4 | This project's quant model (best horizon from §4) | 2026-08-31: **does not robustly beat Level 2** at any horizon after a real leakage/misalignment fix + two feature-widening attempts — **cut, not shipped**. See §4 for the full investigation, including a robustness check showing the one apparent "win" (3D) isn't stable across routine data revisions. |
| 5 | + Stock character (winning variant from §5) | Not built |
| 6 | + Research/Catalyst (shadow-tested, §7) | Not built |
| 7 | + Full portfolio construction + strategist (§6) | Not built |

## 10. Roadmap (real ISTC 2026 dates — corrected 2026-08-31)

- [x] Days 1–2 (28–30 Aug) — repo scaffolding + port the §1 files +
      Streamlit dashboard
- [x] Days 3 (31 Aug) — backtest engine (§3), ranking model + horizon
      ablation (§4, result: no horizon beats momentum yet), price levels
      + honest daily_brief fallback (§7), rationale log + hard-rules
      enforcement (§12, §13)
- [ ] By ~13 Sept — stock character (§5), sector-aware relative strength,
      regime-conditioned ablation (does §12's regime_classifier.py
      actually improve realized return when wired into ranking/sizing? —
      not yet tested, see §6)
- [ ] By ~18 Sept — portfolio construction + strategist (§6), research
      layer + shadow test (§7), paper trading dry run
- [ ] **~18–19 Sept — model frozen.** No exceptions.
- [ ] **21 Sept – 8 Oct — ISTC 2026 live trading window (14 working
      days).** Daily rationale log entries accumulate every trading day
      (§12) — this is not optional bookkeeping, it's the only real-time
      record the Final Stage pitch will have.
- [ ] **8 Oct — all positions manually closed, end of day.** Hard
      platform rule (§13), not a strategy choice.
- [ ] ~14 Oct — top-7 (Preliminary Stage) announced.
- [ ] If top 7: build the pitch deck from
      `scripts/build_pitch_deck_source.py`'s output — raw material, not
      slides; the deck and live presentation are a human/team task (§12).
- [ ] Final Stage pitch — Strategy & Analysis 40% (Fundamental 10% /
      Technical 10% / Money Flow 20%), Risk Management 20%, Macro linkage
      10%, Presentation & QnA 30%.

## 11. Explicitly out of scope for this project

`ai_lab/*`, `challenger_score.py`/`promote_challenger.py`, a live-automated
money-flow *trading signal* (no free reliable source found — §12's proxies
are narrative support only, never a ranking model input), bull/bear
adversarial LLM agents, any new database beyond SQLite/Parquet, any paid
data provider, any dependency on the old repo at runtime, any hard-coded
horizon assumption that skipped the §4 ablation, automating the Final
Stage pitch deck itself (§12's `build_pitch_deck_source.py` produces raw
material, not slides).

## 12. Rationale log — a contemporaneous record for the Final Stage pitch

**Done 2026-08-31.** The single biggest gap versus the actual grading
rubric (§0) was that nothing captured *why* a pick was made, day by day —
built now, alongside the trading engine, rather than reconstructed from
memory in October.

- [x] `data_pipeline/money_flow_proxies.py` — cheap OHLCV-derived proxies
      (volume-spike-without-price-follow-through, price-up-on-declining-
      volume, a volume z-score), explicitly flagged as proxies, never real
      flow data (no free reliable source exists — that conclusion from the
      original free-data audit hasn't changed). Talking points for the
      pitch's Money Flow Analysis (20% of the Final Stage score), never
      fed into `ranking/ranking_model.py` — deliberately kept out of
      `feature_builder.py` so they can't end up in `RANKING_FEATURES` by
      accident.
- [x] `ranking/regime_classifier.py` — deterministic, 5 states (Strong
      Bull → Strong Bear) from IHSG trend/momentum/breadth/volume, built
      now so there's a real macro state to log daily. **Not yet
      ablation-tested** for whether conditioning ranking/sizing on it
      improves realized return (§6's own requirement) — not wired into
      the ranking model or backtest engine; exists purely to produce a
      loggable state today.
- [x] `data_pipeline/fundamental.py` extended to capture `sector`/
      `industry` from the same yfinance `.info` call it already makes —
      chosen over porting the old repo's `issuers.csv` (§1), which only
      covered 52/100 current tickers and was missing several Aug-2026
      rebalance entrants (COIN, EMAS, MINA among them). 100/100 coverage
      now, no manual research, no static file to keep in sync.
- [x] `portfolio/rationale_log.py` writes
      `data/published/rationale_log/{date}.json`, one entry per ticker
      that's open/held/closed that day — a genuine day-over-day diff
      against `data/published/positions_state.json`, not invented.
      `technical_notes`/`fundamental_notes`/`macro_notes`/`risk_notes`
      are auto-populated plain-language summaries of real numbers already
      computed elsewhere (feature_builder, fundamental.py,
      regime_classifier.py, level_calculator.py, daily_brief.py's
      position sizing). `money_flow_proxies` is the auto-computed talking
      points above. `money_flow_notes` is the one field left `null` —
      manual research the team does themselves (broker summaries,
      bandarmology write-ups) — paired with a `{date}_manual_flow_notes.md`
      template (`save_manual_flow_template()`, never overwrites a file the
      team has already filled in). Wired additively into
      `scripts/run_daily_scan.py` right after `daily_brief.json`.
- [x] `scripts/build_pitch_deck_source.py` — concatenates a date range of
      rationale log entries (plus their paired manual-notes files) into
      one readable Markdown timeline. Raw material for October, not the
      deck itself — building slides and presenting is a human/team task.
- [x] **Overall Health score (added 2026-08-31).** `data_pipeline/
      quality_filters.py`'s `compute_overall_health()` re-expresses the
      existing DER/PBV/float/regulatory pass-fail criteria as a weighted
      0-100 composite (leverage/valuation/float/regulatory 20% each,
      profitability 10%, distribution-risk/data-completeness 5% each) plus
      a plain-English per-factor breakdown — pitch-deck material for the
      Final Stage's Fundamental Analysis line (10% of that grade), not a
      new fundamental input and not gated into ranking/sizing (confirmed:
      no reference anywhere in `ranking/`, `portfolio/portfolio_optimizer.py`,
      or `backtest/`). Surfaced in the dashboard (Stock Detail's
      per-ticker breakdown expander, Data Health's sortable "Overall
      Health by Ticker" table) and folded into `rationale_log.py`'s
      `fundamental_notes` (`"overall health NN/100"`). One real bug caught
      before shipping: an initial smooth linear-decay formula for the
      Valuation sub-score scored BBCA's PBV 2.94x at 37.7 while still
      labeling it "reasonable" — self-contradictory pitch material.
      Replaced with a 3-tier scheme (reasonable/rich/excessive) matching
      leverage's existing pattern, using the same 50%-of-`pbv_max` soft
      threshold convention already used for DER's watch band.

## 13. Hard rules — enforced structurally, not just documented

**Done 2026-08-31.**

- [x] Kompas100-only universe: `portfolio/daily_brief.py`'s
      `assert_kompas100_only()` raises (`Kompas100ViolationError`) before
      any shortlist with an out-of-universe ticker can be published —
      verified with a direct test. The dashboard's shortlist view
      (§ dashboard) re-checks and refuses to display, too, in case
      `daily_brief.json` is ever hand-edited or stale. Disqualification
      for one bad name is worse than a mediocre portfolio.
- [x] Trading-days-remaining countdown: the dashboard shows a banner
      derived from the real ISTC 2026 dates (freeze target 18–19 Sept,
      open 21 Sept, close 8 Oct) at every stage — pre-freeze, pre-open,
      during the window (escalating tone inside the last 3 trading days),
      and post-close. "All positions must be manually closed by end of
      day 8 Oct 2026" is a platform rule, not a strategy choice — the
      system should make it impossible to forget, not just documented
      here.
- [x] Real position sizing: `portfolio/daily_brief.py`'s
      `_naive_position_size()` sizes against the actual **Rp100,000,000**
      starting capital, equal-weight across the shortlist capped at 25%
      per name — a placeholder pending the real, ablated
      `portfolio/portfolio_optimizer.py` (§6), not itself validated, but
      real Rupiah numbers instead of an abstract weight.
- [x] Sector concentration made visible: the dashboard's shortlist view
      aggregates current shortlist exposure per sector against a 30%
      placeholder cap — what "diversification" (10% of the Final Stage
      Risk Management score) gets pointed at in the pitch. Real sector
      labels now that `fundamental.py` captures them (see §12).
- [x] R:R surfaced plainly: every shortlist row and rationale log entry
      shows the reward:risk ratio from `level_calculator.py` directly,
      not buried in a nested field — the "risk-reward" talking point
      (10% of the Final Stage Risk Management score).