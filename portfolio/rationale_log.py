"""Builds data/published/rationale_log/{date}.json — a same-day,
auto-populated record of the technical/fundamental/macro/risk reasoning
behind each open/hold/close decision. Every field but one is generated
from data the pipeline already computed elsewhere (feature_builder,
fundamental.py, regime_classifier.py, level_calculator.py, daily_brief.py's
sizing) — this module summarizes and formats, it does not compute anything
new.

money_flow_notes is the one field this module never fills in: it's a
manual research slot for the team (see money_flow_proxies.py for the
auto-computed proxy talking points that go in money_flow_proxies instead,
and build_manual_flow_template() below for the human-fill-in artifact).

Why build this now: the ISTC 2026 Final Stage (40% of the overall score,
top-7 only) is a live pitch graded on exactly these categories — Strategy
& Analysis (fundamental 10% / technical 10% / money flow 20%), Risk
Management 20%, Macro linkage 10%. A contemporaneous daily record beats
reconstructing two months of reasoning from memory in October.

open/hold/close is a genuine day-over-day diff against
data/published/positions_state.json, not invented — a ticker is "open" the
first day it appears in the shortlist, "hold" on every subsequent day it's
still there, "close" the first day it drops out. This tracks what the
system's picks *would* be holding; it is not a record of real broker
executions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

from data_pipeline import money_flow_proxies
from ranking import regime_classifier

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_DIR = ROOT / "data" / "published"
RAW_DIR = ROOT / "data" / "raw"
RATIONALE_LOG_DIR = PUBLISHED_DIR / "rationale_log"
POSITIONS_STATE_PATH = PUBLISHED_DIR / "positions_state.json"
UNIVERSE_SNAPSHOT_PATH = PUBLISHED_DIR / "universe_snapshot_latest.parquet"

ACTION_OPEN = "open"
ACTION_HOLD = "hold"
ACTION_CLOSE = "close"


# ---------------------------------------------------------------------------
# Position-state diff (open / hold / close)
# ---------------------------------------------------------------------------

def _load_previous_open_tickers() -> list[str]:
    if not POSITIONS_STATE_PATH.exists():
        return []
    return json.loads(POSITIONS_STATE_PATH.read_text()).get("open_tickers", [])


def _save_positions_state(date_str: str, open_tickers: list[str]) -> None:
    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    POSITIONS_STATE_PATH.write_text(
        json.dumps({"date": date_str, "open_tickers": open_tickers}, indent=2)
    )


def _resolve_action(ticker: str, today_tickers: set[str], previous_open: set[str]) -> str:
    if ticker in today_tickers and ticker in previous_open:
        return ACTION_HOLD
    if ticker in today_tickers:
        return ACTION_OPEN
    return ACTION_CLOSE


# ---------------------------------------------------------------------------
# Note generators — plain-language summaries of real, already-computed numbers
# ---------------------------------------------------------------------------

def _technical_notes(features_row: pd.Series | None) -> str:
    if features_row is None:
        return "no technical feature data available"
    parts = []
    rsi = features_row.get("rsi14")
    if pd.notna(rsi):
        parts.append(f"RSI14 {rsi:.1f}")
    roc20 = features_row.get("roc20")
    if pd.notna(roc20):
        parts.append(f"20D momentum {roc20:+.1f}%")
    price_vs_ma200 = features_row.get("price_vs_ma200")
    if pd.notna(price_vs_ma200):
        parts.append(f"{price_vs_ma200:+.1f}% vs MA200")
    pct_from_high = features_row.get("pct_from_52w_high")
    if pd.notna(pct_from_high):
        parts.append(f"{pct_from_high:+.1f}% from 52w high")
    if bool(features_row.get("ma_full_alignment")):
        parts.append("full MA bullish alignment (MA20>MA50>MA200)")
    rel_20d = features_row.get("rel_strength_20d")
    if pd.notna(rel_20d):
        parts.append(f"20D relative strength vs IHSG {rel_20d:+.1f}pp")
    return "; ".join(parts) if parts else "insufficient technical data"


def _fundamental_notes(fund_row: pd.Series | None) -> str:
    if fund_row is None or str(fund_row.get("fundamental_status", "missing")) == "missing":
        return "fundamental data unavailable"
    parts = []
    if pd.notna(fund_row.get("pe_ratio")):
        parts.append(f"P/E {fund_row['pe_ratio']:.1f}x")
    if pd.notna(fund_row.get("pbv")):
        parts.append(f"P/BV {fund_row['pbv']:.2f}x")
    if pd.notna(fund_row.get("roe_pct")):
        parts.append(f"ROE {fund_row['roe_pct']:.1f}%")
    if pd.notna(fund_row.get("der")):
        parts.append(f"DER {fund_row['der']:.2f}x")
    status = fund_row.get("final_status")
    if status and status != "eligible":
        parts.append(f"quality flag: {status}")
    if pd.notna(fund_row.get("overall_health_score")):
        parts.append(f"overall health {fund_row['overall_health_score']:.0f}/100")
    return "; ".join(parts) if parts else "fundamental data unavailable"


def _macro_notes(regime: dict) -> str:
    if "score" not in regime:
        return f"regime unavailable — {regime.get('reason', 'no data')}"
    return (
        f"IHSG regime: {regime['state']} (score {regime['score']:+d}); "
        f"IHSG close {regime.get('ihsg_close')}; "
        f"20D annualized volatility {regime.get('annualized_volatility_pct')}%"
    )


def _risk_notes(action: str, shortlist_entry: dict | None) -> str:
    if action == ACTION_CLOSE or shortlist_entry is None:
        return "position closed — dropped out of today's shortlist"
    parts = [
        f"position size {shortlist_entry.get('position_pct', 0):.1f}% "
        f"(Rp {shortlist_entry.get('position_idr', 0):,.0f})"
    ]
    levels = shortlist_entry.get("levels")
    if levels:
        parts.append(
            f"R:R {levels['rr_ratio']}:1 (entry {levels['entry']:,.0f}, "
            f"stop {levels['stop']:,.0f}, target {levels['target']:,.0f})"
        )
    else:
        parts.append("no clear technical setup today — no price level computed")
    sector = shortlist_entry.get("sector", "")
    if sector:
        parts.append(f"sector: {sector}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_rationale_log(
    brief: dict,
    features_df: pd.DataFrame,
    raw_dir: Path | str = RAW_DIR,
) -> list[dict]:
    """One entry per ticker that's open, held, or newly closed today."""
    date_str = brief.get("generated_at_utc", "")[:10] or pd.Timestamp.today().strftime("%Y-%m-%d")
    if not features_df.empty:
        date_str = pd.to_datetime(features_df["date"]).max().strftime("%Y-%m-%d")

    shortlist_by_ticker = {e["ticker"]: e for e in brief.get("shortlist", [])}
    today_tickers = set(shortlist_by_ticker.keys())
    previous_open = set(_load_previous_open_tickers())
    all_tickers = sorted(today_tickers | previous_open)

    latest_features = (
        features_df.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
        if not features_df.empty else pd.DataFrame()
    )

    snapshot = pd.DataFrame()
    if UNIVERSE_SNAPSHOT_PATH.exists():
        snapshot = pd.read_parquet(UNIVERSE_SNAPSHOT_PATH).set_index("ticker")

    regime = regime_classifier.classify_regime_from_files(raw_dir=raw_dir, features_df=features_df)

    entries = []
    for ticker in all_tickers:
        action = _resolve_action(ticker, today_tickers, previous_open)
        features_row = latest_features.loc[ticker] if ticker in latest_features.index else None
        fund_row = snapshot.loc[ticker] if ticker in snapshot.index else None
        shortlist_entry = shortlist_by_ticker.get(ticker)

        raw_path = Path(raw_dir) / f"{ticker}.parquet"
        proxies = None
        if raw_path.exists():
            proxies = money_flow_proxies.compute_money_flow_proxies(pd.read_parquet(raw_path))

        entries.append({
            "date": date_str,
            "ticker": ticker,
            "action": action,
            "technical_notes": _technical_notes(features_row),
            "fundamental_notes": _fundamental_notes(fund_row),
            "macro_notes": _macro_notes(regime),
            "risk_notes": _risk_notes(action, shortlist_entry),
            "money_flow_proxies": proxies,  # auto-computed talking points — see money_flow_proxies.py
            "money_flow_notes": None,       # manual — team fills in, see build_manual_flow_template()
        })

    _save_positions_state(date_str, sorted(today_tickers))
    return entries


def save_rationale_log(date_str: str, entries: list[dict]) -> None:
    RATIONALE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = RATIONALE_LOG_DIR / f"{date_str}.json"
    path.write_text(json.dumps(entries, indent=2, default=str))
    logger.info(f"Rationale log saved → {path} ({len(entries)} entries)")


# ---------------------------------------------------------------------------
# Manual money-flow research template — never overwrite a filled-in file
# ---------------------------------------------------------------------------

def build_manual_flow_template(date_str: str, tickers: list[str]) -> str:
    lines = [
        f"# Money Flow Research — {date_str}",
        "",
        "Fill in by hand: IDX broker summary reads, bandarmology write-ups,",
        "anything not automatable. This is real qualitative research for the",
        "Final Stage pitch's Money Flow Analysis (20% of that score) — not",
        "something the pipeline can generate.",
        "",
    ]
    for ticker in tickers:
        lines += [
            f"## {ticker}",
            "- Broker summary observations:",
            "- Bandarmology / accumulation-distribution notes:",
            "- Conclusion / conviction:",
            "",
        ]
    return "\n".join(lines)


def save_manual_flow_template(date_str: str, tickers: list[str]) -> None:
    RATIONALE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = RATIONALE_LOG_DIR / f"{date_str}_manual_flow_notes.md"
    if path.exists():
        logger.debug(f"{path} already exists — not overwriting the team's notes.")
        return
    path.write_text(build_manual_flow_template(date_str, tickers))
    logger.info(f"Manual flow-notes template created → {path}")
