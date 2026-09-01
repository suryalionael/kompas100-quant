"""Kompas100 Quant — Streamlit dashboard.

Universe overview, per-stock technical detail, and a data-health panel
over real OHLCV data (COMPETITION_PLAN.md §10, Days 1-2). The Rankings
tab shows the real horizon-ablation comparison (§4) once
data/published/ablation_results.json exists — never fabricated picks;
if no horizon has beaten naive momentum yet, it says so instead of
showing a ranked list. Portfolio construction (§6) is still a placeholder.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_pipeline import quality_filters
from portfolio import daily_brief
from ranking import ranking_model

WIB = ZoneInfo("Asia/Jakarta")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FEATURES_DIR = ROOT / "data" / "features"
PUBLISHED_DIR = ROOT / "data" / "published"
UNIVERSE_CSV = ROOT / "configs" / "kompas100_live.csv"

TEAL = "#1E6F63"
TEAL_DARK = "#124A41"
INK = "#171F1D"
PAPER = "#F7F6F2"
CARD = "#FFFFFF"
BORDER = "#CBD1CC"
MUTED_RED = "#B3564A"
AMBER = "#B8862E"
INK_MUTED = "#5B655F"

st.set_page_config(page_title="Kompas100 Quant", layout="wide", initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: {INK};
        }}
        .stApp {{ background-color: {PAPER}; }}

        [data-testid="stSidebar"] {{
            background-color: {CARD};
            border-right: 1px solid {BORDER};
        }}

        .block-container {{ padding-top: 2rem; max-width: 1280px; }}

        h1.k100-title {{
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: {TEAL_DARK};
            margin-bottom: 0.1rem;
        }}
        p.k100-subtitle {{
            font-size: 0.95rem;
            color: {INK_MUTED};
            margin-top: 0;
            margin-bottom: 1.5rem;
        }}
        h2.k100-section {{
            font-size: 1.25rem;
            font-weight: 600;
            color: {INK};
            border-bottom: 1px solid {BORDER};
            padding-bottom: 0.4rem;
            margin-top: 1.8rem;
            margin-bottom: 1rem;
        }}
        h3.k100-subsection {{
            font-size: 1rem;
            font-weight: 600;
            color: {INK};
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }}

        .k100-card {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            height: 100%;
        }}
        .k100-card .label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: {INK_MUTED};
            font-weight: 600;
            margin-bottom: 0.35rem;
        }}
        .k100-card .value {{
            font-size: 1.55rem;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            color: {INK};
        }}
        .k100-card .delta {{
            font-size: 0.85rem;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
            margin-top: 0.2rem;
        }}
        .up {{ color: {TEAL}; }}
        .down {{ color: {MUTED_RED}; }}
        .warn {{ color: {AMBER}; }}
        .neutral {{ color: {INK_MUTED}; }}

        [data-testid="stDataFrame"] * {{ font-variant-numeric: tabular-nums; }}

        .k100-badge {{
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            border: 1px solid {BORDER};
        }}

        .k100-status {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.82rem;
            color: {INK_MUTED};
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 0.5rem 0.9rem;
            margin-bottom: 1.3rem;
        }}
        .k100-status .dot {{
            width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
        }}
        .k100-status .dot.up {{ background: {TEAL}; }}
        .k100-status .dot.warn {{ background: {AMBER}; }}
        .k100-status .dot.down {{ background: {MUTED_RED}; }}
        .k100-status .dot.neutral {{ background: {INK_MUTED}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str | None = None, tone: str = "neutral") -> str:
    delta_html = f'<div class="delta {tone}">{delta}</div>' if delta else ""
    return f'<div class="k100-card"><div class="label">{label}</div><div class="value">{value}</div>{delta_html}</div>'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_universe_list() -> list[str]:
    df = pd.read_csv(UNIVERSE_CSV)
    return df[df["is_active"].astype(str).str.lower().isin(["true", "1"])]["ticker"].tolist()


@st.cache_data(ttl=300)
def load_raw(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path).sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=300)
def load_features_latest() -> pd.DataFrame:
    files = sorted(FEATURES_DIR.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


def load_ticker_chart_data(ticker: str) -> pd.DataFrame:
    """Raw OHLCV (for the candlestick) merged with feature_builder's
    already-computed MA/RSI/MACD — never recomputed here, so the dashboard
    shows the same numbers the ranking model will eventually see.
    """
    raw = load_raw(ticker)
    if raw.empty:
        return raw

    features = load_features_latest()
    tech_cols = ["date", "ma5", "ma20", "ma50", "rsi14", "macd", "macd_signal", "macd_histogram"]
    ticker_features = features[features["ticker"] == ticker][tech_cols] if not features.empty else pd.DataFrame()

    if ticker_features.empty:
        return raw
    return raw.merge(ticker_features, on="date", how="left")


@st.cache_data(ttl=300)
def load_universe_snapshot() -> pd.DataFrame:
    path = PUBLISHED_DIR / "universe_snapshot_latest.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def load_scan_meta() -> dict | None:
    path = PUBLISHED_DIR / "scan_meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(ttl=300)
def load_scheduled_run_state() -> dict | None:
    """Written only when scripts/run_daily_scan.py's trigger is
    "schedule" (GITHUB_EVENT_NAME) — distinct from scan_meta.json, which
    the most recent run of *any* trigger type overwrites. Lets the
    dashboard tell "the automation actually ran on its own schedule"
    apart from "someone ran it by hand," which scan_meta.json alone can't
    once a manual run is the latest one."""
    path = PUBLISHED_DIR / "scheduled_run_state.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(ttl=300)
def load_data_freshness_audit() -> dict | None:
    path = PUBLISHED_DIR / "data_freshness_audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(ttl=300)
def load_ablation_results() -> dict | None:
    path = PUBLISHED_DIR / "ablation_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(ttl=300)
def load_daily_brief() -> dict | None:
    path = PUBLISHED_DIR / "daily_brief.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def liquidity_tier(avg_value_traded_idr: float) -> str:
    if pd.isna(avg_value_traded_idr):
        return "Unknown"
    if avg_value_traded_idr >= 20_000_000_000:
        return "Tier 1"
    if avg_value_traded_idr >= 3_000_000_000:
        return "Tier 2"
    return "Tier 3"


def build_universe_table(tickers: list[str]) -> pd.DataFrame:
    """Last close, % change, avg volume, and a value-traded liquidity tier
    computed directly from raw OHLCV — always available even before the
    fundamentals/quality snapshot has run.
    """
    rows = []
    for t in tickers:
        df = load_raw(t)
        if df.empty or len(df) < 2:
            rows.append({
                "ticker": t, "last_close": None, "pct_change": None,
                "avg_volume_20d": None, "liquidity_tier": "Unknown", "rows": len(df),
            })
            continue
        last = df.iloc[-1]
        prev = df.iloc[-2]
        pct_change = (last["close"] - prev["close"]) / prev["close"] * 100 if prev["close"] else None
        tail20 = df.tail(20)
        avg_vol = tail20["volume"].mean()
        avg_value_traded = (tail20["volume"] * tail20["close"]).mean()
        rows.append({
            "ticker": t,
            "last_close": float(last["close"]),
            "pct_change": float(pct_change) if pct_change is not None else None,
            "avg_volume_20d": float(avg_vol),
            "liquidity_tier": liquidity_tier(avg_value_traded),
            "rows": len(df),
        })
    return pd.DataFrame(rows)


def style_universe_table(df: pd.DataFrame):
    def color_pct(val):
        if pd.isna(val):
            return f"color: {INK_MUTED}"
        return f"color: {TEAL}; font-weight: 600" if val >= 0 else f"color: {MUTED_RED}; font-weight: 600"

    styler = df.style.map(color_pct, subset=["pct_change"])
    return styler.format({
        "last_close": "{:,.0f}",
        "pct_change": "{:+.2f}%",
        "avg_volume_20d": "{:,.0f}",
    })


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def price_detail_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """df must carry the OHLCV + ma5/ma20/ma50/rsi14/macd/macd_signal/macd_histogram
    columns produced by load_ticker_chart_data() — indicators are computed once,
    by data_pipeline.feature_builder, never recomputed in the dashboard."""
    up = df["close"] >= df["close"].shift(1)
    vol_colors = [TEAL if u else MUTED_RED for u in up.fillna(True)]
    hist_colors = [TEAL if v >= 0 else MUTED_RED for v in df["macd_histogram"].fillna(0)]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.15, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=("", "Volume", "RSI (14)", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=TEAL, decreasing_line_color=MUTED_RED,
        increasing_fillcolor=TEAL, decreasing_fillcolor=MUTED_RED,
        name=ticker, showlegend=False,
    ), row=1, col=1)

    for col, color, width in [("ma5", "#8FB8AF", 1.2), ("ma20", TEAL_DARK, 1.4), ("ma50", "#B8862E", 1.2)]:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col], mode="lines", name=col.upper(),
            line=dict(color=color, width=width),
        ), row=1, col=1)

    last_close = df["close"].iloc[-1]
    fig.add_hline(y=last_close, line_dash="dot", line_color=INK_MUTED, line_width=1,
                   annotation_text=f"{last_close:,.0f}", annotation_position="right", row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"], marker_color=vol_colors, name="Volume", showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["rsi14"], mode="lines", name="RSI",
        line=dict(color=TEAL_DARK, width=1.4), showlegend=False,
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color=MUTED_RED, line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color=TEAL, line_width=1, row=3, col=1)

    fig.add_trace(go.Bar(
        x=df["date"], y=df["macd_histogram"], marker_color=hist_colors, name="MACD Hist", showlegend=False,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["macd"], mode="lines", name="MACD",
        line=dict(color=TEAL_DARK, width=1.3), showlegend=False,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["macd_signal"], mode="lines", name="Signal",
        line=dict(color=AMBER, width=1.1), showlegend=False,
    ), row=4, col=1)

    fig.update_layout(
        height=760,
        margin=dict(l=10, r=60, t=30, b=10),
        plot_bgcolor=CARD,
        paper_bgcolor=CARD,
        font=dict(color=INK, size=12),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified",
    )
    for r in range(1, 5):
        fig.update_xaxes(gridcolor=BORDER, showgrid=True, row=r, col=1)
        fig.update_yaxes(gridcolor=BORDER, showgrid=True, zeroline=False, row=r, col=1)

    return fig


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown('<h1 class="k100-title">Kompas100 Quant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="k100-subtitle">Competition data dashboard — universe overview, per-stock technicals, and data health.</p>',
        unsafe_allow_html=True,
    )


def render_status_strip() -> None:
    """Data-freshness strip, driven by data/published/scan_meta.json —
    written by scripts/run_daily_scan.py, refreshed daily via the
    scheduled GitHub Actions workflow (.github/workflows/daily_data_refresh.yml).
    """
    meta = load_scan_meta()
    if meta is None:
        st.markdown(
            '<div class="k100-status"><span class="dot warn"></span>'
            'No scan metadata found — run scripts/run_daily_scan.py to generate data/published/scan_meta.json.</div>',
            unsafe_allow_html=True,
        )
        return

    market_date = meta.get("market_date")
    scanned_at_utc = meta.get("scanned_at_utc")
    tickers_fetched = meta.get("tickers_fetched", 0)
    tickers_total = meta.get("tickers_total", 0)

    scanned_wib = "—"
    staleness_days = None
    if scanned_at_utc:
        scanned_dt = datetime.fromisoformat(scanned_at_utc)
        scanned_wib = scanned_dt.astimezone(WIB).strftime("%Y-%m-%d %H:%M WIB")
        staleness_days = (datetime.now(timezone.utc) - scanned_dt).days

    if staleness_days is not None and staleness_days > 3:
        tone = "warn"
        freshness_note = f" — stale, last run {staleness_days}d ago"
    elif tickers_fetched < tickers_total:
        tone = "warn"
        freshness_note = ""
    else:
        tone = "up"
        freshness_note = ""

    st.markdown(
        f'<div class="k100-status"><span class="dot {tone}"></span>'
        f"Market date <strong>{market_date}</strong> &nbsp;·&nbsp; "
        f"Last scan {scanned_wib} &nbsp;·&nbsp; "
        f"{tickers_fetched}/{tickers_total} tickers fetched{freshness_note}</div>",
        unsafe_allow_html=True,
    )


# ISTC 2026 official rules: live trading window 21 Sept - 8 Oct 2026 (14
# working days), one virtual account, all positions manually closed by end
# of day 8 Oct. Model freeze target ~18-19 Sept per COMPETITION_PLAN.md.
COMPETITION_START = pd.Timestamp("2026-09-21")
COMPETITION_END = pd.Timestamp("2026-10-08")
FREEZE_TARGET = pd.Timestamp("2026-09-19")


def render_countdown_banner() -> None:
    """A rule, not a strategy choice — the ISTC 2026 platform requires all
    positions manually closed by end of day 8 Oct 2026, and this makes it
    impossible to lose track of that while looking at the dashboard.
    """
    today = pd.Timestamp(datetime.now(WIB).date())

    if today < FREEZE_TARGET:
        days_to_freeze = max(len(pd.bdate_range(today, FREEZE_TARGET)) - 1, 0)
        st.markdown(
            f'<div class="k100-status"><span class="dot neutral"></span>'
            f"ISTC 2026: model freeze target {FREEZE_TARGET.strftime('%d %b %Y')} "
            f"({days_to_freeze} trading days away) · live window "
            f"{COMPETITION_START.strftime('%d %b')}–{COMPETITION_END.strftime('%d %b %Y')}"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif today < COMPETITION_START:
        days_to_open = max(len(pd.bdate_range(today, COMPETITION_START)) - 1, 0)
        st.markdown(
            f'<div class="k100-status"><span class="dot warn"></span>'
            f"ISTC 2026 live trading opens {COMPETITION_START.strftime('%d %b %Y')} "
            f"({days_to_open} trading days from now) — model should already be frozen."
            f"</div>",
            unsafe_allow_html=True,
        )
    elif today <= COMPETITION_END:
        days_left = max(len(pd.bdate_range(today, COMPETITION_END)) - 1, 0)
        tone = "down" if days_left <= 3 else "warn"
        st.markdown(
            f'<div class="k100-status"><span class="dot {tone}"></span>'
            f"<strong>{days_left} trading day{'s' if days_left != 1 else ''} left</strong> — "
            f"all positions must be manually closed by end of day "
            f"{COMPETITION_END.strftime('%d %b %Y')}."
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="k100-status"><span class="dot up"></span>'
            f"ISTC 2026 live trading window closed {COMPETITION_END.strftime('%d %b %Y')}."
            f"</div>",
            unsafe_allow_html=True,
        )


def render_universe_tab(tickers: list[str]) -> None:
    st.markdown('<h2 class="k100-section">Universe Overview</h2>', unsafe_allow_html=True)

    table = build_universe_table(tickers)
    valid = table.dropna(subset=["pct_change"])

    n_total = len(table)
    n_up = int((valid["pct_change"] >= 0).sum())
    n_down = int((valid["pct_change"] < 0).sum())
    avg_change = valid["pct_change"].mean() if not valid.empty else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Tickers Tracked", f"{n_total}"), unsafe_allow_html=True)
    with c2:
        tone = "up" if avg_change >= 0 else "down"
        st.markdown(kpi_card("Avg Daily Change", f"{avg_change:+.2f}%", tone=tone), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Advancers", f"{n_up}", tone="up"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Decliners", f"{n_down}", tone="down"), unsafe_allow_html=True)

    st.markdown('<h3 class="k100-subsection">All Constituents</h3>', unsafe_allow_html=True)

    display = table[["ticker", "last_close", "pct_change", "avg_volume_20d", "liquidity_tier"]]

    event = st.dataframe(
        style_universe_table(display),
        use_container_width=True,
        height=560,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="universe_table",
        column_config={
            "ticker": st.column_config.TextColumn(label="Ticker"),
            "last_close": st.column_config.TextColumn(label="Last Close"),
            "pct_change": st.column_config.TextColumn(label="% Change"),
            "avg_volume_20d": st.column_config.TextColumn(label="Avg Vol (20d)"),
            "liquidity_tier": st.column_config.TextColumn(label="Liquidity Tier"),
        },
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        selected_ticker = display.iloc[selected_rows[0]]["ticker"]
        st.session_state["selected_ticker"] = selected_ticker
        st.markdown(f'<h3 class="k100-subsection">Detail: {selected_ticker}</h3>', unsafe_allow_html=True)
        render_ticker_detail(selected_ticker, key_prefix="universe")
    else:
        st.caption("Select the checkbox next to a ticker to see its fundamentals and chart.")


def render_stock_detail_tab(tickers: list[str]) -> None:
    st.markdown('<h2 class="k100-section">Per-Stock Detail</h2>', unsafe_allow_html=True)

    default_ticker = st.session_state.get("selected_ticker")
    default_index = tickers.index(default_ticker) if default_ticker in tickers else 0
    selected = st.selectbox("Ticker", tickers, index=default_index)
    render_ticker_detail(selected, key_prefix="detail")


def render_fundamentals_panel(ticker: str) -> None:
    """Fundamentals + quality/risk status from data/published/universe_snapshot_latest.parquet
    (built by scripts/build_universe_snapshot.py — data_pipeline/fundamental.py +
    quality_filters.py). Shows "—" for anything not fetched rather than a fake number.
    """
    st.markdown('<h4 style="margin:0.8rem 0 0.5rem;font-size:0.95rem;font-weight:600;">Fundamentals</h4>', unsafe_allow_html=True)

    snapshot = load_universe_snapshot()
    row = snapshot[snapshot["ticker"] == ticker] if not snapshot.empty else pd.DataFrame()
    if row.empty:
        st.info("No fundamentals snapshot yet — run `scripts/build_universe_snapshot.py`.")
        return
    r = row.iloc[0]

    def _fmt(val, suffix="", decimals=2):
        return "—" if pd.isna(val) else f"{val:.{decimals}f}{suffix}"

    def _fmt_cap(val):
        if pd.isna(val):
            return "—"
        if val >= 1e12:
            return f"Rp {val / 1e12:.1f}T"
        if val >= 1e9:
            return f"Rp {val / 1e9:.1f}B"
        return f"Rp {val:,.0f}"

    health_score = r.get("overall_health_score")

    fields = [
        ("P/E", _fmt(r.get("pe_ratio"), "x")),
        ("P/BV", _fmt(r.get("pbv"), "x")),
        ("ROE", _fmt(r.get("roe_pct"), "%")),
        ("DER", _fmt(r.get("der"), "x")),
        ("Div Yield", _fmt(r.get("div_yield_pct"), "%")),
        ("Market Cap", _fmt_cap(r.get("market_cap"))),
        ("Overall Health", "—" if pd.isna(health_score) else f"{health_score:.0f}/100"),
    ]
    cols = st.columns(len(fields))
    for i, (label, value) in enumerate(fields):
        with cols[i]:
            if label == "Overall Health" and pd.notna(health_score):
                tone = "up" if health_score >= 75 else ("warn" if health_score >= 50 else "down")
                st.markdown(kpi_card(label, value, tone=tone), unsafe_allow_html=True)
            else:
                st.markdown(kpi_card(label, value), unsafe_allow_html=True)

    final_status = r.get("final_status")
    status_label = str(final_status).replace("_", " ").title() if pd.notna(final_status) else "Unknown"
    tone = "up" if final_status == "eligible" else ("warn" if final_status == "watch_with_risk" else "down")
    fund_status = r.get("fundamental_status", "unknown")
    risk_flags = str(r.get("risk_flags", "") or "").strip()

    detail_line = f'Quality status: <span class="k100-badge {tone}">{status_label}</span> &nbsp; Fundamental data: {fund_status}'
    if risk_flags:
        detail_line += f" &nbsp; Risk: {risk_flags}"
    st.markdown(
        f'<p style="margin-top:0.6rem;font-size:0.85rem;color:{INK_MUTED};">{detail_line}</p>',
        unsafe_allow_html=True,
    )

    if pd.notna(health_score):
        with st.expander(f"Overall Health breakdown — {health_score:.0f}/100 (pitch-deck material, not used for ranking/sizing)"):
            health = quality_filters.compute_overall_health(r.to_dict())
            breakdown_df = pd.DataFrame(health["health_breakdown"])[["factor", "weight", "score", "note"]]
            breakdown_df.columns = ["Factor", "Weight %", "Score /100", "Note"]
            breakdown_df["Factor"] = breakdown_df["Factor"].str.replace("_", " ").str.title()
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)


def render_ticker_detail(ticker: str, key_prefix: str) -> None:
    """Full per-ticker detail — price/volume KPIs, fundamentals, quality
    status, and the technical chart. Shared by the Stock Detail tab and the
    inline "click a ticker to see detail" panels on Universe/Rankings.
    """
    df = load_ticker_chart_data(ticker)
    if df.empty:
        st.warning(f"No data available for {ticker}.")
        return

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    pct_change = (last["close"] - prev["close"]) / prev["close"] * 100 if prev["close"] else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Last Close", f"{last['close']:,.0f}"), unsafe_allow_html=True)
    with c2:
        tone = "up" if pct_change >= 0 else "down"
        st.markdown(kpi_card("1D Change", f"{pct_change:+.2f}%", tone=tone), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Volume", f"{last['volume']:,.0f}"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("History", f"{len(df)} days"), unsafe_allow_html=True)

    render_fundamentals_panel(ticker)

    lookback = st.slider(
        "Lookback (trading days)", min_value=60, max_value=min(len(df), 750),
        value=min(250, len(df)), step=10, key=f"{key_prefix}_lookback_{ticker}",
    )
    chart_df = df.tail(lookback)

    st.plotly_chart(
        price_detail_chart(chart_df, ticker), use_container_width=True,
        key=f"{key_prefix}_chart_{ticker}",
    )




def render_staleness_banner(latest_bar_date: str | None) -> None:
    """Flags when the daily GitHub Actions refresh (.github/workflows/
    daily_data_refresh.yml) has stopped running. Expected trading day only
    skips weekends, not IDX holidays — a holiday will show a harmless
    1-trading-day gap, which is within tolerance (not flagged).
    """
    if not latest_bar_date:
        st.error("**No data at all** — nothing to check for staleness.")
        return

    last_bar = pd.Timestamp(latest_bar_date)
    today_wib = pd.Timestamp(datetime.now(WIB).date())
    expected_last_trading_day = pd.bdate_range(end=today_wib, periods=1)[0]

    if last_bar >= expected_last_trading_day:
        gap = 0
    else:
        gap = len(pd.bdate_range(last_bar + pd.Timedelta(days=1), expected_last_trading_day))

    if gap > 1:
        st.error(
            f"**Data may be stale — last bar is {gap}d old.** Expected daily refresh via "
            f"GitHub Actions may have failed or been delayed. Check the Actions tab."
        )
    else:
        st.success(
            f"Data is current — last bar ({last_bar.strftime('%Y-%m-%d')}) matches the "
            f"expected last trading day ({expected_last_trading_day.strftime('%Y-%m-%d')})."
        )


def render_scheduled_run_health() -> None:
    """Distinguishes "the automation actually fired on its own schedule"
    from "someone ran it by hand" — found as a real gap 2026-08-31: the
    only run in GitHub Actions history was workflow_dispatch (manual);
    the cron's first scheduled slot after the workflow was registered had
    already passed with nothing firing, and nothing before this would
    have surfaced that silently-stale automation to anyone looking at the
    dashboard.
    """
    meta = load_scan_meta()
    scheduled_state = load_scheduled_run_state()

    trigger = meta.get("trigger", "unknown") if meta else "unknown"
    trigger_label = {
        "schedule": "Scheduled (automatic)",
        "workflow_dispatch": "Manual dispatch",
        "local": "Local run",
    }.get(trigger, trigger)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(kpi_card("Latest Run Trigger", trigger_label), unsafe_allow_html=True)

    if scheduled_state is None:
        with c2:
            st.markdown(kpi_card("Last Scheduled Success", "Never", tone="down"), unsafe_allow_html=True)
        st.error(
            "**No scheduled run has ever completed successfully.** Every run so far has been "
            "triggered manually — the daily automation (.github/workflows/daily_data_refresh.yml) "
            "has not been confirmed to work on its own. Check the Actions tab."
        )
        return

    last_scheduled_utc = datetime.fromisoformat(scheduled_state["last_scheduled_run_utc"])
    hours_since = (datetime.now(timezone.utc) - last_scheduled_utc).total_seconds() / 3600

    with c2:
        tone = "up" if hours_since <= 30 else "down"
        st.markdown(kpi_card("Hours Since Last Scheduled Success", f"{hours_since:.1f}h", tone=tone), unsafe_allow_html=True)

    if hours_since > 30:
        st.error(
            f"**Scheduled refresh may have stopped running — {hours_since:.1f}h since the last "
            f"confirmed scheduled success** (cron fires daily). Check the Actions tab."
        )
    else:
        st.success(f"Scheduled refresh confirmed working — last success {hours_since:.1f}h ago.")


def render_data_health_tab(tickers: list[str]) -> None:
    st.markdown('<h2 class="k100-section">Data Health</h2>', unsafe_allow_html=True)

    render_scheduled_run_health()

    rows = []
    for t in tickers:
        df = load_raw(t)
        if df.empty:
            rows.append({"Ticker": t, "Status": "Failed", "Last Update": "—", "Rows": 0})
        else:
            rows.append({
                "Ticker": t, "Status": "OK",
                "Last Update": df["date"].max().strftime("%Y-%m-%d"),
                "Rows": len(df),
            })
    health = pd.DataFrame(rows)

    n_ok = int((health["Status"] == "OK").sum())
    n_failed = int((health["Status"] == "Failed").sum())

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi_card("Fetched OK", f"{n_ok} / {len(tickers)}", tone="up" if n_failed == 0 else "warn"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Fetch Failures", f"{n_failed}", tone="down" if n_failed else "neutral"), unsafe_allow_html=True)
    with c3:
        latest_dt = health.loc[health["Status"] == "OK", "Last Update"].max() if n_ok else "—"
        st.markdown(kpi_card("Most Recent Bar", f"{latest_dt}"), unsafe_allow_html=True)

    render_staleness_banner(latest_dt if n_ok else None)

    st.markdown('<h3 class="k100-subsection">Fetch Status by Ticker</h3>', unsafe_allow_html=True)
    st.dataframe(health, use_container_width=True, height=380, hide_index=True)

    st.markdown('<h3 class="k100-subsection">Per-Ticker Freshness Audit</h3>', unsafe_allow_html=True)
    audit = load_data_freshness_audit()
    if audit is None:
        st.info("No freshness audit yet — run `scripts/run_daily_scan.py` to generate one.")
    else:
        market_last = audit.get("market_last_trade_date")
        stale = audit.get("stale_tickers", [])
        st.caption(
            f"Each ticker's own last-trade date (from the fetched OHLCV itself, not assumed "
            f"from 'last row in the file') vs. the market's last trading day ({market_last}, "
            f"from IHSG)."
        )
        if stale:
            st.error(f"**{len(stale)} ticker(s) lag the market by more than 1 trading day:** {', '.join(stale)}")
        else:
            st.success("All tickers match the market's last trading day.")

    st.markdown('<h3 class="k100-subsection">Quality Filter Results</h3>', unsafe_allow_html=True)
    snapshot = load_universe_snapshot()
    if snapshot.empty:
        st.info(
            "No quality-filter snapshot yet. Run `python scripts/build_universe_snapshot.py` "
            "to fetch fundamentals and compute pass/fail status per ticker."
        )
    else:
        status_cols = [c for c in ["ticker", "final_status", "exclusion_reason", "risk_flags", "fundamental_status"] if c in snapshot.columns]
        excluded = snapshot[snapshot["final_status"] != "eligible"][status_cols] if "final_status" in snapshot.columns else pd.DataFrame()
        counts = snapshot["final_status"].value_counts() if "final_status" in snapshot.columns else pd.Series(dtype=int)

        cols = st.columns(min(4, max(len(counts), 1)))
        for i, (status, count) in enumerate(counts.items()):
            tone = "up" if status == "eligible" else ("warn" if status == "watch_with_risk" else "down")
            with cols[i % len(cols)]:
                st.markdown(kpi_card(status.replace("_", " ").title(), f"{count}", tone=tone), unsafe_allow_html=True)

        if not excluded.empty:
            st.markdown('<h3 class="k100-subsection">Not Eligible</h3>', unsafe_allow_html=True)
            st.dataframe(excluded.rename(columns={
                "ticker": "Ticker", "final_status": "Status", "exclusion_reason": "Reason",
                "risk_flags": "Risk Flags", "fundamental_status": "Fundamental Data",
            }), use_container_width=True, hide_index=True)
        else:
            st.success("All tickers passed quality filters.")

        if "overall_health_score" in snapshot.columns:
            st.markdown('<h3 class="k100-subsection">Overall Health by Ticker</h3>', unsafe_allow_html=True)
            st.caption(
                "Pitch-deck / explainability material for the ISTC 2026 Final Stage rubric "
                "(Fundamental Analysis, 10% of that grade) — re-expresses the same DER/PBV/float/"
                "regulatory criteria above as a weighted 0-100 score. Never used for ranking or "
                "position sizing (quant is the only source of truth for numbers — CLAUDE.md)."
            )
            health_cols = [c for c in ["ticker", "overall_health_score", "final_status", "health_weakest_factor"] if c in snapshot.columns]
            health_df = snapshot[health_cols].sort_values("overall_health_score").reset_index(drop=True)
            health_df = health_df.rename(columns={
                "ticker": "Ticker", "overall_health_score": "Overall Health",
                "final_status": "Status", "health_weakest_factor": "Weakest Factor",
            })
            health_df["Status"] = health_df["Status"].astype(str).str.replace("_", " ").str.title()
            health_df["Overall Health"] = health_df["Overall Health"].map(lambda v: "—" if pd.isna(v) else f"{v:.0f}/100")
            st.dataframe(health_df, use_container_width=True, hide_index=True, height=350)


def render_feature_glossary() -> None:
    """Plain-English descriptions of every feature the ranking model uses
    (ranking/ranking_model.py's FEATURE_DESCRIPTIONS) — a non-quant
    teammate should be able to read these and recognize a real market
    behavior, not just a column name. A model nobody on the team can
    explain in the ISTC 2026 Final Stage pitch is a liability even if it
    backtests well (COMPETITION_PLAN.md §0).
    """
    missing = [f for f in ranking_model.RANKING_FEATURES if f not in ranking_model.FEATURE_DESCRIPTIONS]
    with st.expander("What do these features mean? (glossary)"):
        if missing:
            st.warning(f"FEATURE_DESCRIPTIONS is missing an entry for: {missing} — out of sync with RANKING_FEATURES.")
        rows = [
            {"Feature": f, "Plain-English meaning": ranking_model.FEATURE_DESCRIPTIONS.get(f, "—")}
            for f in ranking_model.RANKING_FEATURES
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=380)


def render_rankings_tab(tickers: list[str]) -> None:
    st.markdown('<h2 class="k100-section">Rankings</h2>', unsafe_allow_html=True)

    results = load_ablation_results()
    if results is None or not results.get("horizons"):
        st.info(
            "**Rankings — coming in a later phase.**\n\n"
            "This tab will show the cross-sectional ranking model's output once "
            "`ranking/ranking_model.py` and the horizon ablation (COMPETITION_PLAN.md §4) "
            "are built and validated against the backtest engine. No ranking data is "
            "fabricated here in the meantime."
        )
        return

    horizons = results["horizons"]
    # A bare non-overlapping mean_return comparison alone is fragile —
    # found 2026-08-31: 3D's non-overlapping point estimate technically
    # beat momentum, but flipped between two ordinary data refreshes
    # (yfinance revises historical adjusted-close on every refetch) and
    # lost on the overlapping view outright. Requiring agreement on BOTH
    # views is a cheap, already-computed robustness check against exactly
    # that kind of noise-level "win" — this is what portfolio/
    # daily_brief.py's own no-live-inference-path guard structurally
    # prevents from ever reaching "validated", but the dashboard message
    # itself needs the same caution, not just the underlying system.
    any_horizon_wins = any(
        h["ranking_model"]["non_overlapping"]["mean_return"] > h["momentum"]["non_overlapping"]["mean_return"]
        and h["ranking_model"]["overlapping"]["mean_return"] > h["momentum"]["overlapping"]["mean_return"]
        for h in horizons.values()
    )
    any_horizon_wins_fragile = not any_horizon_wins and any(
        h["ranking_model"]["non_overlapping"]["mean_return"] > h["momentum"]["non_overlapping"]["mean_return"]
        for h in horizons.values()
    )

    if any_horizon_wins:
        st.success(
            "At least one horizon beats naive momentum on both the overlapping and honest "
            "(non-overlapping) fold sets. See the table below — still not live picks until "
            "portfolio construction (§6) exists."
        )
    elif any_horizon_wins_fragile:
        st.warning(
            "**A horizon's point estimate beats momentum on the non-overlapping view, but not on "
            "the overlapping view too — not treated as a real edge.** Found 2026-08-31: this exact "
            "pattern (3D) flipped between beating and losing to momentum across two ordinary data "
            "refreshes (yfinance revises historical prices on every refetch) — noise-level, not a "
            "robust result. COMPETITION_PLAN.md §4 has the full investigation. No live picks shown."
        )
    else:
        st.warning(
            "**No horizon beats naive momentum yet — the ranking model does not clear the "
            "ablation gate, so no live picks are shown.** Per COMPETITION_PLAN.md's own rule, "
            "a level that doesn't beat the one below it gets cut, even if already built. "
            "This is the real backtest comparison, most recently run 2026-08-31 — not a placeholder."
        )

    bh = results.get("buy_and_hold")
    if bh:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(kpi_card("Kompas100 Buy-and-Hold", _pct(bh.get("compounded_return"))), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card("Max Drawdown", _pct(bh.get("max_drawdown")), tone="down"), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_card("Full Window", "2023-08-31 → 2026-08-28"), unsafe_allow_html=True)

    render_feature_glossary()

    st.markdown('<h3 class="k100-subsection">Horizon Ablation — Non-Overlapping Folds (the honest view)</h3>', unsafe_allow_html=True)
    rows = []
    for h in sorted(horizons, key=int):
        r = horizons[h]
        no = r["ranking_model"]["non_overlapping"]
        mo = r["momentum"]["non_overlapping"]
        rows.append({
            "Horizon": f"{h}D",
            "Model return/fold": no["mean_return"],
            "95% CI low": no["ci_low"],
            "95% CI high": no["ci_high"],
            "Momentum return/fold": mo["mean_return"],
            "Model IC": no["mean_ic"],
            "n folds": no["n_folds"],
            "Beats momentum?": "Yes" if no["mean_return"] > mo["mean_return"] else "No",
        })
    table = pd.DataFrame(rows)

    def _color_beats(val):
        return f"color: {TEAL}; font-weight: 600" if val == "Yes" else f"color: {MUTED_RED}"

    pct_cols = ["Model return/fold", "95% CI low", "95% CI high", "Momentum return/fold"]
    styler = table.style.map(_color_beats, subset=["Beats momentum?"])
    styler = styler.format({c: "{:+.2%}".format for c in pct_cols} | {"Model IC": "{:.3f}".format})
    st.dataframe(styler, use_container_width=True, hide_index=True)
    st.caption(
        "Non-overlapping folds are independent (one decision every `horizon` trading days) — "
        "the statistically honest view. The overlapping view (a decision every trading day) can "
        "look more favorable purely from folds sharing most of their trading days; see "
        "COMPETITION_PLAN.md §4 for both."
    )

    if not any_horizon_wins:
        render_current_shortlist(tickers)


def render_current_shortlist(tickers: list[str]) -> None:
    """Reads data/published/daily_brief.json — the same file
    scripts/run_daily_scan.py publishes and Cowork's daily report will
    read — rather than recomputing momentum here separately, so the
    dashboard shows exactly what the pipeline actually decided, sizes,
    and priced (portfolio/daily_brief.py, portfolio/level_calculator.py).
    """
    st.markdown('<h3 class="k100-subsection">Current Shortlist</h3>', unsafe_allow_html=True)

    brief = load_daily_brief()
    if brief is None:
        st.info("No daily brief published yet — run `scripts/run_daily_scan.py`.")
        return

    status = brief.get("strategy_status")
    status_captions = {
        "validated": "Ablation-gated ranking model — validated.",
        "naive_momentum_interim": (
            "Naive momentum (rank by 20D return) — currently beats every ML horizon in "
            "backtest, shown here because it's the strategy actually winning today, not "
            "because it's proven great."
        ),
        "no_picks": "No picks today — see reason below.",
    }
    st.caption(status_captions.get(status, f"strategy_status: {status}"))

    shortlist = brief.get("shortlist", [])
    if not shortlist:
        st.info("**No picks today.** daily_brief.json reports an empty shortlist — "
                 "honest zero-conviction state, not an error to ignore.")
        return

    # Defense in depth: the hard Kompas100 guard already runs at publish
    # time (portfolio/daily_brief.py's assert_kompas100_only) — this just
    # makes sure the dashboard itself can't be the thing that shows a bad
    # ticker if daily_brief.json was ever hand-edited or came from a stale
    # build.
    bad_tickers = [e["ticker"] for e in shortlist if e["ticker"] not in tickers]
    if bad_tickers:
        st.error(f"**Refusing to display — ticker(s) outside the live Kompas100 universe:** {bad_tickers}")
        shortlist = [e for e in shortlist if e["ticker"] not in bad_tickers]
        if not shortlist:
            return

    # Entry/Stop/Target/R:R are pre-formatted to plain strings (not left as
    # NaN for a Styler formatter to catch) — Streamlit's interactive
    # (on_select) dataframe grid doesn't reliably run the Styler's format
    # callables against NaN cells, so a mixed numeric/NaN column here shows
    # the literal word "None" no matter what the formatter says. A
    # string column has nothing for the grid to reformat by itself.
    rows = []
    for e in shortlist:
        levels = e.get("levels")
        rows.append({
            "Ticker": e["ticker"],
            "Sector": e.get("sector", ""),
            "Score": f"{e.get('score', 0):.2f}",
            "Position": f"{e.get('position_pct', 0):.1f}%",
            "Position (Rp)": f"Rp {e.get('position_idr', 0):,.0f}",
            "Entry": f"{levels['entry']:,.0f}" if levels else "—",
            "Stop": f"{levels['stop']:,.0f}" if levels else "—",
            "Target": f"{levels['target']:,.0f}" if levels else "—",
            "R:R": f"{levels['rr_ratio']:.1f}:1" if levels else "—",
        })
    display = pd.DataFrame(rows)

    event = st.dataframe(
        display, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="rankings_shortlist_table",
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        selected_ticker = display.iloc[selected_rows[0]]["Ticker"]
        st.session_state["selected_ticker"] = selected_ticker
        st.markdown(f'<h3 class="k100-subsection">Detail: {selected_ticker}</h3>', unsafe_allow_html=True)
        render_ticker_detail(selected_ticker, key_prefix="rankings")
    else:
        st.caption("Select the checkbox next to a ticker to see its fundamentals and chart.")

    render_sector_concentration(shortlist)


def render_sector_concentration(shortlist: list[dict]) -> None:
    """Current exposure per sector vs. daily_brief.py's SECTOR_CAP_PCT —
    the thing "diversification" in the ISTC 2026 Final Stage pitch (Risk
    Management, 10% of that score) gets pointed at. This cap is now
    enforced at shortlist-construction time (portfolio/daily_brief.py's
    _select_with_sector_cap()), not just reported here after the fact —
    "Over Cap?" should read "No" for every row unless daily_brief.json
    predates that fix. Still a placeholder value, like level_calculator's
    proximity_pct — portfolio/portfolio_optimizer.py doesn't exist yet to
    derive a real, ablated cap from; imported from daily_brief so the
    enforcement and this display can never drift apart.
    """
    by_sector: dict[str, float] = {}
    for e in shortlist:
        sector = e.get("sector") or "Unknown"
        by_sector[sector] = by_sector.get(sector, 0.0) + e.get("position_pct", 0.0)

    if not by_sector:
        return

    st.markdown('<h3 class="k100-subsection">Sector Concentration</h3>', unsafe_allow_html=True)
    st.caption(f"Current shortlist exposure per sector vs. a {daily_brief.SECTOR_CAP_PCT:.0f}% "
               f"placeholder cap — enforced at construction time, not just reported.")

    table = pd.DataFrame(
        [{"Sector": s, "Exposure": pct, "Over Cap?": "Yes" if pct > daily_brief.SECTOR_CAP_PCT else "No"}
         for s, pct in sorted(by_sector.items(), key=lambda kv: -kv[1])]
    )

    def _color_over(val):
        return f"color: {MUTED_RED}; font-weight: 600" if val == "Yes" else f"color: {TEAL}"

    styler = table.style.map(_color_over, subset=["Over Cap?"])
    styler = styler.format({"Exposure": "{:.1f}%".format})
    st.dataframe(styler, use_container_width=True, hide_index=True)


def _pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:+.2f}%"


def main() -> None:
    inject_css()
    render_header()
    render_status_strip()
    render_countdown_banner()

    tickers = load_universe_list()

    tab_universe, tab_detail, tab_health, tab_rankings = st.tabs(
        ["Universe", "Stock Detail", "Data Health", "Rankings"]
    )
    with tab_universe:
        render_universe_tab(tickers)
    with tab_detail:
        render_stock_detail_tab(tickers)
    with tab_health:
        render_data_health_tab(tickers)
    with tab_rankings:
        render_rankings_tab(tickers)


if __name__ == "__main__":
    main()
