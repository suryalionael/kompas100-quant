"""Kompas100 Quant — Streamlit dashboard.

Session-1 scope (COMPETITION_PLAN.md §10, Days 1-2): universe overview,
per-stock technical detail, and a data-health panel over real OHLCV data.
Ranking/portfolio views are placeholders until those modules exist.
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

    st.dataframe(
        style_universe_table(display),
        use_container_width=True,
        height=560,
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn(label="Ticker"),
            "last_close": st.column_config.TextColumn(label="Last Close"),
            "pct_change": st.column_config.TextColumn(label="% Change"),
            "avg_volume_20d": st.column_config.TextColumn(label="Avg Vol (20d)"),
            "liquidity_tier": st.column_config.TextColumn(label="Liquidity Tier"),
        },
    )


def render_stock_detail_tab(tickers: list[str]) -> None:
    st.markdown('<h2 class="k100-section">Per-Stock Detail</h2>', unsafe_allow_html=True)

    selected = st.selectbox("Ticker", tickers, index=0)
    df = load_ticker_chart_data(selected)

    if df.empty:
        st.warning(f"No data available for {selected}.")
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

    lookback = st.slider("Lookback (trading days)", min_value=60, max_value=min(len(df), 750), value=min(250, len(df)), step=10)
    chart_df = df.tail(lookback)

    st.plotly_chart(price_detail_chart(chart_df, selected), use_container_width=True)


def render_data_health_tab(tickers: list[str]) -> None:
    st.markdown('<h2 class="k100-section">Data Health</h2>', unsafe_allow_html=True)

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

    st.markdown('<h3 class="k100-subsection">Fetch Status by Ticker</h3>', unsafe_allow_html=True)
    st.dataframe(health, use_container_width=True, height=380, hide_index=True)

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


def render_rankings_tab() -> None:
    st.markdown('<h2 class="k100-section">Rankings</h2>', unsafe_allow_html=True)
    st.info(
        "**Rankings — coming in a later phase.**\n\n"
        "This tab will show the cross-sectional ranking model's output once "
        "`ranking/ranking_model.py` and the horizon ablation (COMPETITION_PLAN.md §4) "
        "are built and validated against the backtest engine. No ranking data is "
        "fabricated here in the meantime."
    )


def main() -> None:
    inject_css()
    render_header()
    render_status_strip()

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
        render_rankings_tab()


if __name__ == "__main__":
    main()
