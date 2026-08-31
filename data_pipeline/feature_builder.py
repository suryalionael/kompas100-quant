"""Build technical indicator features dari OHLCV DataFrame.

Semua fitur dihitung inline di sini — tidak ada sub-modul terpisah.
Tambahkan indikator baru ke `build_features()` dan daftarkan di `FEATURE_COLS`.

Dependencies: ta (pip install ta)

TradingView-equivalent indicators (dihitung di Python):
    supertrend_bullish  — Supertrend direction (period=10, mult=3.0)
    stoch_rsi_k / _d   — Stochastic RSI (14, 14, 3, 3)
    adx / adx_pos / _neg — ADX + DMI lines (period=14)
    squeeze_on          — Bollinger Band inside Keltner Channel
    vwap_20d            — Rolling 20-day VWAP approximation
    price_vs_vwap       — % di atas/bawah VWAP
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import ta
from loguru import logger

FEATURE_COLS = [
    "date", "ticker",
    # Trend
    "ma5", "ma20", "ma50", "ma200",
    "ma_full_alignment", "ma_partial_alignment",
    "slope_ma20", "golden_cross", "price_vs_ma200",
    # Momentum
    "rsi14", "macd", "macd_signal", "macd_histogram",
    "roc3", "roc5", "roc20",
    # Refined momentum (added 2026-08-31, kompas100-quant §4 feature-widening
    # pass — see ranking/ranking_model.py's RANKING_FEATURES for why these
    # exist as explicit columns rather than left for the model to infer)
    "sharpe_mom_20d", "mom_vol_confirmed_20d",
    # Breakout
    "high_52w", "pct_from_52w_high",
    "atr14", "atr_breakout",
    # Volume
    "vol_ratio_20d", "vol_spike", "obv_trend",
    # Volatility
    "atr_pct", "bb_width", "hist_vol_20d",
    # TradingView indicators
    "supertrend_bullish",
    "stoch_rsi_k", "stoch_rsi_d",
    "adx", "adx_pos", "adx_neg",
    "squeeze_on",
    "squeeze_release",      # True only on first day squeeze is OFF after being ON
    "vwap_20d", "price_vs_vwap",
    # IHSG relative strength (added for kompas100-quant — not in the original)
    "rel_strength_5d", "rel_strength_20d",
    # Raw (untuk scoring + ML)
    "close", "volume",
]


def build_features(df: pd.DataFrame, ihsg: pd.DataFrame | None = None) -> pd.DataFrame:
    """Hitung semua fitur teknikal untuk satu ticker DataFrame.

    Input: DataFrame dengan kolom OHLCV (minimal: date, open, high, low, close, volume).
    ihsg  : optional IHSG (^JKSE) OHLCV DataFrame — enables rel_strength_5d/20d.
    Output: DataFrame dengan kolom di FEATURE_COLS (kolom yang tidak bisa dihitung di-skip).
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df = _add_trend(df)
    df = _add_momentum(df)
    df = _add_breakout(df)
    df = _add_volume(df)
    df = _add_volatility(df)
    df = _add_tv_indicators(df)
    df = _add_refined_momentum(df)
    if ihsg is not None:
        df = _add_relative_strength(df, ihsg)

    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available]


def build_features_batch(data: dict[str, pd.DataFrame], ihsg: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build features untuk semua ticker dan gabungkan ke satu DataFrame."""
    frames = []
    for ticker, df in data.items():
        try:
            features = build_features(df, ihsg=ihsg)
            frames.append(features)
            logger.info(f"{ticker}: features computed ({len(features)} rows)")
        except Exception as e:
            logger.error(f"{ticker}: feature computation failed — {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_sector_relative_strength(features_df: pd.DataFrame, sector_by_ticker: dict[str, str]) -> pd.DataFrame:
    """Stock's 20-day return minus the average 20-day return of every
    OTHER ticker in the same GICS-style sector, on the same date — added
    2026-08-31 as a feature-widening candidate (COMPETITION_PLAN.md §4):
    rel_strength_20d (vs IHSG) only tells you if a stock beat the whole
    market; this tells you if it beat its own industry peers, which is a
    materially different comparison group (e.g. every commodity name can
    outperform IHSG together in a commodity rally while still ranking
    randomly against each other — sector-relative strength is what would
    separate them).

    sector_by_ticker is a current-day snapshot (data/published/
    universe_snapshot_latest.parquet's sector column) applied across the
    full historical window — a deliberate approximation, since sector
    classification changes far less often than daily prices and we have
    no historical sector-membership time series. Requires >= 2 tickers
    with a known sector on a given date to compute a peer average;
    unmapped tickers or thin sectors get NaN, not a fabricated 0.
    """
    if features_df.empty or not sector_by_ticker:
        features_df["sector_rel_strength_20d"] = np.nan
        return features_df

    df = features_df.copy()
    df["_sector"] = df["ticker"].map(sector_by_ticker)

    def _peer_relative(group: pd.DataFrame) -> pd.Series:
        n = group["roc20"].notna().sum()
        if n < 2:
            return pd.Series(np.nan, index=group.index)
        total = group["roc20"].sum(skipna=True)
        peer_mean = (total - group["roc20"]) / (n - 1)
        return group["roc20"] - peer_mean

    df["sector_rel_strength_20d"] = (
        df.groupby(["date", "_sector"], group_keys=False).apply(_peer_relative)
    )
    return df.drop(columns=["_sector"])


def save_features(df: pd.DataFrame, features_dir: Path, scan_date: str | None = None) -> None:
    label = scan_date or date.today().strftime("%Y-%m-%d")
    features_dir.mkdir(parents=True, exist_ok=True)
    path = features_dir / f"{label}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Feature store saved → {path}")


def load_features(features_dir: Path, scan_date: str) -> pd.DataFrame:
    path = features_dir / f"{scan_date}.parquet"
    if not path.exists():
        logger.warning(f"Feature file not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Internal builders — original indicators
# ---------------------------------------------------------------------------

def _add_trend(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ma5"] = c.rolling(5).mean()
    df["ma20"] = c.rolling(20).mean()
    df["ma50"] = c.rolling(50).mean()
    df["ma200"] = c.rolling(200).mean()

    has_all = df["ma20"].notna() & df["ma50"].notna() & df["ma200"].notna()
    df["ma_full_alignment"] = has_all & (df["ma20"] > df["ma50"]) & (df["ma50"] > df["ma200"])
    df["ma_partial_alignment"] = has_all & (df["ma20"] > df["ma50"])

    df["slope_ma20"] = df["ma20"] - df["ma20"].shift(5)

    ma50_prev = df["ma50"].shift(1)
    ma200_prev = df["ma200"].shift(1)
    df["golden_cross"] = (
        (df["ma50"] > df["ma200"]) & (ma50_prev <= ma200_prev)
    ).fillna(False)

    df["price_vs_ma200"] = ((c - df["ma200"]) / df["ma200"].replace(0, np.nan)) * 100
    return df


def _add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    macd_ind = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_histogram"] = macd_ind.macd_diff()

    df["roc3"] = df["close"].pct_change(3) * 100
    df["roc5"] = df["close"].pct_change(5) * 100
    df["roc20"] = df["close"].pct_change(20) * 100
    return df


def _add_breakout(df: pd.DataFrame) -> pd.DataFrame:
    df["high_52w"] = df["high"].rolling(252, min_periods=50).max()
    df["pct_from_52w_high"] = (
        (df["close"] - df["high_52w"]) / df["high_52w"].replace(0, np.nan) * 100
    )

    atr_ind = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr14"] = atr_ind.average_true_range()

    df["atr_breakout"] = df["close"] > (df["close"].shift(1) + 1.5 * df["atr14"].shift(1))
    df["atr_breakout"] = df["atr_breakout"].fillna(False)
    return df


def _add_volume(df: pd.DataFrame) -> pd.DataFrame:
    vol = df["volume"]
    vol_ma20 = vol.rolling(20).mean()
    df["vol_ratio_20d"] = vol / vol_ma20.replace(0, np.nan)
    df["vol_spike"] = df["vol_ratio_20d"] > 2.5

    obv = ta.volume.OnBalanceVolumeIndicator(df["close"], vol).on_balance_volume()
    df["obv_trend"] = obv > obv.shift(10)
    df["obv_trend"] = df["obv_trend"].fillna(False)
    return df


def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    if "atr14" not in df.columns:
        df = _add_breakout(df)
    df["atr_pct"] = df["atr14"] / df["close"].replace(0, np.nan) * 100

    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan) * 100

    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["hist_vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252) * 100
    return df


# ---------------------------------------------------------------------------
# Refined momentum (added 2026-08-31 — naive momentum was the only thing
# beating the ranking model in the §4 ablation; these give the model
# something raw roc20 doesn't already capture, per COMPETITION_PLAN.md's
# "widen the feature set" directive. Run after _add_volume/_add_volatility
# since both need vol_ratio_20d/hist_vol_20d, which don't exist yet inside
# _add_momentum itself.
# ---------------------------------------------------------------------------

def _add_refined_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """sharpe_mom_20d: 20-day return per unit of 20-day volatility — a
    Sharpe-style measure of whether a move was an efficient trend or just
    noisy volatility, which raw roc20 can't distinguish on its own.

    mom_vol_confirmed_20d: roc20 x vol_ratio_20d, an explicit interaction
    term. A linear model (this project's Ridge regressor) only ever sees
    a weighted sum of individual features — it cannot learn "momentum
    matters more when volume confirms it" unless that interaction is
    itself a feature, so this is that, not a stylistic choice.
    """
    if "hist_vol_20d" in df.columns:
        df["sharpe_mom_20d"] = df["roc20"] / df["hist_vol_20d"].replace(0, np.nan)
    if "vol_ratio_20d" in df.columns:
        df["mom_vol_confirmed_20d"] = df["roc20"] * df["vol_ratio_20d"]
    return df


# ---------------------------------------------------------------------------
# IHSG relative strength (new — not in the original scanner)
# ---------------------------------------------------------------------------

def _add_relative_strength(df: pd.DataFrame, ihsg: pd.DataFrame) -> pd.DataFrame:
    """Stock's N-day return minus IHSG's N-day return over the same dates.

    Positive = outperforming the index, negative = underperforming — a
    cheap, look-ahead-safe signal of whether a move is stock-specific or
    just beta to the broader market.
    """
    idx = ihsg[["date", "close"]].rename(columns={"close": "ihsg_close"}).sort_values("date")
    idx["ihsg_roc5"] = idx["ihsg_close"].pct_change(5) * 100
    idx["ihsg_roc20"] = idx["ihsg_close"].pct_change(20) * 100

    merged = df.merge(idx[["date", "ihsg_roc5", "ihsg_roc20"]], on="date", how="left")
    merged["rel_strength_5d"] = merged["roc5"] - merged["ihsg_roc5"]
    merged["rel_strength_20d"] = merged["roc20"] - merged["ihsg_roc20"]
    return merged.drop(columns=["ihsg_roc5", "ihsg_roc20"])


# ---------------------------------------------------------------------------
# TradingView-equivalent indicators (baru)
# ---------------------------------------------------------------------------

def _add_tv_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Indikator populer TradingView, dihitung di Python dari OHLCV harian.

    Indikator yang ditambahkan:
    1. Supertrend (period=10, multiplier=3.0)
    2. Stochastic RSI (14, 14, 3, 3)
    3. ADX + DMI lines (period=14)
    4. Bollinger Band Squeeze (BB dalam KC = squeeze)
    5. Rolling VWAP 20d + % terhadap VWAP
    """
    # --- 1. Supertrend ---
    try:
        st_val, st_dir = _compute_supertrend(df, period=10, multiplier=3.0)
        df["supertrend_bullish"] = (st_dir == 1)
        # Isi NaN sebagai False (tidak cukup data)
        df["supertrend_bullish"] = df["supertrend_bullish"].fillna(False)
    except Exception as e:
        logger.debug(f"Supertrend gagal: {e}")
        df["supertrend_bullish"] = False

    # --- 2. Stochastic RSI ---
    try:
        stoch_rsi = ta.momentum.StochRSIIndicator(
            df["close"], window=14, smooth1=3, smooth2=3
        )
        df["stoch_rsi_k"] = stoch_rsi.stochrsi_k() * 100
        df["stoch_rsi_d"] = stoch_rsi.stochrsi_d() * 100
    except Exception as e:
        logger.debug(f"StochRSI gagal: {e}")
        df["stoch_rsi_k"] = np.nan
        df["stoch_rsi_d"] = np.nan

    # --- 3. ADX + DMI ---
    try:
        adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
        df["adx"] = adx_ind.adx()
        df["adx_pos"] = adx_ind.adx_pos()   # +DI
        df["adx_neg"] = adx_ind.adx_neg()   # -DI
    except Exception as e:
        logger.debug(f"ADX gagal: {e}")
        df["adx"] = np.nan
        df["adx_pos"] = np.nan
        df["adx_neg"] = np.nan

    # --- 4. Bollinger Band Squeeze ---
    # Squeeze terjadi saat BB berada di dalam Keltner Channel (volatilitas rendah,
    # biasanya mendahului pergerakan besar)
    try:
        bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        kc = ta.volatility.KeltnerChannel(df["high"], df["low"], df["close"],
                                          window=20, window_atr=10)
        df["squeeze_on"] = (
            (bb.bollinger_lband() > kc.keltner_channel_lband()) &
            (bb.bollinger_hband() < kc.keltner_channel_hband())
        ).fillna(False)
    except Exception as e:
        logger.debug(f"Squeeze gagal: {e}")
        df["squeeze_on"] = False

    # --- squeeze_release: True pada hari PERTAMA squeeze berakhir ---
    # PENTING: harus dihitung di sini (history lengkap), BUKAN di signal_engine
    # karena signal_engine hanya melihat baris terakhir (single-row DataFrame).
    # squeeze_on shift(1) di signal_engine selalu False → bug.
    try:
        sq = df["squeeze_on"].astype(bool)
        df["squeeze_release"] = (~sq) & sq.shift(1).fillna(False)
    except Exception as e:
        logger.debug(f"Squeeze release gagal: {e}")
        df["squeeze_release"] = False

    # --- 5. Rolling VWAP 20d ---
    # VWAP harian (intraday tidak tersedia); pakai typical price × volume rolling sum
    try:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol = df["volume"].rolling(20).sum().replace(0, np.nan)
        df["vwap_20d"] = (typical * df["volume"]).rolling(20).sum() / cum_vol
        df["price_vs_vwap"] = (
            (df["close"] - df["vwap_20d"]) / df["vwap_20d"].replace(0, np.nan) * 100
        )
    except Exception as e:
        logger.debug(f"VWAP gagal: {e}")
        df["vwap_20d"] = np.nan
        df["price_vs_vwap"] = np.nan

    return df


def _compute_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Hitung Supertrend (versi Python dari indikator TradingView).

    Returns:
        (supertrend_line, direction)
        direction: 1 = bullish (harga di atas supertrend), -1 = bearish
    """
    if "atr14" in df.columns:
        atr = df["atr14"]
    else:
        atr = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=period
        ).average_true_range()

    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = (hl2 + multiplier * atr).values
    basic_lower = (hl2 - multiplier * atr).values
    close = df["close"].values
    n = len(close)

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    st = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)

    for i in range(1, n):
        if np.isnan(basic_upper[i]) or np.isnan(basic_lower[i]):
            continue

        # Final upper: tidak naik, kecuali harga kemarin sudah tembus
        final_upper[i] = (
            basic_upper[i]
            if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]
            else final_upper[i - 1]
        )
        # Final lower: tidak turun, kecuali harga kemarin sudah tembus
        final_lower[i] = (
            basic_lower[i]
            if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]
            else final_lower[i - 1]
        )

        # Direction flip
        if direction[i - 1] == -1 and close[i] > final_upper[i]:
            direction[i] = 1
        elif direction[i - 1] == 1 and close[i] < final_lower[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        st[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    return (
        pd.Series(st, index=df.index),
        pd.Series(direction, index=df.index),
    )
