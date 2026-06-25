"""
Feature engineering for the PRECOG quant task.

Every per-ticker indicator below is computed with rolling windows / EWMs
that only ever look backward from the current row (`.rolling(...)`,
`.shift(...)`, `.ewm(..., adjust=False)`), so each feature value at date t
uses only data observed on or before t. The cross-sectional features
(`add_cross_sectional_features`) compare tickers *within* the same date,
which is also causal: the cross-section at date t never uses data from any
other date.

No full-sample statistics (global mean/std/quantile) are computed anywhere
in this module -- that kind of computation is exactly the "seeing future
data" leakage this rebuild is designed to avoid.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Single-series indicators (operate on one ticker's chronologically sorted
# Series at a time)
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Each value depends only on prior price changes."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, and histogram (all causal EMAs)."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0):
    """Bollinger %B (position within the band) and band width."""
    sma = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    pct_b = (close - lower) / (upper - lower)
    width = (upper - lower) / sma
    return pct_b, width


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative signed volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Per-ticker feature table
# ---------------------------------------------------------------------------

def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the per-ticker causal feature table.

    Parameters
    ----------
    panel : long-format DataFrame with columns
            [date, ticker, open, high, low, close, volume], one row per
            (ticker, date), already cleaned (see cleaning.py).

    Returns
    -------
    DataFrame with [date, ticker, close, <~27 feature columns>], sorted by
    ticker then date. Rows near the start of each ticker's history will
    contain NaNs in the longer-lookback features (max lookback = 50 days)
    -- this is expected and handled at the modeling stage.
    """
    out = []
    for ticker, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)

        open_ = g["open"]
        high = g["high"]
        low = g["low"]
        close = g["close"]
        volume = g["volume"]
        prev_close = close.shift(1)

        feat = pd.DataFrame(index=g.index)
        feat["date"] = g["date"]
        feat["ticker"] = g["ticker"]
        feat["close"] = close  # kept for target construction downstream

        # --- Returns -----------------------------------------------------
        feat["ret_1d"] = close.pct_change(1)
        feat["ret_5d"] = close.pct_change(5)
        feat["ret_10d"] = close.pct_change(10)
        feat["ret_21d"] = close.pct_change(21)
        feat["log_ret_1d"] = np.log(close / prev_close)

        # --- Gap / intraday shape -----------------------------------------
        feat["overnight_gap"] = (open_ - prev_close) / prev_close
        feat["intraday_range"] = (high - low) / close
        feat["close_position"] = (close - low) / (high - low)

        # --- Momentum oscillators ------------------------------------------
        feat["rsi_14"] = _rsi(close, 14)

        macd_line, macd_signal, macd_hist = _macd(close)
        feat["macd"] = macd_line
        feat["macd_signal"] = macd_signal
        feat["macd_hist"] = macd_hist

        # --- Volatility ------------------------------------------------------
        feat["atr_14"] = _atr(high, low, close, 14) / close

        bb_pct, bb_width = _bollinger(close, 20, 2.0)
        feat["bb_pct"] = bb_pct
        feat["bb_width"] = bb_width

        ret_1d = feat["ret_1d"]
        feat["realized_vol_10"] = ret_1d.rolling(10, min_periods=10).std() * np.sqrt(252)
        feat["realized_vol_20"] = ret_1d.rolling(20, min_periods=20).std() * np.sqrt(252)

        # --- Volume ------------------------------------------------------------
        feat["volume_ratio_10"] = volume / volume.rolling(10, min_periods=10).mean()
        feat["volume_ratio_20"] = volume / volume.rolling(20, min_periods=20).mean()

        obv = _obv(close, volume)
        obv_mean = obv.rolling(20, min_periods=20).mean()
        obv_std = obv.rolling(20, min_periods=20).std()
        feat["obv_zscore_20"] = (obv - obv_mean) / obv_std

        # --- Trend / moving averages --------------------------------------------
        sma_10 = close.rolling(10, min_periods=10).mean()
        sma_20 = close.rolling(20, min_periods=20).mean()
        sma_50 = close.rolling(50, min_periods=50).mean()
        feat["sma_10_ratio"] = close / sma_10 - 1
        feat["sma_20_ratio"] = close / sma_20 - 1
        feat["sma_50_ratio"] = close / sma_50 - 1

        ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        feat["ema_12_26_ratio"] = ema_12 / ema_26 - 1

        # --- Price location z-scores --------------------------------------------
        roll_mean_20 = close.rolling(20, min_periods=20).mean()
        roll_std_20 = close.rolling(20, min_periods=20).std()
        feat["price_zscore_20"] = (close - roll_mean_20) / roll_std_20

        roll_mean_50 = close.rolling(50, min_periods=50).mean()
        roll_std_50 = close.rolling(50, min_periods=50).std()
        feat["price_zscore_50"] = (close - roll_mean_50) / roll_std_50

        out.append(feat)

    features = pd.concat(out, ignore_index=True)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.sort_values(["ticker", "date"]).reset_index(drop=True)
    return features


# ---------------------------------------------------------------------------
# Cross-sectional features
# ---------------------------------------------------------------------------

CROSS_SECTIONAL_BASE_COLS = ("ret_5d", "rsi_14", "realized_vol_20")


def add_cross_sectional_features(
    features: pd.DataFrame,
    cols: tuple[str, ...] = CROSS_SECTIONAL_BASE_COLS,
) -> pd.DataFrame:
    """Add cross-sectional rank and z-score features.

    For each date, each feature in `cols` is ranked / z-scored across the
    universe of tickers available *on that date only*. This is causal in
    time (date t's cross-section never references date t+k), but it does
    mean each row's value depends on the other tickers' values on the same
    date -- which is exactly the point for a cross-sectional ranking
    strategy.
    """
    features = features.copy()
    for col in cols:
        grp = features.groupby("date")[col]
        features[f"xs_rank_{col}"] = grp.rank(pct=True)
        features[f"xs_zscore_{col}"] = (features[col] - grp.transform("mean")) / grp.transform("std")
    return features


if __name__ == "__main__":
    from data_loader import load_raw_panel
    from cleaning import fix_ohlc_integrity, flag_return_outliers

    panel = load_raw_panel()
    panel, _ = fix_ohlc_integrity(panel)
    panel = flag_return_outliers(panel)

    feats = build_features(panel)
    feats = add_cross_sectional_features(feats)

    feature_cols = [c for c in feats.columns if c not in ("date", "ticker", "close")]
    print("n_features:", len(feature_cols))
    print(feature_cols)
    print(feats.shape)
    print("NaN rows per ticker (head):")
    print(feats.groupby("ticker").apply(lambda g: g[feature_cols].isna().any(axis=1).sum()).head())
