"""
Data cleaning utilities for the PRECOG quant task.

Both functions here are deliberately *causal*: every computed value at row
t depends only on information available at or before t (and, for the OHLC
fix, only on the same row). Nothing here ever looks at the full-sample
distribution, so none of it can leak test-period information into the
train/validation periods.
"""

import numpy as np
import pandas as pd


def fix_ohlc_integrity(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ensure High = max(O,H,L,C) and Low = min(O,H,L,C) for every row.

    A handful of rows in the raw data have High/Low values that don't
    bound the Open/Close (almost certainly floating-point noise from the
    synthetic data generator). This recomputes High/Low from that same
    row's O/H/L/C values only -- no information from any other row or time
    period is used, so it cannot introduce look-ahead bias.
    """
    panel = panel.copy()
    cols = ["open", "high", "low", "close"]
    row_max = panel[cols].max(axis=1)
    row_min = panel[cols].min(axis=1)

    n_high_fixed = int((panel["high"] < row_max).sum())
    n_low_fixed = int((panel["low"] > row_min).sum())

    panel["high"] = row_max
    panel["low"] = row_min

    report = {"high_fixed": n_high_fixed, "low_fixed": n_low_fixed}
    return panel, report


def flag_return_outliers(
    panel: pd.DataFrame,
    window: int = 252,
    n_std: float = 8.0,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Flag daily close-to-close returns that are extreme relative to their
    own *trailing* history.

    For each ticker, the threshold at time t is the mean/std of returns over
    the `window` days strictly before t (via `.shift(1).rolling(...)`), so
    nothing at or after t is used. This is purely diagnostic -- it adds a
    boolean `return_outlier` column without altering any prices, so it is
    safe to run before any train/val/test split.
    """
    panel = panel.copy()
    ret_1d = panel.groupby("ticker")["close"].pct_change()

    roll_mean = (
        ret_1d.groupby(panel["ticker"])
        .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
    )
    roll_std = (
        ret_1d.groupby(panel["ticker"])
        .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).std())
    )

    is_outlier = ((ret_1d - roll_mean).abs() > n_std * roll_std)
    panel["return_outlier"] = is_outlier.fillna(False)
    return panel


if __name__ == "__main__":
    from data_loader import load_raw_panel

    panel = load_raw_panel()
    panel, report = fix_ohlc_integrity(panel)
    print("OHLC integrity fixes:", report)

    panel = flag_return_outliers(panel)
    print("Return outliers flagged:", int(panel["return_outlier"].sum()), "/", len(panel))
