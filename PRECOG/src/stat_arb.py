"""
Statistical arbitrage (pairs trading) overlay for the PRECOG quant task (Part 4).

Pairs are *selected* using only the formation period (TRAIN + VALIDATION,
2016-2021): a correlation pre-screen followed by an Engle-Granger
cointegration test on log prices, with a fixed hedge ratio estimated by OLS
on the same formation-period data. The selected pairs and hedge ratios are
then frozen and the resulting mean-reversion strategy is backtested entirely
out-of-sample on the test period (2022-2026) -- no pair-selection statistic
or hedge ratio is ever recomputed using test-period data.

The mean-reversion signal itself (a rolling z-score of the spread) uses a
trailing window computed from the full price history, so the first OOS
dates have a populated window from late-formation prices -- this is still
fully causal (no future information), it simply means the strategy "has been
running" continuously rather than restarting cold on day 1 of the test
period.
"""

import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from src.config import TRANSACTION_COST_BPS


def find_candidate_pairs(
    formation_prices: pd.DataFrame,
    corr_threshold: float = 0.8,
    pvalue_threshold: float = 0.01,
) -> tuple[pd.DataFrame, int]:
    """Screen all ticker pairs for cointegration on the formation period.

    Parameters
    ----------
    formation_prices : wide DataFrame, index=date, columns=ticker, values=close,
                        restricted to the formation period only.
    corr_threshold : minimum |correlation| of log prices for a pair to be
                      passed on to the (expensive) cointegration test.
    pvalue_threshold : Engle-Granger p-value cutoff for inclusion.

    Returns
    -------
    (pairs_df, n_tests) where `pairs_df` has columns
    [ticker_a, ticker_b, correlation, coint_pvalue, hedge_ratio] for *every*
    pair passing `pvalue_threshold`, sorted by `coint_pvalue` ascending, and
    `n_tests` is the number of cointegration tests actually run (i.e. pairs
    passing the correlation pre-screen) -- used to report a
    Bonferroni-corrected significance threshold and the multiple-testing
    context (`len(pairs_df)` vs. the number expected by chance at
    `pvalue_threshold`).
    """
    log_prices = np.log(formation_prices).dropna()
    corr = log_prices.corr()
    tickers = list(log_prices.columns)

    results = []
    n_tests = 0
    for a, b in itertools.combinations(tickers, 2):
        c = corr.loc[a, b]
        if abs(c) < corr_threshold:
            continue
        n_tests += 1
        _, pvalue, _ = coint(log_prices[a], log_prices[b])
        if pvalue < pvalue_threshold:
            hedge_ratio = float(np.polyfit(log_prices[b], log_prices[a], 1)[0])
            results.append({
                "ticker_a": a,
                "ticker_b": b,
                "correlation": c,
                "coint_pvalue": pvalue,
                "hedge_ratio": hedge_ratio,
            })

    pairs_df = pd.DataFrame(results).sort_values("coint_pvalue").reset_index(drop=True)
    return pairs_df, n_tests


def rolling_zscore(series: pd.Series, window: int = 60, min_periods: int = 20) -> pd.Series:
    """Causal rolling z-score: uses only `series[t-window+1 : t]` to score `t`."""
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std


def pair_signal(zscore: pd.Series, entry_z: float = 2.0, exit_z: float = 0.5, stop_z: float = 4.0) -> pd.Series:
    """State-machine position series from a spread z-score series.

    Standard mean-reversion rules:
      - flat -> long the spread  (+1) when zscore < -entry_z
      - flat -> short the spread (-1) when zscore > +entry_z
      - any open position -> flat (0) once |zscore| < exit_z (reverted) or
        |zscore| > stop_z (stop-loss: the spread kept diverging)

    `position[t]` depends only on `zscore[0..t]` -- the caller shifts this
    series forward by one day before multiplying by returns, so the position
    decided using information available on day t earns the return realized
    from t to t+1.
    """
    z = zscore.to_numpy()
    position = np.zeros(len(z))
    state = 0.0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            state = 0.0
        elif state == 0.0:
            if zi > entry_z:
                state = -1.0
            elif zi < -entry_z:
                state = 1.0
        elif abs(zi) < exit_z or abs(zi) > stop_z:
            state = 0.0
        position[i] = state
    return pd.Series(position, index=zscore.index)


def backtest_pair(
    price_pivot: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    hedge_ratio: float,
    test_start: pd.Timestamp,
    zscore_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    cost_bps: float = TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """Backtest one pair's mean-reversion strategy, reported for `date >= test_start`.

    Each leg is weighted so the two legs sum to unit gross exposure
    (`w_a + w_b = 1`), split in proportion to the hedge ratio -- a "1 unit"
    pair position. Returns DataFrame [date, zscore, position, gross_return,
    cost, net_return].
    """
    log_prices = np.log(price_pivot[[ticker_a, ticker_b]]).dropna()
    spread = log_prices[ticker_a] - hedge_ratio * log_prices[ticker_b]
    zscore = rolling_zscore(spread, window=zscore_window)
    position = pair_signal(zscore, entry_z, exit_z, stop_z)

    ret_a = price_pivot[ticker_a].pct_change()
    ret_b = price_pivot[ticker_b].pct_change()

    w_b = abs(hedge_ratio) / (1 + abs(hedge_ratio))
    w_a = 1 - w_b
    h_sign = np.sign(hedge_ratio)
    spread_return = w_a * ret_a - h_sign * w_b * ret_b

    position_shifted = position.shift(1).fillna(0.0)
    gross_return = position_shifted * spread_return

    turnover = position.diff().abs().fillna(0.0)
    turnover_shifted = turnover.shift(1).fillna(0.0)
    cost = turnover_shifted * (cost_bps / 10000.0)

    net_return = gross_return - cost

    out = pd.DataFrame({
        "date": zscore.index,
        "zscore": zscore.to_numpy(),
        "position": position_shifted.to_numpy(),
        "gross_return": gross_return.to_numpy(),
        "cost": cost.to_numpy(),
        "net_return": net_return.to_numpy(),
    })
    return out[out["date"] >= test_start].reset_index(drop=True)


def aggregate_pairs_portfolio(pair_results: dict) -> pd.DataFrame:
    """Equal-weight combination of each pair's `net_return` series.

    `pair_results` maps a pair name to the DataFrame returned by
    `backtest_pair` (all sharing the same date index). Returns
    DataFrame [date, return].
    """
    returns = pd.concat(
        {name: df.set_index("date")["net_return"] for name, df in pair_results.items()},
        axis=1,
    )
    portfolio_return = returns.mean(axis=1)
    return pd.DataFrame({"date": portfolio_return.index, "return": portfolio_return.to_numpy()}).reset_index(drop=True)
