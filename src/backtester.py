"""
Long/short decile portfolio backtester for the PRECOG quant task (Part 3).

Given the daily ensemble `score` produced by `src.walkforward` for every
(date, ticker) in the test period, and a price panel covering the same
dates:

- Every `rebalance_freq` trading days, rank all tickers by `score` on that
  date and set target weights: `+1/n_long` for the top `n_long` names,
  `-1/n_short` for the bottom `n_short` names, 0 otherwise (gross exposure
  200%, net exposure 0% -- a dollar-neutral long/short decile portfolio).
- Positions are held fixed between rebalances.
- A weight set using the score observed *on* date d is applied to the
  return realized from d to the next trading day -- it cannot "use" that
  return to decide the weight, so there is no look-ahead in the simulation
  mechanics.
- At each rebalance, a transaction cost of `cost_bps` basis points is
  applied to the turnover (sum of absolute weight changes).
"""

import numpy as np
import pandas as pd

from src.config import N_LONG, N_SHORT, REBALANCE_FREQ, TRANSACTION_COST_BPS


def _target_weights(scores_on_date: pd.Series, n_long: int, n_short: int) -> pd.Series:
    """Equal-weight long the top `n_long` / short the bottom `n_short` names by score."""
    ranked = scores_on_date.rank(ascending=False, method="first")
    n = len(scores_on_date)
    weights = pd.Series(0.0, index=scores_on_date.index)
    weights[ranked <= n_long] = 1.0 / n_long
    weights[ranked > n - n_short] = -1.0 / n_short
    return weights


class Backtester:
    def __init__(
        self,
        n_long: int = N_LONG,
        n_short: int = N_SHORT,
        rebalance_freq: int = REBALANCE_FREQ,
        cost_bps: float = TRANSACTION_COST_BPS,
    ):
        self.n_long = n_long
        self.n_short = n_short
        self.rebalance_freq = rebalance_freq
        self.cost_bps = cost_bps

    def run(self, scores: pd.DataFrame, prices: pd.DataFrame, apply_costs: bool = True) -> pd.DataFrame:
        """Run the backtest.

        Parameters
        ----------
        scores : DataFrame with columns [date, ticker, score], one row per
                 (date, ticker) for the backtest period.
        prices : DataFrame with columns [date, ticker, close] covering at
                 least the same dates as `scores`.
        apply_costs : if False, transaction costs are set to zero (used for
                 the with/without cost comparison).

        Returns
        -------
        DataFrame indexed by row with columns: date, gross_return, cost,
        net_return, turnover, n_long, n_short. The first score date is
        dropped (no prior position exists yet to earn a return).
        """
        score_pivot = scores.pivot(index="date", columns="ticker", values="score").sort_index()
        tickers = score_pivot.columns
        price_pivot = (
            prices.pivot(index="date", columns="ticker", values="close")
            .reindex(columns=tickers)
            .sort_index()
        )
        ret_pivot = price_pivot.pct_change()

        score_dates = score_pivot.index

        current_weights = pd.Series(0.0, index=tickers)
        weight_rows = []
        turnover_rows = []

        for i, date in enumerate(score_dates):
            if i % self.rebalance_freq == 0:
                today_scores = score_pivot.loc[date].dropna()
                new_weights = _target_weights(today_scores, self.n_long, self.n_short)
                new_weights = new_weights.reindex(tickers).fillna(0.0)
                turnover = (new_weights - current_weights).abs().sum()
                current_weights = new_weights
            else:
                turnover = 0.0
            weight_rows.append(current_weights.copy())
            turnover_rows.append(turnover)

        weights_df = pd.DataFrame(weight_rows, index=score_dates)
        turnover_series = pd.Series(turnover_rows, index=score_dates)

        # A weight set using the score on date d is held for the return
        # realized from d to d+1 -- shift forward by one row to align with
        # ret_pivot (whose row at date d+1 is the return ending on d+1).
        weights_shifted = weights_df.shift(1).fillna(0.0)
        turnover_shifted = turnover_series.shift(1).fillna(0.0)

        ret_aligned = ret_pivot.reindex(score_dates)
        gross_return = (weights_shifted * ret_aligned).sum(axis=1)

        if apply_costs:
            cost = turnover_shifted * (self.cost_bps / 10000.0)
        else:
            cost = pd.Series(0.0, index=score_dates)

        net_return = gross_return - cost

        daily = pd.DataFrame({
            "date": score_dates,
            "gross_return": gross_return.to_numpy(),
            "cost": cost.to_numpy(),
            "net_return": net_return.to_numpy(),
            "turnover": turnover_shifted.to_numpy(),
            "n_long": (weights_df > 0).sum(axis=1).to_numpy(),
            "n_short": (weights_df < 0).sum(axis=1).to_numpy(),
        })
        # First row has no prior position (zero return by construction) -- drop it.
        return daily.iloc[1:].reset_index(drop=True)


def equal_weight_benchmark(prices: pd.DataFrame, tickers=None, start_date=None) -> pd.DataFrame:
    """Buy-and-hold equal-weight benchmark.

    Each ticker is allocated an equal dollar weight at `start_date`
    (default: the first date in `prices`) and held without rebalancing, so
    weights drift naturally with relative price moves -- a standard passive
    benchmark.
    """
    price_pivot = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    if tickers is not None:
        price_pivot = price_pivot[list(tickers)]
    if start_date is not None:
        price_pivot = price_pivot[price_pivot.index >= start_date]

    normalized = price_pivot / price_pivot.iloc[0]
    portfolio_value = normalized.mean(axis=1)
    daily_return = portfolio_value.pct_change().fillna(0.0)

    return pd.DataFrame({
        "date": price_pivot.index,
        "return": daily_return.to_numpy(),
        "portfolio_value": portfolio_value.to_numpy(),
    })


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown of the cumulative-return series implied by `returns`."""
    cumulative = (1 + returns.fillna(0.0)).cumprod()
    running_max = cumulative.cummax()
    return cumulative / running_max - 1


def compute_performance_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Standard performance metrics for a series of daily returns."""
    returns = returns.dropna()
    cumulative = (1 + returns).cumprod()

    total_return = cumulative.iloc[-1] - 1
    n_years = len(returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    std = returns.std(ddof=1)
    annualized_vol = std * np.sqrt(periods_per_year)
    sharpe = (returns.mean() / std) * np.sqrt(periods_per_year) if std > 0 else np.nan

    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1)
    sortino = (
        (returns.mean() / downside_std) * np.sqrt(periods_per_year)
        if len(downside) > 1 and downside_std > 0
        else np.nan
    )

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    win_rate = (returns > 0).mean()
    gross_profit = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_drawdown,
        "avg_drawdown": avg_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "n_days": int(len(returns)),
    }
