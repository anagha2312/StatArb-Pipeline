"""
Shared configuration for the PRECOG quant pipeline.

Defining the train/validation/test split dates and key strategy parameters
in one place ensures every notebook (Part 1-4) uses exactly the same
boundaries -- a single source of truth that prevents any notebook from
accidentally drifting onto a different (leaky) split.

Timeline (2016-01-25 -> 2026-01-16, 2511 trading days, 100 tickers):
  - Train      : 2016-01-25 -> 2019-12-31  (initial walk-forward training window)
  - Validation : 2020-01-01 -> 2021-12-31  (strategy / hyperparameter tuning)
  - Test       : 2022-01-01 -> 2026-01-16  (final OOS backtest, ~4 years)

Walk-forward retraining: models are refit on an expanding window at the
start of each calendar year from 2020 onward, so every prediction in
2020-2026 is made by a model that has never seen data from that year (or
later). An EMBARGO_DAYS gap is dropped immediately before each retrain
cutoff so that no training label's forward-return window crosses into the
holdout period.
"""

import pandas as pd

TRAIN_START = pd.Timestamp("2016-01-25")
TRAIN_END = pd.Timestamp("2019-12-31")

VAL_START = pd.Timestamp("2020-01-01")
VAL_END = pd.Timestamp("2021-12-31")

TEST_START = pd.Timestamp("2022-01-01")
TEST_END = pd.Timestamp("2026-01-16")

# Prediction target: forward return over this many trading days.
TARGET_HORIZON = 5

# Trading days dropped immediately before each walk-forward retrain cutoff,
# so that a training label's forward-return window never overlaps the
# out-of-sample period that follows it.
EMBARGO_DAYS = TARGET_HORIZON

# Portfolio rebalance frequency (trading days). Chosen in Part 2 via a small
# grid search over the VALIDATION period (2020-2021) only -- see "Strategy
# hyperparameter selection" there for the with-cost Sharpe comparison across
# rebalance frequencies and decile sizes that justifies this value.
REBALANCE_FREQ = 10

# Long/short decile portfolio sizes (out of a 100-ticker universe), also
# selected via the Part 2 validation-period grid search.
N_LONG = 10
N_SHORT = 10

# One-way transaction cost, in basis points, applied to traded turnover.
TRANSACTION_COST_BPS = 10
