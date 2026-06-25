"""
Shared evaluation metrics: the Information Coefficient (IC) and friends.

IC = daily cross-sectional Spearman rank correlation between a score (a raw
feature in Part 1's EDA, or a model's ensemble output in Part 2) and the
realized forward return. This is the standard way to evaluate a noisy
cross-sectional ranking signal, and is used identically in Part 1 (EDA
sanity check, training period only) and Part 2 (walk-forward OOS
evaluation).
"""

import numpy as np
import pandas as pd


def cross_sectional_ic(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    date_col: str = "date",
) -> pd.Series:
    """Daily cross-sectional Spearman IC between `score_col` and `target_col`.

    Returns a Series of IC values indexed by date. Implemented via per-date
    rank standardization followed by a vectorized groupby-mean (no
    `.apply`), so it scales to many features/dates.
    """
    work = df[[date_col, score_col, target_col]].dropna()

    def _z_rank(s: pd.Series, by: pd.Series) -> pd.Series:
        r = s.groupby(by).rank()
        mean = r.groupby(by).transform("mean")
        std = r.groupby(by).transform("std")
        return (r - mean) / std

    z_score = _z_rank(work[score_col], work[date_col])
    z_target = _z_rank(work[target_col], work[date_col])

    ic_by_date = (z_score * z_target).groupby(work[date_col]).mean()
    return ic_by_date


def ic_summary(ic_by_date: pd.Series) -> dict:
    """Summary statistics for a series of daily ICs."""
    ic_by_date = ic_by_date.dropna()
    mean_ic = ic_by_date.mean()
    std_ic = ic_by_date.std()
    return {
        "mean_ic": mean_ic,
        "ic_std": std_ic,
        "ic_ir": mean_ic / std_ic if std_ic else np.nan,
        "hit_rate": (ic_by_date > 0).mean(),
        "n_days": int(len(ic_by_date)),
    }
