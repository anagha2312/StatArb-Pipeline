"""
Walk-forward training pipeline for the PRECOG quant task (Part 2).

Implements the no-look-ahead training scheme:

- **Target**: `TARGET_HORIZON`-day forward return, cross-sectionally
  demeaned (subtract that date's equal-weight universe mean). This is the
  natural target for a long/short ranking strategy -- "how much did this
  name outperform the average name over the next 5 trading days" -- and is
  what the Information Coefficient (IC) directly measures.
- **Folds**: expanding-window, retrained at the start of every calendar
  year from the year after `TRAIN_END` through the last year in the data.
  Each fold's training set excludes the last `EMBARGO_DAYS` trading days
  before the holdout year, so no training label's forward-return window
  overlaps the holdout period (fixes "seeing future data" at retrain
  boundaries).
- **Models**: Ridge regression + LightGBM regressor (both fit on the
  demeaned continuous target) and a Logistic Regression classifier (fit on
  "does this name beat the cross-sectional mean over the next
  `TARGET_HORIZON` days?"). All three -- including the feature scaler -- are
  refit from scratch on every fold's training data only.
- **Ensemble score**: for each holdout row, each model's output is
  cross-sectionally z-scored (per date) and the three z-scores are averaged
  to give the final ranking `score`.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

from src.config import EMBARGO_DAYS, TARGET_HORIZON, TRAIN_END


def build_targets(features: pd.DataFrame, horizon: int = TARGET_HORIZON) -> pd.DataFrame:
    """Add forward-return targets to the feature table.

    Adds:
      - `fwd_return`: raw `horizon`-day forward return, `pct_change(horizon)`
        computed per ticker and shifted back so it sits on the date the
        prediction would be made.
      - `target`: cross-sectionally demeaned forward return (regression
        target) -- "excess return vs the equal-weight universe over the
        next `horizon` days".
      - `target_binary`: 1 if `target` > 0 else 0 (classification target).

    The last `horizon` trading days of the panel have no forward return and
    so get NaN targets -- these rows are dropped before training but kept
    (with NaN target) in any holdout set passed through `predict_fold`.
    """
    df = features.copy()
    df["fwd_return"] = df.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(horizon).shift(-horizon)
    )
    df["target"] = df["fwd_return"] - df.groupby("date")["fwd_return"].transform("mean")
    df["target_binary"] = np.where(df["target"] > 0, 1.0, 0.0)
    df.loc[df["target"].isna(), "target_binary"] = np.nan
    return df


def generate_folds(dates: pd.Series, embargo_days: int = EMBARGO_DAYS, train_end: pd.Timestamp = TRAIN_END):
    """Yield `(year, train_dates, holdout_dates)` for each walk-forward fold.

    Folds are calendar years strictly after `train_end`'s year through the
    last year present in `dates`. Each fold's `train_dates` is every unique
    date <= the holdout year's start, with the last `embargo_days` dates
    dropped (embargo).
    """
    unique_dates = np.sort(pd.to_datetime(dates).unique())
    holdout_years = sorted({pd.Timestamp(d).year for d in unique_dates if pd.Timestamp(d) > train_end})

    for year in holdout_years:
        cutoff = pd.Timestamp(f"{year}-01-01")
        next_cutoff = pd.Timestamp(f"{year + 1}-01-01")

        train_dates = unique_dates[unique_dates < cutoff]
        if embargo_days > 0:
            train_dates = train_dates[:-embargo_days]

        holdout_dates = unique_dates[(unique_dates >= cutoff) & (unique_dates < next_cutoff)]
        if len(holdout_dates) == 0 or len(train_dates) == 0:
            continue

        yield year, train_dates, holdout_dates


def train_fold(train_df: pd.DataFrame, feature_cols: list, seed: int = 42) -> dict:
    """Fit the feature scaler and the three ensemble models on one fold's
    training data. Returns a dict of fitted objects."""
    X = train_df[feature_cols]
    y_reg = train_df["target"].to_numpy()
    y_clf = train_df["target_binary"].to_numpy()

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=train_df.index)

    ridge = Ridge(alpha=10.0, random_state=seed)
    ridge.fit(X_scaled, y_reg)

    lgbm = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        importance_type="gain",
        verbose=-1,
    )
    lgbm.fit(X_scaled, y_reg)

    logit = LogisticRegression(max_iter=1000, C=0.1, random_state=seed)
    logit.fit(X_scaled, y_clf)

    return {"scaler": scaler, "ridge": ridge, "lgbm": lgbm, "logit": logit}


def predict_fold(models: dict, holdout_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Score a fold's holdout rows with each model and combine into an
    ensemble `score` (mean of per-date cross-sectional z-scores)."""
    X = holdout_df[feature_cols]
    X_scaled = pd.DataFrame(models["scaler"].transform(X), columns=feature_cols, index=holdout_df.index)

    out = holdout_df.copy()
    out["pred_ridge"] = models["ridge"].predict(X_scaled)
    out["pred_lgbm"] = models["lgbm"].predict(X_scaled)
    out["pred_logit"] = models["logit"].predict_proba(X_scaled)[:, 1]

    z_cols = []
    for col in ("pred_ridge", "pred_lgbm", "pred_logit"):
        grp = out.groupby("date")[col]
        z_col = f"z_{col}"
        out[z_col] = (out[col] - grp.transform("mean")) / grp.transform("std")
        z_cols.append(z_col)

    out["score"] = out[z_cols].mean(axis=1)
    return out


def run_walkforward(df: pd.DataFrame, feature_cols: list, embargo_days: int = EMBARGO_DAYS, seed: int = 42):
    """Run the full walk-forward training / prediction loop.

    Parameters
    ----------
    df : feature table with `build_targets` already applied (must contain
         `date`, `ticker`, `feature_cols`, `target`, `target_binary`).
    feature_cols : list of feature column names to use as model inputs.

    Returns
    -------
    oos_df : concatenation of every fold's holdout predictions, with
             `score`, the per-model predictions/z-scores, and `fold_year`.
    feature_importance : DataFrame of per-fold LightGBM feature importances.
    """
    oos_parts = []
    importance_parts = []

    for year, train_dates, holdout_dates in generate_folds(df["date"], embargo_days):
        train_df = df[df["date"].isin(train_dates)].dropna(subset=feature_cols + ["target", "target_binary"])
        holdout_df = df[df["date"].isin(holdout_dates)].dropna(subset=feature_cols)

        models = train_fold(train_df, feature_cols, seed=seed)
        preds = predict_fold(models, holdout_df, feature_cols)
        preds["fold_year"] = year
        oos_parts.append(preds)

        importance_parts.append(pd.DataFrame({
            "fold_year": year,
            "feature": feature_cols,
            "importance": models["lgbm"].feature_importances_,
        }))

    oos_df = pd.concat(oos_parts, ignore_index=True)
    feature_importance = pd.concat(importance_parts, ignore_index=True)
    return oos_df, feature_importance
