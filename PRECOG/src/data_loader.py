"""
Data loading utilities for the PRECOG quant task.

Loads the per-ticker OHLCV CSVs (downloaded from Kaggle) into a single
long-format panel: one row per (date, ticker), sorted by ticker then date.
This long format is the common input for all downstream cleaning, feature
engineering, and modeling steps.
"""

from pathlib import Path

import pandas as pd

# Repo root is two levels up from this file (PRECOG/src/data_loader.py -> PRECOG/)
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw" / "anonymized_data"


def load_raw_panel(raw_dir: Path | str = RAW_DATA_DIR) -> pd.DataFrame:
    """Load all Asset_*.csv files into one long-format DataFrame.

    Parameters
    ----------
    raw_dir : path to the directory containing Asset_001.csv ... Asset_100.csv

    Returns
    -------
    DataFrame with columns [date, ticker, open, high, low, close, volume],
    sorted by ticker then date, with a clean RangeIndex.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("Asset_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No Asset_*.csv files found in {raw_dir}. "
            "Run the Kaggle download step described in the README first."
        )

    frames = []
    for f in files:
        ticker = f.stem  # e.g. "Asset_001"
        df = pd.read_csv(f, parse_dates=["Date"])
        df = df.rename(columns=str.lower).rename(columns={"date": "date"})
        df["ticker"] = ticker
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[["date", "ticker", "open", "high", "low", "close", "volume"]]
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


if __name__ == "__main__":
    panel = load_raw_panel()
    print(panel.shape)
    print(panel["ticker"].nunique(), "tickers")
    print(panel["date"].min(), "to", panel["date"].max())
    print(panel.head())
    print(panel.isna().sum().sum(), "total NaNs")
