"""
Configuration for custom-data-test download scripts.

Same layout as config.py but with 2025-2026/08 dates and test-specific paths.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_US = ROOT / "raw" / "us"
RAW_CN = ROOT / "raw" / "cn"
MERGED_OUTPUT = ROOT / "raw" / "all_stocks.csv"

START_DATE = "2025-01-01"
END_DATE = "2026-08-10"

US_TICKERS: list[str] = []
CN_TICKERS: list[str] = []

FETCH_FUNDAMENTALS = True

SLEEP_BETWEEN_TICKERS = 0.5
MAX_RETRIES = 3


def load_tickers_from_csv(path: str | Path, column: str | None = None) -> list[str]:
    df = pd.read_csv(str(path), dtype=str)
    if column:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in {path}. Available: {list(df.columns)}")
        return df[column].dropna().str.strip().tolist()

    for col in ["ticker", "code", "symbol"]:
        if col in df.columns:
            return df[col].dropna().str.strip().tolist()

    return df.iloc[:, 0].dropna().str.strip().tolist()
