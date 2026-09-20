"""
Configuration for custom data download scripts.

Fill in the ticker lists and adjust date ranges before running.
Tickers can also be loaded from a CSV file at runtime via --tickers-csv.
"""

from pathlib import Path

import pandas as pd

# ---- Paths ----
ROOT = Path(__file__).resolve().parent
RAW_US = ROOT / "raw" / "us"
RAW_CN = ROOT / "raw" / "cn"
MERGED_OUTPUT = ROOT / "raw" / "all_stocks.csv"

# ---- Date Range ----
START_DATE = "2008-01-01"
END_DATE = "2026-09-30"  # extended past training (2008-2024) for the backtest window

# ---- US Stock Tickers ----
# Fill this list or use --tickers-csv to load from a file.
# Example: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", ...]
US_TICKERS: list[str] = []

# ---- China A-Share Tickers ----
# Fill this list or use --tickers-csv to load from a file.
# Baostock format: "sh.600000" (Shanghai) or "sz.000001" (Shenzhen).
# Example: ["sh.600000", "sz.000001", "sh.600519", ...]
CN_TICKERS: list[str] = []

# ---- Fundamentals ----
FETCH_FUNDAMENTALS = True

# ---- Download Settings ----
SLEEP_BETWEEN_TICKERS = 0.5
MAX_RETRIES = 3

def load_tickers_from_csv(path: str | Path, column: str | None = None) -> list[str]:
    """Load ticker list from a CSV file.

    If *column* is given, reads that column.
    Otherwise tries columns named 'ticker', 'code', 'symbol', then falls back to the first column.
    Each ticker is stripped of whitespace; empty entries are dropped.
    """
    df = pd.read_csv(str(path), dtype=str)
    if column:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in {path}. Available: {list(df.columns)}")
        return df[column].dropna().str.strip().tolist()

    for col in ["ticker", "code", "symbol"]:
        if col in df.columns:
            return df[col].dropna().str.strip().tolist()

    return df.iloc[:, 0].dropna().str.strip().tolist()
