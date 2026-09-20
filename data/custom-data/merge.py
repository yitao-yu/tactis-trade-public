"""
Merge US and CN stock CSVs into a unified training dataset.

Reads all CSVs from custom-data/raw/us/ and custom-data/raw/cn/,
concatenates them, and writes a single combined CSV.

Usage:  python merge.py
        python merge.py --output ../processed/unified.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

import config

COLUMN_ORDER = [
    "date", "ticker", "open", "high", "low", "close", "volume",
    "market_cap", "sector", "industry", "name",
]


def load_directory(directory: Path) -> list[pd.DataFrame]:
    """Load all CSV files from a directory into a list of DataFrames."""
    if not directory.exists():
        return []
    dfs = []
    for csv_path in sorted(directory.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                dfs.append(df)
                print(f"  Loaded {csv_path.name}: {len(df)} rows")
        except Exception as e:
            print(f"  Skipping {csv_path.name}: {e}")
    return dfs


def main():
    parser = argparse.ArgumentParser(description="Merge US and CN CSVs into one dataset")
    parser.add_argument(
        "--output", default=str(config.MERGED_OUTPUT),
        help=f"Output CSV path (default: {config.MERGED_OUTPUT})"
    )
    parser.add_argument(
        "--us-dir", default=str(config.RAW_US),
        help=f"US data directory (default: {config.RAW_US})"
    )
    parser.add_argument(
        "--cn-dir", default=str(config.RAW_CN),
        help=f"China data directory (default: {config.RAW_CN})"
    )
    args = parser.parse_args()

    print("Loading US data...")
    us_dfs = load_directory(Path(args.us_dir))
    print("Loading CN data...")
    cn_dfs = load_directory(Path(args.cn_dir))

    all_dfs = us_dfs + cn_dfs
    if not all_dfs:
        print("No data found. Run download_us.py and download_cn.py first.")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)

    for col in COLUMN_ORDER:
        if col not in combined.columns:
            combined[col] = None

    present_cols = [c for c in COLUMN_ORDER if c in combined.columns]
    extra_cols = [c for c in combined.columns if c not in COLUMN_ORDER]
    combined = combined[present_cols + extra_cols]

    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    n_tickers = combined["ticker"].nunique()
    print(f"\nMerged: {len(combined)} rows, {n_tickers} tickers")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
