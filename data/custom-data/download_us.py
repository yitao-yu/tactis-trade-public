"""
Download US stock data using yfinance.

Usage:  python download_us.py
        python download_us.py --tickers AAPL MSFT GOOGL
        python download_us.py --tickers-csv nasdaq100.csv
"""

import argparse
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

import config


def _next_day(date_str: str) -> str:
    """ISO date string + 1 day (used for incremental --update resumes)."""
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


def update_ticker(ticker: str, end: str, out_dir: Path) -> None:
    """
    Append rows after a ticker's last stored date to its per-ticker CSV.

    Missing file → full download from ``config.START_DATE``.  The CSV is
    deduplicated by date and kept in the original column order.
    """
    path = out_dir / f"{ticker}.csv"
    last = None
    df_old = None
    if path.exists():
        df_old = pd.read_csv(path)
        if df_old.empty:
            path.unlink()
            df_old = None
        else:
            last = pd.to_datetime(df_old["date"]).max().strftime("%Y-%m-%d")

    if last is not None and last >= end:
        print(f"  [{ticker}] already up to date ({last}).")
        return

    start = _next_day(last) if last else config.START_DATE
    try:
        df_new = download_one(ticker, start, end)
    except Exception as e:
        print(f"  [{ticker}] update failed: {e}")
        return
    if df_new.empty:
        print(f"  [{ticker}] no new data ({start} → {end}).")
        return

    if last is None:
        merged = df_new
    else:
        assert df_old is not None  # guaranteed when `last` is set
        merged = pd.concat([df_old, df_new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date").reset_index(drop=True)
        for col in df_old.columns:
            if col not in merged.columns:
                merged[col] = None
        merged = merged[df_old.columns]
    merged.to_csv(path, index=False)
    print(f"  [{ticker}] updated to {merged['date'].iloc[-1]} ({len(merged)} rows)")


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV for a single ticker."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df.insert(0, "ticker", ticker)
    df.index.name = "date"
    df = df.reset_index()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]


def fetch_fundamentals(ticker: str) -> dict:
    """Fetch market cap, sector, industry for a single ticker."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception:
        return {"market_cap": None, "sector": None, "industry": None}


def download_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV + optionally fundamentals for one ticker."""
    print(f"  [{ticker}] Downloading...")
    df = fetch_ohlcv(ticker, start, end)
    if df.empty:
        print(f"  [{ticker}] No data (may be delisted or invalid ticker). Skipping.")
        return df

    if config.FETCH_FUNDAMENTALS:
        print(f"  [{ticker}] Fetching fundamentals...")
        fund = fetch_fundamentals(ticker)
        for col, val in fund.items():
            df[col] = val

    print(f"  [{ticker}] {len(df)} rows from {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download US stock data via yfinance")
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="Tickers to download (space-separated)."
    )
    parser.add_argument(
        "--tickers-csv", default=None,
        help="Path to a CSV file with tickers (auto-detects 'ticker'/'code'/'symbol' column)."
    )
    parser.add_argument(
        "--start", default=config.START_DATE,
        help=f"Start date (default: {config.START_DATE})"
    )
    parser.add_argument(
        "--end", default=config.END_DATE,
        help=f"End date (default: {config.END_DATE})"
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to save per-ticker CSVs. Default: config.RAW_US"
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Incremental mode: append only rows after each ticker's last stored date."
    )
    parser.add_argument(
        "--output", default=None,
        help="Single output CSV path. Default: one CSV per ticker in raw/us/"
    )
    args = parser.parse_args()

    tickers = config.US_TICKERS
    if args.tickers:
        tickers = args.tickers
    elif args.tickers_csv:
        tickers = config.load_tickers_from_csv(args.tickers_csv)
    if not tickers:
        print("No tickers specified. Populate config.US_TICKERS, pass --tickers, or use --tickers-csv.")
        sys.exit(1)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.output:
        out_dir = Path(args.output).parent
    else:
        out_dir = config.RAW_US
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.update:
        if args.output:
            print("--update writes per-ticker CSVs; --output is not supported together.")
            sys.exit(1)
        print(f"Update mode: appending rows up to {args.end} in {out_dir}")
        for i, ticker in enumerate(tickers):
            update_ticker(ticker, args.end, out_dir)
            if i < len(tickers) - 1:
                time.sleep(config.SLEEP_BETWEEN_TICKERS)
        print("\nDone.")
        return

    all_dfs = []
    for i, ticker in enumerate(tickers):
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                df = download_one(ticker, args.start, args.end)
                break
            except Exception as e:
                print(f"  [{ticker}] Attempt {attempt} failed: {e}")
                if attempt == config.MAX_RETRIES:
                    print(f"  [{ticker}] All retries exhausted. Skipping.")
                    df = pd.DataFrame()
                else:
                    time.sleep(5)

        if not df.empty:
            if args.output:
                all_dfs.append(df)
            else:
                df.to_csv(out_dir / f"{ticker}.csv", index=False)
        if i < len(tickers) - 1:
            time.sleep(config.SLEEP_BETWEEN_TICKERS)

    if args.output and all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nSaved {len(combined)} rows to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
