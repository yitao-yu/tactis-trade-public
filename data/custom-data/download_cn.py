"""
Download Chinese A-share stock data using baostock.

Usage:  python download_cn.py
        python download_cn.py --tickers sh.600000 sz.000001
        python download_cn.py --tickers-csv hs300_components.csv
"""

import argparse
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import baostock as bs

import config


def _next_day(date_str: str) -> str:
    """ISO date string + 1 day (used for incremental --update resumes)."""
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


def update_ticker(ticker: str, end: str, out_dir: Path, fetch_fund: bool) -> None:
    """
    Append rows after a ticker's last stored date to its per-ticker CSV.

    Missing file → full download from ``config.START_DATE``.  The CSV is
    deduplicated by date and kept in the original column order.
    """
    path = out_dir / f"{ticker.replace('.', '_')}.csv"
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
        df_new = download_one(ticker, start, end, fetch_fund=fetch_fund)
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


def _login():
    lg = bs.login()
    if lg.error_code != "0":
        raise ConnectionError(f"Baostock login failed: {lg.error_msg}")
    print("  Baostock logged in.")


def _logout():
    bs.logout()


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV for a single A-share ticker."""
    fields = "date,open,high,low,close,volume"
    rs = bs.query_history_k_data_plus(
        ticker, fields,
        start_date=start, end_date=end,
        frequency="d", adjustflag="2"  # 2 = forward-adjusted
    )
    if rs.error_code != "0":
        raise RuntimeError(f"Query failed: {rs.error_msg}")

    records = []
    while rs.next():
        records.append(rs.get_row_data())

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records, columns=fields.split(","))
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.insert(0, "ticker", ticker)
    return df


def fetch_stock_basic(ticker: str) -> dict:
    """Fetch stock name and industry from baostock stock_basic."""
    code = ticker.split(".")[-1] if "." in ticker else ticker
    rs = bs.query_stock_basic(code=code)
    if rs.error_code != "0":
        return {"name": None, "industry": None}
    record = None
    while rs.next():
        record = rs.get_row_data()
        break
    if record:
        return {"name": record[1] if len(record) > 1 else None,
                "industry": record[-1] if len(record) > 2 else None}
    return {"name": None, "industry": None}


def download_one(ticker: str, start: str, end: str, fetch_fund: bool = True) -> pd.DataFrame:
    """Download OHLCV + optionally industry info for one ticker."""
    clean_ticker = ticker.replace(".", "_")
    print(f"  [{clean_ticker}] Downloading...")
    df = fetch_ohlcv(ticker, start, end)
    if df.empty:
        print(f"  [{clean_ticker}] No data. Skipping.")
        return df

    if fetch_fund and config.FETCH_FUNDAMENTALS:
        try:
            info = fetch_stock_basic(ticker)
            df["name"] = info["name"]
            df["industry"] = info["industry"]
        except Exception as e:
            print(f"  [{clean_ticker}] Fundamentals skipped: {e}")

    print(f"  [{clean_ticker}] {len(df)} rows from {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download Chinese A-share data via baostock")
    parser.add_argument(
        "--tickers", nargs="*", default=None,
        help="Tickers in baostock format (sh.600000 sz.000001)."
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
        help="Directory to save per-ticker CSVs. Default: config.RAW_CN"
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Incremental mode: append only rows after each ticker's last stored date."
    )
    parser.add_argument(
        "--output", default=None,
        help="Single output CSV path. Default: one CSV per ticker in raw/cn/"
    )
    parser.add_argument(
        "--no-fundamentals", action="store_true",
        help="Skip fetching stock name and industry info."
    )
    args = parser.parse_args()

    tickers = config.CN_TICKERS
    if args.tickers:
        tickers = args.tickers
    elif args.tickers_csv:
        tickers = config.load_tickers_from_csv(args.tickers_csv)
    if not tickers:
        print("No tickers specified. Populate config.CN_TICKERS, pass --tickers, or use --tickers-csv.")
        sys.exit(1)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.output:
        out_dir = Path(args.output).parent
    else:
        out_dir = config.RAW_CN
    out_dir.mkdir(parents=True, exist_ok=True)

    _login()
    try:
        if args.update:
            if args.output:
                print("--update writes per-ticker CSVs; --output is not supported together.")
                sys.exit(1)
            print(f"Update mode: appending rows up to {args.end} in {out_dir}")
            for i, ticker in enumerate(tickers):
                update_ticker(ticker, args.end, out_dir, fetch_fund=not args.no_fundamentals)
                if i < len(tickers) - 1:
                    time.sleep(config.SLEEP_BETWEEN_TICKERS)
            print("\nDone.")
            return

        all_dfs = []
        for i, ticker in enumerate(tickers):
            for attempt in range(1, config.MAX_RETRIES + 1):
                try:
                    df = download_one(ticker, args.start, args.end, fetch_fund=not args.no_fundamentals)
                    break
                except Exception as e:
                    print(f"  [{ticker}] Attempt {attempt} failed: {e}")
                    if attempt == config.MAX_RETRIES:
                        print(f"  [{ticker}] All retries exhausted. Skipping.")
                        df = pd.DataFrame()
                    else:
                        time.sleep(5)
                        _logout()
                        _login()

            if not df.empty:
                clean_name = ticker.replace(".", "_")
                if args.output:
                    all_dfs.append(df)
                else:
                    df.to_csv(out_dir / f"{clean_name}.csv", index=False)
            if i < len(tickers) - 1:
                time.sleep(config.SLEEP_BETWEEN_TICKERS)

        if args.output and all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(args.output, index=False)
            print(f"\nSaved {len(combined)} rows to {args.output}")
    finally:
        _logout()

    print("\nDone.")


if __name__ == "__main__":
    main()
