"""Primary data fetch for the daily Batch job: yfinance (US) + baostock (CN).

Downloads recent daily closes for the training universe and writes the
long-format CSV (``date,ticker,close``) that infer.py consumes.  Fails
loudly (exit != 0) on stale/short data so Batch retries or the job aborts
before trading on bad data.

Fallback: app/data_fetch.py --via ibkr (IBKR historical bars, offline-safe).

Usage (inside the Batch container):
    python3 /opt/app/data_fetch.py --out /opt/data/all_stocks.csv \
        --tickers-csv /opt/data/universe.csv --days 40

Smoke (tiny universe):
    python3 app/data_fetch.py --out /tmp/smoke.csv --tickers AAPL MSFT
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PACER_DELAY = 1.0        # seconds between per-ticker downloads
MIN_DAYS_COVERAGE = 15   # a ticker must have >= this many rows to count


def fetch_yf(tickers: list[str], days: int) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    start = (datetime.utcnow() - timedelta(days=int(days * 1.6))).date().isoformat()
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        for attempt in (1, 2, 3):
            try:
                df = yf.download(t, start=start, progress=False, auto_adjust=True)
                if df is None or df.empty:
                    continue
                close = df["Close"]
                if isinstance(close, pd.DataFrame):  # MultiIndex [Price, Ticker]
                    close = close.iloc[:, 0]
                s = close.dropna()
                if len(s):
                    out[t] = s.reset_index()
                    break
            except Exception as e:  # noqa: BLE001 — retry any provider hiccup
                print(f"[data_fetch] {t} attempt {attempt} failed: {e}",
                      file=sys.stderr)
                time.sleep(2 * attempt)
        else:
            print(f"[data_fetch] WARN: {t} unavailable, skipping",
                  file=sys.stderr)
        time.sleep(PACER_DELAY)
    return out


def fetch_bs(tickers: list[str], days: int) -> dict[str, pd.DataFrame]:
    import baostock as bs

    start = (datetime.utcnow() - timedelta(days=int(days * 1.6))).date().isoformat()
    out: dict[str, pd.DataFrame] = {}
    lg = bs.login()
    if lg.error_code != "0":
        print(f"[data_fetch] baostock login failed: {lg.error_msg}",
              file=sys.stderr)
        return out
    try:
        for t in tickers:
            rs = bs.query_history_k_data_plus(
                t, "date,close", start_date=start,
                frequency="d", adjustflag="2")
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df = pd.DataFrame(rows, columns=["date", "close"])
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df = df.dropna()
                if len(df):
                    out[t] = df
            time.sleep(PACER_DELAY)
    finally:
        bs.logout()
    return out


def merge_and_write(frames: dict[str, pd.DataFrame], out: Path) -> tuple[int, str]:
    parts = []
    for t, df in frames.items():
        if len(df) < MIN_DAYS_COVERAGE:
            print(f"[data_fetch] WARN: {t} only {len(df)} rows (<{MIN_DAYS_COVERAGE})",
                  file=sys.stderr)
            continue
        d = df.copy()
        d.columns = ["date", "close"]
        d["ticker"] = t
        parts.append(d)
    if not parts:
        return 0, ""
    df = pd.concat(parts)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset=["date", "ticker"]).sort_values(
        ["ticker", "date"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return len(df), df["date"].max()


def read_tickers(path: str | None, inline: list[str] | None,
                 us_csv: str | None, cn_csv: str | None) -> tuple[list[str], list[str]]:
    def _col(path: str) -> list[str]:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for col in ("ticker", "code", "symbol"):
            if rows and col in rows[0]:
                return [r[col] for r in rows if r.get(col)]
        return []

    us = inline or (us_csv and _col(us_csv)) or []
    cn = (cn_csv and _col(cn_csv)) or []
    return us, cn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tickers", nargs="*", default=None, help="US tickers")
    ap.add_argument("--tickers-csv", default=None, help="US universe csv")
    ap.add_argument("--cn-tickers-csv", default=None, help="CN universe csv (sh./sz.)")
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--skip-cn", action="store_true")
    args = ap.parse_args()

    us, cn = read_tickers(args.tickers_csv, args.tickers,
                          args.tickers_csv, args.cn_tickers_csv)
    print(f"[data_fetch] US={len(us)} CN={0 if args.skip_cn else len(cn)} "
          f"days={args.days} -> {args.out}")

    frames: dict[str, pd.DataFrame] = {}
    if us:
        frames.update(fetch_yf(us, args.days))
    if cn and not args.skip_cn:
        frames.update(fetch_bs(cn, args.days))

    if not frames:
        print("[data_fetch] FATAL: no data fetched (primary AND fallback "
              "should be tried by caller before this point)", file=sys.stderr)
        return 3
    n, last = merge_and_write(frames, Path(args.out))
    if not n:
        print("[data_fetch] FATAL: all tickers unusable", file=sys.stderr)
        return 3
    print(f"[data_fetch] wrote {n} rows, last date {last}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
