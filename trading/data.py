"""
Date-aware price/return data for the backtest venue (Venue 0).

``BacktestData`` loads the merged ``all_stocks.csv`` long-format file (or any
equivalent long DataFrame with ``date, ticker, open, close`` columns), pivots it
into wide open/close matrices and exposes:

- a frozen universe (tickers present in the base training range),
- per-series z-score normalisation statistics pinned to the base range,
- day-indexed close-to-close returns and validity masks,
- per-market classification (``sh.``/``sz.`` prefixes → cn).

The universe freeze and the norm-stat freeze are the two hard constraints that
let the backtest timeline be extended past the training data without shifting
the model's input distribution or its embedding size.
"""

from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


def _is_cn(ticker: str) -> bool:
    """A-share tickers carry ``sh.`` / ``sz.`` exchange prefixes."""
    return ticker.startswith("sh.") or ticker.startswith("sz.")


class BacktestData:
    """
    Wide, date-indexed backtest price data.

    Parameters
    ----------
    open_px : pd.DataFrame
        Wide frame [date, ticker] of open prices.  Columns sorted alphabetically.
    close_px : pd.DataFrame
        Wide frame [date, ticker] of close prices.  Same index/columns as ``open_px``.
    base_end_date : str | pd.Timestamp | None
        Last date of the *base* (training) range.  The universe and the norm stats
        are pinned to this range; rows after it are backtest-only.
    train_frac : float
        Fraction of the base range used for per-series normalisation statistics.
    """

    def __init__(
        self,
        open_px: pd.DataFrame,
        close_px: pd.DataFrame,
        base_end_date: str | pd.Timestamp | None = None,
        train_frac: float = 0.8,
    ):
        if not open_px.index.equals(close_px.index):
            raise ValueError("open and close frames must share the same date index")
        if not open_px.columns.equals(close_px.columns):
            raise ValueError("open and close frames must share the same ticker columns")

        self.dates: np.ndarray = open_px.index.to_numpy()
        self.tickers: list[str] = list(open_px.columns)
        self._ticker_index: dict[str, int] = {t: i for i, t in enumerate(self.tickers)}
        self.base_end_date = pd.Timestamp(base_end_date) if base_end_date else self.dates[-1]

        # NaN preserved for missing prices (executor freezes such names).
        self.close: np.ndarray = close_px.to_numpy(dtype=np.float64)
        self.open: np.ndarray = open_px.to_numpy(dtype=np.float64)

        # Close-to-close arithmetic returns.  Row t prices close[t] against
        # close[t-1]; row 0 has no predecessor and is marked invalid.
        ret = np.full_like(self.close, 0.0)
        ret[1:] = (self.close[1:] - self.close[:-1]) / self.close[:-1]
        valid = np.isfinite(ret)
        ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)

        self.ret: np.ndarray = ret.astype(np.float32)
        self.valid: np.ndarray = valid

        # ---- Universe freeze: only tickers present in the base range ----
        base_rows = self.dates <= self.base_end_date
        if not base_rows.any():
            raise ValueError("base_end_date precedes all available data")
        in_base = np.isfinite(self.close[base_rows]).any(axis=0)
        dropped = int((~in_base).sum())
        if dropped:
            print(f"[BacktestData] dropping {dropped} ticker(s) with no base-range data "
                  f"(universe frozen to training names)")
            self._restrict(np.where(in_base)[0])

        # ---- Norm stats frozen to the first train_frac of the base range ----
        base_row_count = int(base_rows.sum())
        # Return row 0 has no predecessor (invalid); real base returns are rows 1..B-1.
        n_base = 1 + int((base_row_count - 1) * train_frac)
        self._compute_norm_stats(n_base)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _restrict(self, keep: np.ndarray) -> None:
        self.tickers = [self.tickers[i] for i in keep]
        self._ticker_index = {t: i for i, t in enumerate(self.tickers)}
        self.close = self.close[:, keep]
        self.open = self.open[:, keep]
        self.ret = self.ret[:, keep]
        self.valid = self.valid[:, keep]

    def _compute_norm_stats(self, n_base: int) -> None:
        stats: list[tuple[float, float]] = []
        for s in range(self.num_series):
            vals = self.ret[:n_base, s][self.valid[:n_base, s]]
            if vals.size < 2:
                stats.append((0.0, 1.0))
                continue
            mean = float(vals.mean())
            std = float(vals.std()) + 1e-8
            stats.append((mean, std))
        self.norm_stats: list[tuple[float, float]] = stats

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        price_columns: Iterable[str] = ("open", "close"),
        base_end_date: str | pd.Timestamp | None = None,
        train_frac: float = 0.8,
    ) -> "BacktestData":
        """
        Build from a long-format DataFrame with ``date, ticker`` plus price columns.

        Rows missing any requested price column are dropped so a day where a
        market is closed does not enter the shared trading calendar.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        cols = list(price_columns)
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"price column {col!r} missing from DataFrame")

        df = df.dropna(subset=cols).sort_values(["date", "ticker"])

        frames = {}
        for col in cols:
            frame = df.pivot(index="date", columns="ticker", values=col)
            frame = frame.sort_index()
            frame = frame[frame.columns.sort_values()]
            frames[col] = frame
        open_px, close_px = frames["open"], frames["close"]
        if not open_px.columns.equals(close_px.columns):
            raise ValueError("open/close ticker sets differ after pivot")
        return cls(open_px, close_px, base_end_date=base_end_date, train_frac=train_frac)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        price_columns: Iterable[str] = ("open", "close"),
        base_end_date: str | pd.Timestamp | None = None,
        train_frac: float = 0.8,
    ) -> "BacktestData":
        """Build from the merged ``all_stocks.csv`` (or an equivalent long CSV)."""
        df = pd.read_csv(path, usecols=lambda c: c in ("date", "ticker") or c in price_columns)
        return cls.from_dataframe(
            df, price_columns=price_columns, base_end_date=base_end_date, train_frac=train_frac
        )

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    @property
    def num_series(self) -> int:
        return len(self.tickers)

    @property
    def n_rows(self) -> int:
        return len(self.dates)

    def index_of(self, ticker: str) -> int:
        return self._ticker_index[ticker]

    def market_of(self, ticker: str) -> str:
        return "cn" if _is_cn(ticker) else "us"

    def market_mask(self, market: str) -> np.ndarray:
        """[S] bool — series belonging to ``market`` ('cn' | 'us')."""
        return np.array([self.market_of(t) == market for t in self.tickers], dtype=bool)

    # ------------------------------------------------------------------
    # Day / window navigation
    # ------------------------------------------------------------------

    def row_of(self, date: str | pd.Timestamp) -> int:
        """Price-row index whose date is >= ``date`` (first trading row at/after)."""
        ts = pd.Timestamp(date)
        pos = bisect_left(self.dates, ts)
        if pos >= self.n_rows:
            raise ValueError(f"date {date} is past the last available row {self.dates[-1]}")
        return int(pos)

    def trading_rows(self, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> np.ndarray:
        """Row indices with ``start_date <= date <= end_date``."""
        lo = bisect_left(self.dates, pd.Timestamp(start_date))
        hi = bisect_right(self.dates, pd.Timestamp(end_date))
        return np.arange(lo, hi)

    def window_valid(self, hist_end_row: int, hist_len: int) -> np.ndarray:
        """[S] bool — series with at least one genuine return in ``[row-H+1, row]``."""
        lo = max(0, hist_end_row - hist_len + 1)
        return self.valid[lo : hist_end_row + 1].any(axis=0)

    def tradeable(self, row: int) -> np.ndarray:
        """
        [S] bool — series tradable on this signal row.

        Requires a finite close on ``row`` **and** a finite open on the fill row
        ``row + 1`` (orders are filled at the next open).  This is what keeps a
        closed market out of the allocation: e.g. CN names during Golden Week
        (US-only rows in the union calendar) are not tradable, and vice-versa on
        US holidays.  Without it the allocator targets a market it cannot fill,
        silently dropping half the reallocation and corrupting the partition
        books.
        """
        close_ok = np.isfinite(self.close[row])
        if row + 1 < self.n_rows:
            open_ok = np.isfinite(self.open[row + 1])
        else:
            open_ok = close_ok  # last row: no fill row, fall back to close
        return close_ok & open_ok

    def z_hist(
        self,
        hist_end_row: int,
        hist_len: int,
    ) -> np.ndarray:
        """
        Z-scored history window ending at ``hist_end_row``.

        Returns ``[S, hist_len]`` in float32 — the model input convention.
        Missing (0-filled) returns are normalized by the same per-series stats
        used at train time, mirroring ``CustomReturnDataset``.
        """
        if hist_end_row - hist_len + 1 < 0:
            raise ValueError(
                f"need {hist_len} rows of history before row {hist_end_row} "
                f"(only {hist_end_row + 1} available)"
            )
        lo = hist_end_row - hist_len + 1
        window = self.ret[lo : hist_end_row + 1].T.astype(np.float64)  # [S, H]
        means = np.array([m for m, _s in self.norm_stats], dtype=np.float64)[:, None]
        stds = np.array([s for _m, s in self.norm_stats], dtype=np.float64)[:, None]
        return ((window - means) / stds).astype(np.float32)

    # ------------------------------------------------------------------
    # Tensor conveniences (used by the strategy / samplers)
    # ------------------------------------------------------------------

    def norm_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-series ``(mean, std)`` as float32 tensors on ``device``."""
        means = torch.tensor([m for m, _s in self.norm_stats], dtype=torch.float32, device=device)
        stds = torch.tensor([s for _m, s in self.norm_stats], dtype=torch.float32, device=device)
        return means, stds
