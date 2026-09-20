"""
Datasets for TACTiS training.

- MitsuiReturnDataset:  mitsui-commodity-prediction-challenge (single CSV, named columns)
- CustomReturnDataset:  yfinance/baostock CSVs (per-ticker files or merged CSV)

Both compute daily arithmetic returns from a single price column (close or open),
store returns and a per-position validity mask separately, apply per-series z-score
normalization from train-split statistics, and provide walk-forward windowed samples.

``__getitem__`` returns a 3-tuple ``(hist, pred, pred_mask)``:
- ``hist`` / ``pred``: normalized returns, NaN-filled with 0.0
- ``pred_mask``: boolean tensor — True for real prediction targets, False for NaN-filled positions that should not contribute to the loss
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MITSUI_DIR = Path(__file__).resolve().parent / "mitsui-commodity-prediction-challenge"
CUSTOM_US_DIR = Path(__file__).resolve().parent / "custom-data" / "raw" / "us"
CUSTOM_CN_DIR = Path(__file__).resolve().parent / "custom-data" / "raw" / "cn"
CUSTOM_MERGED = Path(__file__).resolve().parent / "custom-data" / "raw" / "all_stocks.csv"


def _normalize(
    data: np.ndarray,
    mask: np.ndarray,
    train_end: int,
) -> list[tuple[float, float]]:
    """
    Per-series z-score normalization using train-region statistics only.

    Parameters
    ----------
    data : np.ndarray  [num_series, timesteps]
        Returns matrix (NaN already filled with 0.0).  Modified in-place.
    mask : np.ndarray  [num_series, timesteps]
        Boolean mask — True where the return is genuine data.
    train_end : int
        Index of the last timestep in the train split (first ``splits[0]`` fraction).

    Returns
    -------
    list[tuple[float, float]]
        ``[(mean, std), ...]`` per series, usable with ``_apply_norm_stats``.
    """
    stats = []
    for i in range(data.shape[0]):
        valid = mask[i, :train_end]
        if valid.sum() < 2:
            data[i] = 0.0
            mask[i] = False
            stats.append((0.0, 1.0))
            continue
        train_vals = data[i, :train_end][valid]
        mean = float(np.mean(train_vals))
        std = float(np.std(train_vals)) + 1e-8
        data[i] = (data[i] - mean) / std
        stats.append((mean, std))
    return stats


def _apply_norm_stats(
    data: np.ndarray,
    stats: list[tuple[float, float]],
) -> None:
    """Apply pre-computed normalization stats in-place (e.g. for official test set)."""
    for i in range(data.shape[0]):
        mean, std = stats[i]
        data[i] = (data[i] - mean) / std


def _price_columns(
    df: pd.DataFrame,
    price_column: str = "close",
) -> list[str]:
    """
    Select price columns from the mitsui dataset DataFrame.

    ``price_column`` must be ``"close"`` or ``"open"``.

    Close mode includes every asset: LME → ``_Close``, JPX → ``_Close``,
    FX → bare name, US → ``_adj_close``.  Open mode includes only JPX
    ``_Open`` columns — LME, FX, and US assets have no open prices and are
    dropped (a warning is printed with the count).
    """
    if price_column not in ("close", "open"):
        raise ValueError(
            f"price_column must be 'close' or 'open', got {price_column!r}"
        )

    def _match(col: str, suffix: str | None) -> bool:
        """``suffix=None`` selects JPX open columns only; otherwise close-mode."""
        if col == "date_id":
            return False
        if suffix is None:
            return col.startswith("JPX_") and col.endswith("_Open")
        if col.startswith("LME_") and col.endswith("_Close"):
            return True
        if col.startswith("JPX_") and col.endswith(suffix):
            return True
        if col.startswith("FX_") and "_" not in col[len("FX_"):]:
            return True
        if col.startswith("US_") and col.endswith("_adj_close"):
            return True
        return False

    if price_column == "open":
        names = [col for col in df.columns if _match(col, None)]
        close_only = [col for col in df.columns if _match(col, "_Close")]
        dropped = [col for col in close_only if col not in names]
        if dropped:
            print(
                f"[MitsuiReturnDataset] price_column='open': dropped "
                f"{len(dropped)} close-only assets (LME/FX/US have no open "
                f"prices); using {len(names)} JPX open series only. "
                f"Recommend price_column='close' for full 143-series coverage."
            )
    else:
        names = [col for col in df.columns if _match(col, "_Close")]
    return names


class _BaseDataset(Dataset):
    """
    Shared sliding-window logic.

    Subclasses must set ``self.data`` and ``self.mask`` before calling
    ``_compute_split_bounds()``.
    """

    def _compute_split_bounds(self) -> tuple[int, int]:
        total = self.data.shape[1]
        boundary = int(total * self.splits[0])
        boundary2 = int(total * (self.splits[0] + self.splits[1]))

        if self.split == "train":
            return (0, boundary - self.hist_len - self.pred_len)
        elif self.split == "val":
            return (boundary, boundary2 - self.hist_len - self.pred_len)
        else:
            return (boundary2, total - self.hist_len - self.pred_len)

    def __len__(self) -> int:
        return max(0, self.window_end - self.window_start + 1)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.window_start + idx
        H = self.hist_len
        P = self.pred_len
        hist = self.data[:, start : start + H]
        pred = self.data[:, start + H : start + H + P]
        pred_mask = self.mask[:, start + H : start + H + P]
        return (
            torch.from_numpy(hist.copy()),
            torch.from_numpy(pred.copy()),
            torch.from_numpy(pred_mask.copy()),
        )

    @property
    def num_series(self) -> int:
        return self.data.shape[0]

    @property
    def timesteps(self) -> int:
        return self.data.shape[1]


class MitsuiReturnDataset(_BaseDataset):
    """
    Mitsui commodity-prediction-challenge dataset.

    143 assets → 143 TACTiS series.
    1,961 trading days in train.csv, 134 in official test.csv.
    """

    def __init__(
        self,
        hist_len: int,
        pred_len: int,
        split: Literal["train", "val", "test"],
        splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        data_dir: str | Path | None = None,
        price_column: str = "close",
        use_official_test: bool = False,
        norm_stats: list[tuple[float, float]] | None = None,
    ):
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.split = split
        self.splits = splits
        self.seed = seed
        self.price_column = price_column
        self.use_official_test = use_official_test
        self.official_test = use_official_test and split == "test"
        self.data_dir = Path(data_dir) if data_dir else MITSUI_DIR

        self.data, self.mask = self._load_and_process(norm_stats)
        self.norm_stats = norm_stats
        if self.official_test:
            self.window_start = 0
            self.window_end = self.data.shape[1] - self.hist_len - self.pred_len
        else:
            self.window_start, self.window_end = self._compute_split_bounds()

    def _load_dataframe(self) -> pd.DataFrame:
        if self.official_test:
            df_train = pd.read_csv(self.data_dir / "train.csv")
            df_test = pd.read_csv(self.data_dir / "test.csv")
            return pd.concat([df_train.tail(1), df_test], ignore_index=True)
        return pd.read_csv(self.data_dir / "train.csv")

    def _load_and_process(
        self,
        norm_stats: list[tuple[float, float]] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        df = self._load_dataframe()
        data, mask = self._load(df)

        if norm_stats is not None:
            _apply_norm_stats(data, norm_stats)
        else:
            train_end = int(data.shape[1] * self.splits[0])
            self.norm_stats = _normalize(data, mask, train_end)

        return data, mask

    def _load(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a mitsui DataFrame into a returns matrix and validity mask.

        Returns
        -------
        data : np.ndarray  [num_assets, timesteps] float32
            Arithmetic returns, NaN / inf → 0.0, NOT normalized.
        mask : np.ndarray  [num_assets, timesteps] bool
            True where the return is finite (genuine data).
        """
        cols = _price_columns(df, self.price_column)
        prices = df[cols].values.astype(np.float64)

        returns = (prices[1:] - prices[:-1]) / prices[:-1]
        mask = np.isfinite(returns)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return returns.T, mask.T


class CustomReturnDataset(_BaseDataset):
    """
    US + CN stocks from yfinance / baostock.

    Prefers the merged ``all_stocks.csv``; falls back to per-ticker CSV files.
    1,302 tickers → 1,302 TACTiS series.  4,416 return timesteps (2008–2024).
    """

    def __init__(
        self,
        hist_len: int,
        pred_len: int,
        split: Literal["train", "val", "test"],
        splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        us_dir: str | Path | None = None,
        cn_dir: str | Path | None = None,
        merged_path: str | Path | None = None,
        price_column: str = "close",
    ):
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.split = split
        self.splits = splits
        self.seed = seed
        self.price_column = price_column
        self.us_dir = Path(us_dir) if us_dir else CUSTOM_US_DIR
        self.cn_dir = Path(cn_dir) if cn_dir else CUSTOM_CN_DIR
        self.merged_path = Path(merged_path) if merged_path else CUSTOM_MERGED

        self.data, self.mask = self._load_and_process()
        self.window_start, self.window_end = self._compute_split_bounds()

    def _load_and_process(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load raw prices, compute arithmetic returns, normalize.

        Normalization uses statistics from the first ``splits[0]`` fraction of
        timesteps (train region) and is applied in-place to the full timeline.
        """
        data, mask = self._load()

        train_end = int(data.shape[1] * self.splits[0])
        self.norm_stats = _normalize(data, mask, train_end)
        return data, mask

    def _load_raw_df(self, cols: list[str]) -> pd.DataFrame:
        """
        Read price columns from disk as a long-format DataFrame.

        Prefers the merged ``all_stocks.csv``; falls back to concatenating
        per-ticker CSV files from ``us/`` and ``cn/`` directories.

        Parameters
        ----------
        cols : list[str]
            Price column names to include (e.g. ``["close"]``).

        Returns
        -------
        pd.DataFrame
            Long-format with columns ``["date", "ticker", *cols]`` — one
            row per (date, ticker) pair.  ``date`` is cast to datetime.
        """
        if self.merged_path.exists():
            df = pd.read_csv(self.merged_path, usecols=["date", "ticker"] + cols)
        else:
            dfs = []
            for ticker_dir in [self.us_dir, self.cn_dir]:
                if not ticker_dir.exists():
                    continue
                for csv_path in sorted(ticker_dir.glob("*.csv")):
                    dfs.append(pd.read_csv(csv_path, usecols=["date", "ticker"] + cols))
            df = pd.concat(dfs, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _pivot(self, df: pd.DataFrame, col: str) -> np.ndarray:
        """
        Pivot a long-format DataFrame into a wide price matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Long-format with ``["date", "ticker", col]``.
        col : str
            Column to pivot (e.g. ``"close"``, ``"open"``).

        Returns
        -------
        np.ndarray
            Shape ``(timesteps, num_tickers)`` — rows sorted by date,
            columns sorted alphabetically by ticker.
        """
        prices = df.pivot(index="date", columns="ticker", values=col)
        prices = prices.sort_index()
        tickers = sorted(prices.columns.tolist())
        return prices[tickers].values.astype(np.float64)

    def _load(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Pivot raw prices into a wide returns matrix with validity mask.

        Returns
        -------
        data : np.ndarray  [num_assets, timesteps] float32
            Arithmetic returns, NaN / inf → 0.0, NOT normalized.
        mask : np.ndarray  [num_assets, timesteps] bool
            True where the return is finite (genuine data).
        """
        df = self._load_raw_df([self.price_column])
        prices = self._pivot(df, self.price_column)

        returns = (prices[1:] - prices[:-1]) / prices[:-1]
        mask = np.isfinite(returns)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return returns.T, mask.T
