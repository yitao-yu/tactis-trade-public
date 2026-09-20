"""
Append-only, crash-durable CSV ledger shared by every venue.

A ``CsvLedger`` owns one CSV file (per account / per partition), appends
dict-rows tagged with an ``event`` column, and flushes incrementally so a
long-running live venue (IBKR / Xueqiu) accumulates durable history on disk
without holding it all in RAM.  Restart recovery reads the tail of the file
(keyed by a monotonic ``seq`` when present).

The backtest writes the same files its run produces, doubling as the
integration test harness for the shared ledger.
"""

import csv
from pathlib import Path
from typing import Any

import pandas as pd


class CsvLedger:
    """
    Append-only CSV writer with per-row flush.

    Parameters
    ----------
    path : str | Path
        CSV file path (created with its header on first use).
    columns : list[str]
        Column order for every appended row.
    """

    def __init__(self, path: str | Path, columns: list[str]):
        self.path = Path(path)
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(self.columns)
            self._file.flush()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def append(self, row: dict[str, Any]) -> None:
        """Append one row (missing columns are left blank) and flush."""
        self._writer.writerow([row.get(c, "") for c in self.columns])
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "CsvLedger":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self) -> pd.DataFrame:
        """Read the full ledger back into a DataFrame (analysis / restart)."""
        self._file.flush()
        return pd.read_csv(self.path)

    @classmethod
    def read_file(cls, path: str | Path) -> pd.DataFrame:
        return pd.read_csv(path)
