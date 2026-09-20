"""
Per-sub-portfolio accounting for the rolling-partition strategy.

A :class:`Partition` owns a sub-portfolio's live state — position dollars per
ticker plus an attributed cash float — and its full trading history:

- ``snapshots``: per-day totals (NAV, dollars, cash, gross/net, period return).
- ``rolls``: every re-allocation event, carrying the *full per-ticker weight
  vector* so any partition's exact holdings on any date can be reconstructed.

History is written incrementally to an append-only CSV ledger (via a row sink)
so a long-running live venue (IBKR / Xueqiu) accumulates fine-grained history
on disk without holding it in RAM.  Live state (``_dollars`` / ``_cash`` /
``_nav``) stays in memory for the decisions that need it every step.

Cash attribution convention: a partition's cash is the residual that makes
``dollars.sum() + cash == partition NAV`` exactly (short proceeds accrue to the
partition's cash, mirroring the account).  Partition NAVs are re-anchored to
``account NAV / N`` on every roll (``share_reset: at_roll``).
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

#: Ledger row keys shared by snapshot and roll events.
ROW_KEYS = [
    "event", "date", "partition", "nav", "dollars", "cash", "gross", "net",
    "period_return", "cost", "cash_weight", "weights",
]

RowSink = Callable[[dict[str, Any]], None]


@dataclass
class PartitionSnapshot:
    """Per-day totals for one partition."""

    date: str
    partition: int
    nav: float
    dollars: float
    cash: float
    gross: float
    net: float
    period_return: float
    cost: float = 0.0
    cash_weight: float = 0.0


@dataclass
class RollRecord:
    """One re-allocation event, with the full per-ticker weight vector."""

    date: str
    partition: int
    nav_before_reset: float
    nav_after_reset: float
    period_return: float
    cost: float = 0.0
    cash_weight: float = 0.0
    weights: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["weights"] = json.dumps(self.weights.tolist()) if self.weights is not None else ""
        return d


class Partition:
    """
    A sub-portfolio book with drift, rebalance (roll), cash and full history.

    Parameters
    ----------
    index : int
        Partition id (0..N-1).
    n_series : int
        Universe size (position-dollar vector length).
    share_reset : str
        ``"at_roll"`` or ``"none"`` (see RollingPartitionStrategy).
    sink : RowSink | None
        Optional append-only row sink (e.g. a ``CsvLedger.append``) for durable
        CSV history.  ``None`` → pure in-memory.
    """

    def __init__(
        self,
        index: int,
        n_series: int,
        share_reset: str = "at_roll",
        sink: RowSink | None = None,
    ):
        self.index = index
        self.n_series = n_series
        self.share_reset = share_reset
        self.sink = sink

        self._dollars: np.ndarray = np.zeros(n_series, dtype=np.float64)
        self._cash: float = 0.0
        self._nav: float = 0.0
        self._base: float | None = None     # NAV at last deploy (period-return base)
        self._cost_total: float = 0.0

        self.snapshots: list[PartitionSnapshot] = []
        self.rolls: list[RollRecord] = []

    # ------------------------------------------------------------------
    # Live state
    # ------------------------------------------------------------------

    @property
    def dollars(self) -> np.ndarray:
        """Position dollars per ticker (a copy is *not* made; read-only by convention)."""
        return self._dollars

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def nav(self) -> float:
        return self._nav

    @property
    def gross(self) -> float:
        return float(np.abs(self._dollars).sum())

    @property
    def net(self) -> float:
        return float(self._dollars.sum())

    @property
    def cash_weight(self) -> float:
        return self._cash / self._nav if self._nav else 0.0

    @property
    def period_return(self) -> float:
        """Return since the last deploy (0.0 before the first deploy)."""
        if self._base is None or self._base <= 0:
            return 0.0
        return self._nav / self._base - 1.0

    def _update_nav(self) -> None:
        self._nav = float(self._dollars.sum()) + self._cash

    def refresh_nav(self) -> None:
        """Recompute NAV after external position/cash edits (reconciliation)."""
        self._update_nav()

    def set_dollar_value(self, i: int, value: float) -> None:
        """Set the position dollar value at series ``i`` (reconciliation)."""
        self._dollars[i] = float(value)

    def set_cash_value(self, value: float) -> None:
        """Set the partition's attributed cash (reconciliation)."""
        self._cash = float(value)
        self._update_nav()

    # ------------------------------------------------------------------
    # Daily mechanics
    # ------------------------------------------------------------------

    def drift(self, ret: np.ndarray) -> None:
        """Apply realized per-name returns; positions move, cash is static."""
        self._dollars *= (1.0 + ret)
        self._update_nav()

    def deploy(
        self,
        dollars: np.ndarray,
        total: float,
        date: str,
        cost: float = 0.0,
    ) -> np.ndarray:
        """
        Re-allocate the partition to ``dollars`` at a target total value.

        The partition's cash is the residual ``total - dollars.sum()`` so that
        ``dollars.sum() + cash == total`` exactly.  Records a roll event with
        the full weight vector; returns the per-ticker dollar delta.

        Parameters
        ----------
        dollars : np.ndarray
            New position dollars (assets only).
        total : float
            Target partition NAV (``account NAV / N`` for ``at_roll``, or the
            partition's own current value for ``none``).
        date : str
            ISO date of the roll signal.
        cost : float
            Costs charged to this partition's cash (commission, etc.).
        """
        nav_before = self._nav
        prev = self._dollars.copy()
        period_return = self.period_return

        self._dollars = dollars.astype(np.float64).copy()
        self._cash = float(total - self._dollars.sum())
        self._nav = float(total)
        self._base = self._nav
        self._cost_total += cost

        weights = self._dollars / self._nav if self._nav else self._dollars.copy()
        self.rolls.append(
            RollRecord(
                date=date, partition=self.index,
                nav_before_reset=nav_before, nav_after_reset=self._nav,
                period_return=period_return, cost=cost,
                cash_weight=self._cash / self._nav if self._nav else 0.0,
                weights=weights.copy(),
            )
        )
        self._emit(self._roll_row(self.rolls[-1]))
        return self._dollars - prev

    def charge_cost(self, cost: float) -> None:
        """Debit realised costs from this partition's cash (e.g. roll fills)."""
        if cost == 0.0:
            return
        self._cash -= cost
        self._cost_total += cost
        self._update_nav()

    def snapshot(self, date: str) -> PartitionSnapshot:
        """Record a per-day totals snapshot."""
        snap = PartitionSnapshot(
            date=date, partition=self.index, nav=self._nav,
            dollars=float(self._dollars.sum()), cash=self._cash,
            gross=self.gross, net=self.net, period_return=self.period_return,
            cash_weight=self.cash_weight,
        )
        self.snapshots.append(snap)
        self._emit(self._snapshot_row(snap))
        return snap

    # ------------------------------------------------------------------
    # History / analysis
    # ------------------------------------------------------------------

    def period_returns(self) -> list[float]:
        """Return earned over each deployed holding period (one per roll)."""
        return [r.period_return for r in self.rolls]

    def cumulative_return(self) -> float:
        """Compounded return across all rolls (chain of period returns)."""
        rets = self.period_returns()
        if not rets:
            return 0.0
        return float(np.prod([1.0 + r for r in rets]) - 1.0)

    @property
    def cost_total(self) -> float:
        return self._cost_total

    # ------------------------------------------------------------------
    # Ledger rows
    # ------------------------------------------------------------------

    def _roll_row(self, rec: RollRecord) -> dict[str, Any]:
        row = {
            "event": "roll",
            "date": rec.date,
            "partition": rec.partition,
            "nav": rec.nav_after_reset,
            "dollars": float(rec.weights.sum()) if rec.weights is not None else "",
            "cash": "",
            "gross": float(np.abs(rec.weights).sum()) if rec.weights is not None else "",
            "net": float(rec.weights.sum()) if rec.weights is not None else "",
            "period_return": rec.period_return,
            "cost": rec.cost,
            "cash_weight": rec.cash_weight,
            "weights": json.dumps(rec.weights.tolist()) if rec.weights is not None else "",
        }
        return {k: row.get(k, "") for k in ROW_KEYS}

    def _snapshot_row(self, snap: PartitionSnapshot) -> dict[str, Any]:
        row = {
            "event": "snapshot",
            "date": snap.date,
            "partition": snap.partition,
            "nav": snap.nav,
            "dollars": snap.dollars,
            "cash": snap.cash,
            "gross": snap.gross,
            "net": snap.net,
            "period_return": snap.period_return,
            "cost": snap.cost,
            "cash_weight": snap.cash_weight,
            "weights": "",
        }
        return {k: row.get(k, "") for k in ROW_KEYS}

    def _emit(self, row: dict[str, Any]) -> None:
        if self.sink is not None:
            self.sink(row)
