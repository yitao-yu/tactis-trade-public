"""
Span-forecast rolling-partition strategy (Venue 0 default).

The account NAV is split into ``N`` virtual :class:`Partition` sub-books
(default ``N = pred_len``).  Each trading day exactly one partition — the one
whose ``(day % N)`` index is due — is re-allocated from a fresh model sample:

    1. sample the joint predictive distribution for the next ``pred_len`` days;
    2. de-normalize each horizon step and compound ``span = prod(1 + r_h) - 1``
       per sample path (the paths are joint across series *and* time, so the
       compounded span is a legitimate joint draw);
    3. hand the span samples to the allocator unchanged (``[1, series, n]`` is
       its input contract), passing the due partition's current reconciled
       weights (with its attributed cash) as ``prev_weights`` for the turnover
       penalty;
    4. the partition's dollar book is reset to ``NAV / N`` (``share_reset:
       at_roll``) or redeploys its own value (``share_reset: none``).

Bookkeeping is grounded in the ACCOUNT every day: before each roll the
partition books are *reconciled* to the executor's actual position dollars
and cash (``reconcile``), so skipped fills, missing prices and open-vs-close
gaps cannot let the books drift from reality.  Without this the emitted
deltas chase a phantom book and the account silently accumulates unbounded
leverage.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .base import TargetWeights
from .ledger import CsvLedger
from .portfolio import ROW_KEYS, Partition

import inspect

ALLOCATE_PREV_SUPPORT = {"var_deployment"}


def _supports_prev_weights(allocator) -> bool:
    """
    True when the allocator's ``allocate`` accepts ``prev_weights``.

    Capability check on the signature (not a name list — the earlier
    class-name-vs-config-name mismatch silently disabled the turnover penalty).
    """
    try:
        params = inspect.signature(allocator.allocate).parameters
    except (TypeError, ValueError):
        return False
    if "prev_weights" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class RollingPartitionStrategy:
    """
    Parameters
    ----------
    allocator : BaseAllocator
        Allocator instance (built from config, span-tuned bands).
    tickers : list[str]
        Universe in model series order.
    n_partitions : int
        Number of virtual books; one rolls per trading day.
    share_reset : str
        ``"at_roll"`` — reset the rolling partition's NAV to ``NAV / N``.
        ``"none"`` — the partition redeploys its own value.
    sampler : Callable[[int], torch.Tensor]
        ``sample_span(row) -> [1, S, n_samples]`` raw (de-normalized) span
        returns on the model device.
    valid_fn : Callable[[int], np.ndarray]
        ``valid(row) -> [S]`` bool — which names are investable at ``row``.
    min_valid_series : int
        Minimum number of investable series required to re-allocate the due
        partition.  Below this the partition stays put (target = current
        holdings, no orders) — prevents single-name concentration in sparse
        windows (early history, a losing-symbols tail).
    history_dir : Path | None
        If set, each partition appends its fine-grained history (snapshots +
        rolls with full weight vectors) to ``<history_dir>/partition_{p}.csv``.
    max_gross_exposure : float | None
        Account-level cap on ``Σ|w|``; the due partition's delta is scaled so
        the account target cannot exceed it (implies net ≤ 1 ⇒ cash ≥ 0).
        ``None`` disables the cap.
    max_net_exposure : float | None
        Account-level cap on ``Σw`` (net).  ``1.0`` keeps account cash ≥ 0 (no
        borrowing).  ``None`` disables.
    prev_weight_clip : float | None
        Clamp ``prev_weights`` magnitude before the allocator's turnover penalty,
        so a degenerate partition NAV can't poison the reference.  ``None`` off.
    """

    def __init__(
        self,
        allocator,
        tickers: list[str],
        n_partitions: int,
        share_reset: str = "at_roll",
        sampler: Callable[[int], torch.Tensor] | None = None,
        valid_fn: Callable[[int], np.ndarray] | None = None,
        min_valid_series: int = 1,
        max_gross_exposure: float | None = 1.0,
        max_net_exposure: float | None = 1.0,
        prev_weight_clip: float | None = None,
        history_dir: Path | None = None,
    ):
        if share_reset not in ("at_roll", "none"):
            raise ValueError(f"share_reset must be 'at_roll' or 'none', got {share_reset!r}")
        self.allocator = allocator
        self.tickers = tickers
        self.S = len(tickers)
        self.n_partitions = int(n_partitions)
        self.share_reset = share_reset
        self.sampler = sampler
        self._valid_fn = valid_fn or (lambda _row: np.ones(self.S, dtype=bool))
        self.min_valid_series = int(min_valid_series)
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure
        self.prev_weight_clip = prev_weight_clip

        self.partitions: list[Partition] = []
        self._ledgers: list[CsvLedger] = []
        for p in range(self.n_partitions):
            sink = None
            if history_dir is not None:
                ledger = CsvLedger(Path(history_dir) / f"partition_{p}_history.csv", ROW_KEYS)
                self._ledgers.append(ledger)
                sink = ledger.append
            self.partitions.append(
                Partition(index=p, n_series=self.S, share_reset=share_reset, sink=sink)
            )
        self.step = 0
        self._last_due = 0

    # ------------------------------------------------------------------
    # Daily accounting: ground the books in the account's reality
    # ------------------------------------------------------------------

    def reconcile(self, account_dollars: np.ndarray, cash: float) -> None:
        """
        Re-anchor every partition book to the executor's actual position
        dollars and cash.  Names the account no longer holds are zeroed
        (kills phantom books from skipped fills); held names are distributed
        across partitions as a slice of the *actual* position, weighted by each
        partition's **gross** book share on that name.

        Using gross (absolute) shares — not signed ``book_p / Σ book`` — is what
        keeps this stable: when two partitions held opposite-sign claims on the
        same name, ``Σ book`` was near zero and the signed ratio blew the
        attribution up into a phantom large short, which later triggered a huge
        "correction" order (gross > 1, negative cash).  Gross shares sum to 1
        whenever any book exists, so ``|p.dollars_i| ≤ |account_dollars_i|``
        always holds.  After reconciliation ``sum(partition navs) == NAV``.

        ``account_dollars`` is ``[S]`` float — share count × last mark price
        per ticker.
        """
        eps = 1e-9
        for i in range(self.S):
            actual = float(account_dollars[i])
            if abs(actual) < eps:
                for p in self.partitions:
                    p.set_dollar_value(i, 0.0)
                continue
            gross_i = float(sum(abs(float(p.dollars[i])) for p in self.partitions))
            if gross_i < eps:
                owner = next(
                    (p for p in self.partitions
                     if float(np.abs(p.dollars).sum()) > eps),
                    None,
                )
                (owner or self.partitions[0]).set_dollar_value(i, actual)
            else:
                for p in self.partitions:
                    p.set_dollar_value(i, actual * abs(float(p.dollars[i])) / gross_i)

        total_gross = float(sum(np.abs(p.dollars).sum() for p in self.partitions))
        if total_gross > eps:
            # Normalise cash by the SUM of partition grosses: mixed-sign books
            # (one partition long, another short the same name) make the sum of
            # partition grosses exceed the account's net-position gross, and
            # dividing by the account gross would multiply the cash several
            # times over.
            for p in self.partitions:
                p.set_cash_value(cash * float(np.abs(p.dollars).sum()) / total_gross)
        else:
            for p in self.partitions:
                p.set_cash_value(cash / self.n_partitions)
        for p in self.partitions:
            p.refresh_nav()

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def on_close(
        self,
        date: str,
        row: int,
        nav: float,
        account_dollars: np.ndarray,
        cash: float,
        device: torch.device,
    ) -> TargetWeights:
        """
        Produce the next account-level target after the close of ``row``.

        Parameters
        ----------
        date : str
            ISO date of this close.
        row : int
            Backtest row whose close just happened (history through ``row``).
        nav : float
            Account NAV after marking close(row).
        account_dollars : np.ndarray
            ``[S]`` actual position dollars per ticker at the last mark
            (share count × last mark price) — the ground truth for the books.
        cash : float
            Account cash at the last mark.
        device : torch.device
            Where allocator tensors live.

        Returns the full target weight vector; the caller diffs it against the
        executor's holdings (via the PositionManager) and queues the orders.
        """
        self.reconcile(account_dollars, cash)

        p = self.step % self.n_partitions
        self.step += 1
        self._last_due = p

        target: TargetWeights = {
            t: float(account_dollars[i] / nav)
            for i, t in enumerate(self.tickers)
            if abs(account_dollars[i]) > 1e-12
        }
        if self.sampler is None:
            return target

        n_valid = int(np.count_nonzero(self._valid_fn(row)))
        if n_valid < self.min_valid_series:
            return target  # sparse window — stay put instead of concentrating

        span = self.sampler(row)                                  # [1, S, n]
        valid = torch.as_tensor(self._valid_fn(row), dtype=torch.bool, device=device)
        aug_samples, aug_mask, cash_idx = self._augment(span, valid)
        prev = self._prev_weights(p, device)

        w_out = self._allocate(aug_samples, aug_mask, cash_idx, prev)
        w_assets = w_out[0, : self.S].detach().cpu().numpy()

        part = self.partitions[p]
        total = self._deploy_total(p, nav)
        prev_book = part.dollars.copy()
        new_dollars = total * w_assets.astype(np.float64)

        # Preserve non-tradable positions (closed market / suspended): the
        # allocator zeroes masked names, which would otherwise emit a sell for
        # frozen holdings that can't be filled today.  Carry them unchanged.
        hold = ~self._valid_fn(row)
        new_dollars[hold] = prev_book[hold]

        raw_delta = new_dollars - prev_book
        scale = self._delta_scale(account_dollars, raw_delta, nav)
        delta_dollars = part.deploy(prev_book + scale * raw_delta, total, date)

        for i, tkr in enumerate(self.tickers):
            dw = delta_dollars[i]
            if abs(dw) > 1e-9 * nav:
                target[tkr] = float(account_dollars[i] / nav) + dw / nav
        return target

    def _delta_scale(
        self,
        account_dollars: np.ndarray,
        raw_delta: np.ndarray,
        nav: float,
    ) -> float:
        """
        Largest ``s ∈ [0, 1]`` such that ``account + s·delta`` satisfies the
        account-level gross / net caps.  ``s = 0`` freezes the due partition.
        """
        if nav <= 0:
            return 0.0
        g0 = float(np.abs(account_dollars).sum()) / nav
        n0 = float(account_dollars.sum()) / nav
        gd = float(np.abs(raw_delta).sum()) / nav
        nd = float(raw_delta.sum()) / nav
        s = 1.0
        if self.max_gross_exposure is not None and gd > 1e-12:
            s = min(s, max(0.0, (self.max_gross_exposure - g0) / gd))
        if self.max_net_exposure is not None and nd > 1e-12:
            s = min(s, max(0.0, (self.max_net_exposure - n0) / nd))
        return float(min(1.0, max(0.0, s)))

    def charge_due_costs(self, cost: float) -> None:
        """Debit realised fill costs to the partition that last rolled."""
        if cost:
            self.partitions[self._last_due].charge_cost(cost)

    def snapshot_all(self, date: str) -> None:
        """Record a per-day totals snapshot for every partition."""
        for part in self.partitions:
            part.snapshot(date)

    def close(self) -> None:
        for ledger in self._ledgers:
            ledger.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deploy_total(self, p: int, nav: float) -> float:
        if self.share_reset == "at_roll":
            return nav / self.n_partitions
        own = self.partitions[p].nav
        return own if own > 0 else nav / self.n_partitions

    def _augment(
        self,
        span: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch = span.shape[0]
        cash_samples = torch.zeros(batch, 1, span.shape[2], device=span.device)
        cash_mask = torch.ones(batch, 1, dtype=torch.bool, device=span.device)
        aug = torch.cat([span, cash_samples], dim=1)
        mask = torch.cat([valid[None, :], cash_mask], dim=1)
        return aug, mask, self.S

    def _prev_weights(self, p: int, device: torch.device) -> torch.Tensor:
        """
        Due partition's current weight vector (assets + attributed cash) as
        fractions of its own reconciled NAV — the reference for the allocator's
        turnover penalty.
        """
        part = self.partitions[p]
        navp = part.nav if part.nav > 0 else 1.0
        assets = (part.dollars / navp).astype(np.float32)
        if self.prev_weight_clip is not None:
            assets = np.clip(assets, -self.prev_weight_clip, self.prev_weight_clip)
        cash_w = max(0.0, 1.0 - float(np.abs(assets).sum()))
        vec = np.concatenate([assets, [cash_w]]).astype(np.float32)
        return torch.from_numpy(vec)[None].to(device)

    def _allocate(self, samples, mask, cash_idx: int, prev):
        kwargs = dict(samples=samples, mask=mask, cash_index=cash_idx)
        if _supports_prev_weights(self.allocator):
            kwargs["prev_weights"] = prev
        with torch.enable_grad():
            return self.allocator.allocate(**kwargs)