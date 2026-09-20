"""
Venue 0 — in-repo backtest executor.

Simulates order fills at a chosen execution price (default: the next session's
open), tracks a shares-and-cash book, applies optional commission/slippage, and
marks the book to market each close.

Accounting invariants (enforced by ``tests/test_backtest_accounting.py``):
- NAV identity at every mark: ``NAV == cash + sum(shares * price)``
- orders are dollar deltas: ``shares_delta = delta_weight * nav / fill_price``
- cash is always derived from the NAV identity, never from an allocator's
  internal cash column.

``MarketCost``/``CostModel`` now live in :mod:`trading.costs` (shared with the
live venues) and are re-exported here so existing imports keep working.
"""

from ..base import ExecutorBase, Fill, FillReport, Order
from ..costs import CostModel, MarketCost

__all__ = ["BacktestExecutor", "CostModel", "MarketCost"]


class BacktestExecutor(ExecutorBase):
    """
    Shares-and-cash backtest book.

    Parameters
    ----------
    costs : CostModel
        Commission and slippage configuration (defaults to zero).
    allow_short : bool
        Venue-level shortability backstop (per-market rules live in the
        PositionManager; this is the executor's own guard).
    long_only_tickers : set[str] | None
        Names that must never end up net short (e.g. CN A-shares).  A backstop
        against sub-share rounding dust pushing a long-only position negative.
    no_borrow : bool
        If True (default), a buy that would drive cash below zero is skipped, so
        the account never borrows.
    """

    def __init__(
        self,
        initial_cash: float = 1.0,
        costs: CostModel | None = None,
        allow_short: bool = True,
        long_only_tickers: set[str] | None = None,
        no_borrow: bool = True,
    ):
        super().__init__(initial_cash=initial_cash)
        self.costs = costs or CostModel()
        self.allow_short = allow_short
        self.long_only_tickers = long_only_tickers or set()
        self.no_borrow = no_borrow
        self._day_turnover = 0.0

    # ------------------------------------------------------------------
    # ExecutorBase API
    # ------------------------------------------------------------------

    def get_shares(self) -> dict[str, float]:
        return dict(self.shares)

    def get_nav(self) -> float:
        return self.last_nav

    def get_holdings(self) -> dict[str, float]:
        nav = self.last_nav
        if nav <= 0:
            return {}
        return {t: shares * self.last_prices.get(t, 0.0) / nav for t, shares in self.shares.items()}

    def mark(self, prices: dict[str, float]) -> float:
        self.last_prices = dict(prices)
        nav = self.cash + sum(sh * float(prices[t]) for t, sh in self.shares.items() if t in prices)
        self.last_nav = nav
        return nav

    def execute(self, orders: list[Order], prices: dict[str, float]) -> FillReport:
        nav = self.last_nav
        report = FillReport(nav_before=nav)
        self.bought_today = {}

        for order in orders:
            ticker = order.ticker
            if ticker not in prices or not _finite(prices[ticker]):
                report.skipped[ticker] = "no price"
                continue
            price = float(prices[ticker])
            delta = order.delta_weight

            # Per-name short backstop: a net sell below the current short position
            # is only allowed when the venue permits shorting.
            current_weight = self.shares.get(ticker, 0.0) * price / nav if nav > 0 else 0.0
            if not self.allow_short and current_weight + delta < -1e-12:
                delta = -current_weight
                if abs(delta) < 1e-12:
                    continue

            side = 1.0 if delta > 0 else -1.0
            eff_price = self.costs.hit_price(price, side)
            if eff_price <= 0:
                report.skipped[ticker] = "non-positive fill price"
                continue

            value = delta * nav
            shares_delta = value / eff_price
            new_shares = self.shares.get(ticker, 0.0) + shares_delta

            # Long-only backstop (e.g. CN A-shares): never end up net short from
            # a sell — clamp the fill so at most the held quantity is sold.
            if ticker in self.long_only_tickers and new_shares < 0.0:
                shares_delta = -self.shares.get(ticker, 0.0)
                new_shares = 0.0
                value = shares_delta * eff_price

            if abs(new_shares) < 1e-12:
                new_shares = 0.0

            commission = self.costs.commission_nav(value, shares_delta, ticker)

            # No-borrow backstop: a buy that would drive cash below zero is
            # skipped (the strategy's net-exposure cap should prevent this; this
            # guarantees cash ≥ 0 regardless).
            if self.no_borrow and shares_delta > 0.0 and (
                self.cash - shares_delta * eff_price - commission < 0.0
            ):
                report.skipped[ticker] = "no cash (borrow blocked)"
                continue

            self.shares[ticker] = new_shares
            self.cash -= shares_delta * eff_price
            self.cash -= commission

            report.fills.append(
                Fill(ticker=ticker, delta_weight=(value / nav if nav else 0.0),
                     shares=shares_delta, price=eff_price, value=value)
            )
            report.costs += commission
            report.turnover += abs(delta)
            if shares_delta > 0:
                self.bought_today[ticker] = self.bought_today.get(ticker, 0.0) + shares_delta

        # Drop zero-position entries to keep the book clean.
        self.shares = {t: s for t, s in self.shares.items() if abs(s) > 1e-12}
        self._day_turnover = report.turnover

        # Revalue at the same prices used for fills if a mark hasn't happened yet
        # (fills occur at open; NAV is marked at close in the runner loop).
        report.nav_after = self.cash + sum(
            s * float(prices[t]) for t, s in self.shares.items() if t in prices
        )
        return report


def _finite(x: float) -> bool:
    return x == x  # NaN check
