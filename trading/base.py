"""
Shared application-layer contract for all venues.

The model/allocator emits a target-weight vector (``TargetWeights`` =
``dict[str, float]``, ticker → weight, negative = short, **cash is implicit**).
A :class:`PositionManager` reads the executor's current holdings, computes the
diff, applies venue risk filters (missing price / suspended, per-market
``allow_short``, CN T+1, ``min_order_weight``), then dispatches to an
:class:`ExecutorBase`.  Each venue (backtest, IBKR, Xueqiu) is one executor.

Weight/cash convention: the executor is the source of truth for cash and NAV.
``cash = NAV - sum(shares * price)`` always holds; the allocator's internal
"cash column" is a normalisation device (``sum|w| + cash = 1``), not account
truth once shorts exist, so it is never part of a target.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

TargetWeights = dict[str, float]


@dataclass
class Order:
    """A planned weight change for one ticker (target-space delta)."""

    ticker: str
    delta_weight: float
    reason: str = ""        # free-form / categorical (roll, rebalance, overlay, ...)


@dataclass
class Fill:
    """One executed order leg."""

    ticker: str
    delta_weight: float
    shares: float
    price: float
    value: float


@dataclass
class FillReport:
    """Outcome of an execution batch."""

    fills: list[Fill] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)   # ticker -> reason
    costs: float = 0.0
    turnover: float = 0.0                                   # sum|delta_weight|
    nav_before: float = 0.0
    nav_after: float = 0.0


@dataclass
class MarketRule:
    """Per-market execution rules (mirrors the venue's own constraints)."""

    allow_short: bool = True
    t_plus_1: bool = False      # CN: shares bought this session cannot be sold same session
    price_limit: float | None = None   # |daily move| cap — informational for now


class ExecutorBase(ABC):
    """
    Interface every venue executor implements.

    Shared bookkeeping state lives in ``__init__`` so the :class:`PositionManager`
    filters can rely on it without knowing the venue:
      - ``shares`` / ``cash``   : the position book (venue-specific units)
      - ``last_prices``         : most recent mark prices (ticker -> price)
      - ``last_nav``            : most recent marked NAV
      - ``bought_today``        : shares acquired in the current fill session (CN T+1)
    """

    def __init__(self, initial_cash: float = 1.0):
        self.cash: float = initial_cash
        self.shares: dict[str, float] = {}
        self.bought_today: dict[str, float] = {}
        self.last_prices: dict[str, float] = {}
        self.last_nav: float = initial_cash

    # ------------------------------------------------------------------
    # Venue-agnostic diagnostics (computed from the shared book)
    # ------------------------------------------------------------------

    @property
    def gross_exposure(self) -> float:
        nav = self.last_nav or 1.0
        return sum(abs(s) * self.last_prices.get(t, 0.0) for t, s in self.shares.items()) / nav

    @property
    def net_exposure(self) -> float:
        nav = self.last_nav or 1.0
        return sum(s * self.last_prices.get(t, 0.0) for t, s in self.shares.items()) / nav

    @property
    def cash_weight(self) -> float:
        return self.cash / self.last_nav if self.last_nav else 1.0

    @abstractmethod
    def get_holdings(self) -> TargetWeights:
        """Current position weights vs the last marked NAV (tickers only)."""

    @abstractmethod
    def get_shares(self) -> dict[str, float]:
        """Current share quantities per ticker."""

    @abstractmethod
    def get_nav(self) -> float:
        """Most recently marked NAV."""

    @abstractmethod
    def mark(self, prices: dict[str, float]) -> float:
        """Revalue the book at ``prices`` and return the NAV."""

    @abstractmethod
    def execute(self, orders: list[Order], prices: dict[str, float]) -> FillReport:
        """Fill the planned orders at ``prices``; returns a report."""


class PositionManager:
    """
    Diff + risk-filter + dispatch layer shared by every venue.

    ``plan`` is pure (no trading): it diffs a target against current holdings,
    applies the risk filters and returns the orders to send.  The venue drives
    when fills happen (e.g. queue at close(t), fill at open(t+1)).
    """

    def __init__(
        self,
        executor: ExecutorBase,
        market_rules: dict[str, MarketRule] | None = None,
        default_market: str = "us",
        min_order_weight: float = 1e-6,
    ):
        self.executor = executor
        self.market_rules = market_rules or {}
        self.default_market = default_market
        self.min_order_weight = min_order_weight

    def market_of(self, ticker: str) -> str:
        if ticker.startswith("sh.") or ticker.startswith("sz."):
            return "cn"
        return self.default_market

    def _rule(self, ticker: str) -> MarketRule:
        return self.market_rules.get(self.market_of(ticker), MarketRule())

    def plan(self, target: TargetWeights) -> list[Order]:
        """
        Compute the filtered orders to reach ``target`` from current holdings.

        Filters (in order):
        1. names with no held position and a sub-threshold target are skipped;
        2. non-tradeable direction per market (short clamp for long-only);
        3. CN T+1: a sell cannot exceed shares held before today's session;
        4. sub-threshold residual deltas are dropped (float noise).
        """
        current = self.executor.get_holdings()
        shares = self.executor.get_shares()
        nav = self.executor.get_nav()
        tickers = set(current) | set(target)
        orders: list[Order] = []

        for ticker in sorted(tickers):
            w_cur = current.get(ticker, 0.0)
            w_tgt = target.get(ticker, 0.0)
            delta = w_tgt - w_cur
            if abs(delta) < self.min_order_weight:
                continue

            rule = self._rule(ticker)

            # Short clamp for long-only markets: never end up short.
            if not rule.allow_short:
                if w_tgt < 0:
                    w_tgt = 0.0
                    delta = w_tgt - w_cur
                    if abs(delta) < self.min_order_weight:
                        continue
                if delta < 0 and w_cur <= 0:
                    continue  # nothing long to sell

            # CN T+1: selling this session can only reference pre-session shares.
            if rule.t_plus_1 and delta < 0:
                held_before = shares.get(ticker, 0.0) - self.executor.bought_today.get(ticker, 0.0)
                if held_before <= 1e-12:
                    continue
                price = self._price_of(ticker)
                if price is None or nav <= 0:
                    continue  # cannot price the feasibility check; leave the order as-is
                max_sell_shares = min(shares.get(ticker, 0.0), held_before)
                sell_shares = -delta * nav / price
                if sell_shares > max_sell_shares + 1e-9:
                    delta = -(max_sell_shares * price) / nav
                    if abs(delta) < self.min_order_weight:
                        continue

            if abs(delta) < self.min_order_weight:
                continue
            orders.append(Order(ticker=ticker, delta_weight=delta))

        return orders

    def _price_of(self, ticker: str) -> float | None:
        prices = self.executor.last_prices or {}
        price = prices.get(ticker)
        return float(price) if price is not None and price == price else None
