"""
Transaction-cost model shared by every venue.

The backtest (Venue 0) and the live venues (IBKR) both need to price fills the
same way, so the schedule lives here rather than inside a single executor.
``MarketCost``/``CostModel`` were originally defined in
``trading/venues/backtest.py``; that module re-exports them for backward
compatibility, and ``build_cost_model`` replaces the runner's private
``_build_cost_model`` so the config-to-model mapping has one home.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MarketCost:
    """
    Per-market cost parameters.

    Two shapes are supported:
    - percentage of notional (``bps``), e.g. CN A-shares ~3 bp/side;
    - per-share with per-order floor and %-of-notional cap (IBKR):
      ``per_share`` ($/share) clamped to ``[min_per_order, max_pct·notional]``.
    """

    bps: float = 0.0
    per_share: float = 0.0
    min_per_order: float = 0.0
    max_pct: float = 0.0          # cap as a fraction of notional (0 = no cap)

    def commission(self, notional: float, shares: float) -> float:
        n = abs(notional)
        if self.per_share > 0:
            fee = self.per_share * abs(shares)
        else:
            fee = self.bps / 1e4 * n
        if self.min_per_order > 0:
            fee = max(fee, self.min_per_order)
        if self.max_pct > 0:
            fee = min(fee, self.max_pct * n)
        return fee


@dataclass
class CostModel:
    """
    Transaction costs for the backtest.

    ``commission_bps`` applies to every market unless explicit per-market
    ``us``/``cn`` ``MarketCost`` objects are given.  ``account_notional`` is the
    real account size in currency units: the backtest book is unit-normalised
    (NAV ≈ 1), so per-share/per-order fees (in $) are converted via
    ``fee_nav = fee_dollars / account_notional``.
    """

    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    account_notional: float = 1.0
    us: MarketCost | None = None
    cn: MarketCost | None = None

    def __post_init__(self) -> None:
        if self.us is None:
            self.us = MarketCost(bps=self.commission_bps)
        if self.cn is None:
            self.cn = MarketCost(bps=self.commission_bps)

    def market_cost(self, ticker: str) -> MarketCost:
        """CN A-shares use the ``cn`` schedule; everything else ``us``."""
        if ticker.startswith("sh.") or ticker.startswith("sz."):
            return self.cn  # type: ignore[return-value]
        return self.us  # type: ignore[return-value]

    def commission_nav(self, notional_nav: float, shares_nav: float, ticker: str) -> float:
        """Commission in NAV units (account_notional scales $ fees → unit NAV)."""
        if self.account_notional <= 0:
            return 0.0
        notional_dollars = notional_nav * self.account_notional
        shares_real = shares_nav * self.account_notional
        fee_dollars = self.market_cost(ticker).commission(notional_dollars, shares_real)
        return fee_dollars / self.account_notional

    def hit_price(self, price: float, side: float) -> float:
        """Fill price for a buy (+1) / sell (-1): slippage is adverse."""
        return price * (1 + self.slippage_bps / 1e4 * side)


def build_cost_model(b: Any) -> CostModel:
    """
    Build a :class:`CostModel` from a config node.

    ``commission_model: bps``  → flat ``commission_bps`` on all markets.
    ``commission_model: ibkr`` → US per-share with per-order floor and %-cap
    (``us_per_share`` / ``us_min_per_order`` / ``us_max_pct``), CN at
    ``cn_commission_bps`` (default 3 bp).  ``account_notional`` scales the
    unit NAV book to real dollars for per-share/per-order fees.
    """
    slippage = float(b.get("slippage_bps", 0.0))
    notional = float(b.get("account_notional", 1.0))
    model = str(b.get("commission_model", "bps")).lower()
    if model == "ibkr":
        us = MarketCost(
            per_share=float(b.get("us_per_share", 0.005)),
            min_per_order=float(b.get("us_min_per_order", 1.0)),
            max_pct=float(b.get("us_max_pct", 0.01)),
        )
        cn = MarketCost(bps=float(b.get("cn_commission_bps", 3.0)))
        return CostModel(slippage_bps=slippage, account_notional=notional, us=us, cn=cn)
    if model == "rate":
        us = MarketCost(bps=float(b.get("us_commission_bps", 7.5)))
        cn = MarketCost(bps=float(b.get("cn_commission_bps", 3.0)))
        return CostModel(slippage_bps=slippage, account_notional=notional, us=us, cn=cn)
    return CostModel(
        commission_bps=float(b.get("commission_bps", 0.0)),
        slippage_bps=slippage,
        account_notional=notional,
    )
