"""
Venue 1 — IBKR executor (paper or live).

Trades **US** stocks (long + short where shortable) and **China A-shares** via
Stock Connect (``sh.``/``sz.``; long-only — Stock Connect has no shorting).
All broker I/O goes through an injectable :class:`IBKRBroker` so the executor is
fully testable offline; the real broker lazily imports ``ib_async`` (with an
``ib_insync`` fallback) and only connects when built via
:meth:`IBKRBroker.from_config`.

Per-market behavior (contracts, lot size, shortability) is driven by the
``markets`` table; see :data:`DEFAULT_MARKETS`.  US symbols use SMART routing
with a ``.`` → `` `` transform (BRK.B → BRK B); CN symbols strip the exchange
prefix (``sh.600000`` → ``600000``) and route ``SEHKNTL`` (Shanghai) /
``SEHKSZSE`` (Shenzhen) in ``CNH``.

Safety: every order is gated by :class:`trading.live.LiveGuard` — the default is
dry-run (log only, no submission, **no book mutation**), and live requires both
``live: true`` in config and the ``TACTIS_LIVE`` env var.  A kill-switch and a
per-order notional cap apply on top.

No-look-ahead discipline mirrors Venue 0: the runner computes a target at
close(t) and submits at-the-open (``tif=OPG``) orders for open(t+1).
"""

import math
from typing import Any

from ..base import ExecutorBase, Fill, FillReport, Order
from ..costs import CostModel, build_cost_model
from ..live import LiveGuard

#: Default US stock contract (IBKR SMART-routed).
DEFAULT_US_SPEC: dict[str, Any] = {
    "secType": "STK",
    "exchange": "SMART",
    "currency": "USD",
    "primaryExchange": None,
}

#: Per-market defaults.  ``lot_size`` = board lot for buys (US whole shares,
#: CN 100-share lots); ``allow_short`` is the market's shortability.
DEFAULT_MARKETS: dict[str, dict[str, Any]] = {
    "us": {"exchange": "SMART", "currency": "USD", "lot_size": 1, "allow_short": True},
    "cn": {
        "sh_exchange": "SEHKNTL",   # Shanghai–HK Stock Connect
        "sz_exchange": "SEHKSZSE",  # Shenzhen–HK Stock Connect
        "currency": "CNH",
        "lot_size": 100,
        "allow_short": False,
    },
}


def _finite(x: float) -> bool:
    return x == x


def _merge_markets(override: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Merge a config override onto :data:`DEFAULT_MARKETS` (per-key)."""
    merged = {m: dict(v) for m, v in DEFAULT_MARKETS.items()}
    for market, values in (override or {}).items():
        merged.setdefault(str(market), {})
        merged[str(market)].update(values or {})
    return merged


class IBKRBroker:
    """
    Thin wrapper over an ``ib_async``/``ib_insync`` connection.

    Parameters
    ----------
    host, port, client_id : str | int
        IB Gateway / TWS connection (Gateway paper = 4002, TWS paper = 7497).
    connect_on_init : bool
        If True (default) connect immediately; otherwise lazy-connect on first
        use.  Tests inject a fake broker instead and never touch this path.
    ib : object | None
        Pre-built client (tests / dependency injection).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 17,
        connect_on_init: bool = True,
        ib: Any = None,
    ):
        self.host = str(host)
        self.port = int(port)
        self.client_id = int(client_id)
        self._ib = ib
        if self._ib is None and connect_on_init:
            self.connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @staticmethod
    def _ib_module() -> Any:
        """Import the IB client lazily (applications-only dependency)."""
        try:
            import ib_async  # type: ignore

            return ib_async
        except ImportError:
            pass
        try:
            import ib_insync  # type: ignore

            return ib_insync
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "IBKR venue requires 'ib_async' (or 'ib_insync'). "
                "Install the application extras: pip install -r requirements-app.txt"
            ) from exc

    def connect(self) -> Any:
        if self._ib is None:
            mod = self._ib_module()
            self._ib = mod.IB()
        if not self._ib.isConnected():
            self._ib.connect(self.host, self.port, clientId=self.client_id)
        return self._ib

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    @property
    def ib(self) -> Any:
        return self.connect() if self._ib is None else self._ib

    # ------------------------------------------------------------------
    # Contracts
    # ------------------------------------------------------------------

    def make_contract(self, spec: dict[str, Any]) -> Any:
        """Build an IB ``Contract`` from a resolved spec dict."""
        mod = self._ib_module()
        return mod.Stock(
            symbol=spec["symbol"],
            exchange=spec.get("exchange", "SMART"),
            currency=spec.get("currency", "USD"),
            primaryExchange=spec.get("primaryExchange") or "",
        )

    def qualify(self, contract: Any) -> Any:
        """Resolve contract details (no-op if the client has no qualifier)."""
        qualify = getattr(self.ib, "qualifyContracts", None)
        if qualify is not None:
            qualify(contract)
        return contract

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    def positions(self) -> dict[str, float]:
        """Currently held share counts keyed by IB contract symbol."""
        out: dict[str, float] = {}
        for pos in self.ib.positions():
            contract = getattr(pos, "contract", None)
            if contract is None:
                continue
            symbol = getattr(contract, "symbol", None)
            if symbol:
                out[str(symbol)] = out.get(str(symbol), 0.0) + float(pos.position)
        return out

    def cash_and_nav(self) -> tuple[float, float]:
        """``(cash, net_liquidation)`` from the account summary."""
        cash = nav = 0.0
        for row in self.ib.accountSummary():
            tag = str(getattr(row, "tag", ""))
            val = float(getattr(row, "value", 0.0) or 0.0)
            if tag == "TotalCashValue":
                cash = val
            elif tag == "NetLiquidation":
                nav = val
        return cash, nav

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place(
        self,
        contract: Any,
        action: str,
        quantity: float,
        order_type: str = "MOO",
        tif: str = "OPG",
        limit_price: float | None = None,
    ) -> Any:
        """Submit one order and return the IB ``Trade``."""
        mod = self._ib_module()
        otype = order_type.upper()
        if otype in ("LOO", "LMT") and limit_price is not None:
            order = mod.LimitOrder(action, abs(quantity), limit_price)
        else:
            order = mod.MarketOrder(action, abs(quantity))
        order.tif = "OPG" if otype in ("MOO", "LOO") else tif
        return self.ib.placeOrder(contract, order)


class IBKRExecutor(ExecutorBase):
    """
    IBKR executor over the shared book.

    Parameters
    ----------
    broker : IBKRBroker-like
        Injectable broker (must expose ``positions``, ``cash_and_nav``,
        ``make_contract``, ``qualify``, ``place``).
    guard : LiveGuard
        Submission gate.  Defaults to a dry-run guard (never submits).
    contract_map : dict[str, dict] | None
        Per-ticker contract overrides; other tickers fall back to the market
        defaults in :data:`DEFAULT_MARKETS` (+ ``symbol_replace`` for US).
    costs : CostModel | None
        Pre-trade commission estimate (reused from the backtest schedule).
    allow_fractional : bool
        If False (default), US quantities are whole shares (CN still uses lots).
    order_type, tif : str
        Order style (``MOO``/``LOO``/``MKT``/``LMT``) and time-in-force.
    allow_short : bool
        Global shortability kill-switch (ANDed with each market's own flag);
        per-market rules in PositionManager still apply.
    long_only_tickers : set[str] | None
        Names that must never end up net short.
    markets : dict[str, dict] | None
        Per-market overrides (``us``/``cn``) merged onto :data:`DEFAULT_MARKETS`.
    symbol_replace : dict[str, str] | None
        Forward string replacements for US IB symbols (e.g. ``{".": " "}`` turns
        ``BRK.B`` into IB's ``BRK B``); reversed when mapping positions back.
    """

    def __init__(
        self,
        broker: Any,
        guard: LiveGuard | None = None,
        contract_map: dict[str, dict[str, Any]] | None = None,
        costs: CostModel | None = None,
        allow_fractional: bool = False,
        order_type: str = "MOO",
        tif: str = "OPG",
        allow_short: bool = True,
        long_only_tickers: set[str] | None = None,
        markets: dict[str, dict[str, Any]] | None = None,
        symbol_replace: dict[str, str] | None = None,
        initial_cash: float = 1.0,
    ):
        super().__init__(initial_cash=initial_cash)
        self.broker = broker
        self.guard = guard or LiveGuard(dry_run=True)
        self.contract_map = {k: dict(v) for k, v in (contract_map or {}).items()}
        self.costs = costs
        self.allow_fractional = allow_fractional
        self.order_type = order_type
        self.tif = tif
        self.allow_short = allow_short
        self.long_only_tickers = long_only_tickers or set()
        self.markets = _merge_markets(markets)
        self.symbol_replace = symbol_replace or {}
        self.submitted: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        ibkr_cfg: Any,
        live_cfg: Any = None,
        broker: Any = None,
    ) -> "IBKRExecutor":
        get = ibkr_cfg.get
        guard = LiveGuard.from_config(
            live_cfg,
            live=bool(get("live", False)),
            dry_run=bool(get("dry_run", True)),
            max_notional=get("max_notional", None),
        )
        if broker is None:
            broker = IBKRBroker(
                host=str(get("host", "127.0.0.1")),
                port=int(get("port", 7497)),
                client_id=int(get("client_id", 17)),
                connect_on_init=False,
            )
        costs_cfg = get("costs", None)
        costs = build_cost_model(costs_cfg) if costs_cfg else None
        contract_map = {
            str(k): dict(v) for k, v in dict(get("contract_map", {}) or {}).items()
        }
        markets = {
            str(m): dict(v) for m, v in dict(get("markets", {}) or {}).items()
        }
        return cls(
            broker=broker,
            guard=guard,
            contract_map=contract_map,
            costs=costs,
            allow_fractional=bool(get("allow_fractional", False)),
            order_type=str(get("order_type", "MOO")),
            tif=str(get("tif", "OPG")),
            allow_short=bool(get("allow_short", True)),
            long_only_tickers=set(get("long_only_tickers", []) or []),
            markets=markets,
            symbol_replace=dict(get("symbol_replace", {".": " "}) or {}),
        )

    # ------------------------------------------------------------------
    # Market classification
    # ------------------------------------------------------------------

    @staticmethod
    def _market_of(ticker: str) -> str:
        if ticker.startswith("sh.") or ticker.startswith("sz."):
            return "cn"
        return "us"

    def _market_cfg(self, market: str) -> dict[str, Any]:
        return self.markets.get(market, {})

    def _allow_short(self, ticker: str) -> bool:
        """Shorting is allowed only if the global flag AND the market's flag do."""
        return bool(self.allow_short) and bool(
            self._market_cfg(self._market_of(ticker)).get("allow_short", True)
        )

    # ------------------------------------------------------------------
    # Ticker <-> contract
    # ------------------------------------------------------------------

    def _to_ib_symbol(self, ticker: str) -> str:
        symbol = ticker
        for src, dst in self.symbol_replace.items():
            symbol = symbol.replace(src, dst)
        return symbol

    def _to_repo_ticker(self, ib_symbol: str) -> str:
        # Explicit overrides always win.
        for ticker, spec in self.contract_map.items():
            if str(spec.get("symbol")) == ib_symbol:
                return ticker
        # 6-digit A-share codes: infer the exchange from the code range
        # (Shanghai 6xxxxx / 9xxxxx, Shenzhen 0xxxxx / 3xxxxx).
        if len(ib_symbol) == 6 and ib_symbol.isdigit():
            prefix = "sh." if ib_symbol[0] in ("6", "9") else "sz."
            return f"{prefix}{ib_symbol}"
        # US: reverse the symbol transform (BRK B → BRK.B).
        out = ib_symbol
        for src, dst in self.symbol_replace.items():
            out = out.replace(dst, src)
        return out

    def contract_spec(self, ticker: str) -> dict[str, Any]:
        """Resolve the full contract spec for a repo ticker."""
        market = self._market_of(ticker)
        cfg = self._market_cfg(market)
        if market == "cn":
            number = ticker.split(".", 1)[1] if "." in ticker else ticker
            exchange = (
                cfg.get("sh_exchange", "SEHKNTL")
                if ticker.startswith("sh.")
                else cfg.get("sz_exchange", "SEHKSZSE")
            )
            spec: dict[str, Any] = {
                "secType": "STK",
                "symbol": number,
                "exchange": exchange,
                "currency": cfg.get("currency", "CNH"),
                "primaryExchange": exchange,
            }
        else:
            spec = dict(DEFAULT_US_SPEC)
            spec["symbol"] = self._to_ib_symbol(ticker)
        spec.update(self.contract_map.get(ticker, {}))
        return spec

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
        """Revalue the book; held names absent from ``prices`` carry the last mark."""
        for t, p in prices.items():
            if _finite(p):
                self.last_prices[t] = float(p)
        nav = self.cash
        for t, sh in self.shares.items():
            p = self.last_prices.get(t)
            if p is not None:
                nav += sh * p
        self.last_nav = nav
        return nav

    def sync(self) -> float:
        """
        Overwrite the shared book from the broker (source of truth).

        Called on start and after fills; positions/NAV no longer drift from
        what IBKR actually holds.
        """
        positions = self.broker.positions() or {}
        self.shares = {
            self._to_repo_ticker(sym): float(qty)
            for sym, qty in positions.items()
            if abs(float(qty)) > 1e-12
        }
        cash, nav = self.broker.cash_and_nav()
        self.cash = float(cash)
        if nav:
            self.last_nav = float(nav)
        else:
            self.last_nav = self.cash + sum(
                sh * self.last_prices.get(t, 0.0) for t, sh in self.shares.items()
            )
        return self.last_nav

    def _round_shares(self, shares_delta: float, market: str = "us") -> float:
        """
        Round a share delta to a tradable quantity.

        Fractional allowed → verbatim.  Otherwise **buys** are floored to the
        market's board lot (CN = 100) and **sells** stay whole shares (A-share
        odd-lot sells are permitted); the long-only clamp caps sells at held.
        """
        if self.allow_fractional:
            return shares_delta
        lot = int(self._market_cfg(market).get("lot_size", 1) or 1)
        if shares_delta > 0 and lot > 1:
            return float(math.floor(shares_delta / lot) * lot)
        return float(math.trunc(shares_delta))

    def execute(self, orders: list[Order], prices: dict[str, float]) -> FillReport:
        """
        Plan fills at ``prices`` and submit them through the guard.

        In dry-run (the default) **nothing is submitted and the book is not
        mutated** — each order is reported in ``skipped`` with reason
        ``dry_run``.  When live, orders are submitted and the book updated
        optimistically; call :meth:`sync` after the open to reconcile.
        """
        nav = self.last_nav
        report = FillReport(nav_before=nav)
        self.bought_today = {}
        self.submitted = []

        for order in orders:
            ticker = order.ticker
            price = prices.get(ticker)
            if price is None or not _finite(price):
                report.skipped[ticker] = "no price"
                continue
            price = float(price)
            delta = order.delta_weight
            market = self._market_of(ticker)

            current = self.shares.get(ticker, 0.0)
            shares_delta = self._round_shares(delta * nav / price, market)
            new_shares = current + shares_delta

            # Long-only / shortability backstop (per-market + explicit names).
            long_only = ticker in self.long_only_tickers or not self._allow_short(ticker)
            if long_only and new_shares < -1e-12:
                shares_delta = -current
                new_shares = 0.0

            if abs(shares_delta) < 1e-12:
                report.skipped[ticker] = "rounds_to_zero"
                continue

            action = "BUY" if shares_delta > 0 else "SELL"
            notional = shares_delta * price

            reason = self.guard.block_reason(notional)
            if reason is not None:
                report.skipped[ticker] = reason
                continue

            spec = self.contract_spec(ticker)
            try:
                contract = self.broker.make_contract(spec)
                self.broker.qualify(contract)
                self.broker.place(
                    contract, action, abs(shares_delta),
                    order_type=self.order_type, tif=self.tif,
                    limit_price=price,
                )
            except Exception as exc:  # broker errors must not crash the cycle
                report.skipped[ticker] = f"broker_error:{type(exc).__name__}"
                continue

            commission = (
                self.costs.commission_nav(notional / nav, shares_delta, ticker)
                if self.costs is not None and nav
                else 0.0
            )
            self.shares[ticker] = new_shares if abs(new_shares) > 1e-12 else 0.0
            self.cash -= shares_delta * price + commission
            report.fills.append(
                Fill(
                    ticker=ticker, delta_weight=(notional / nav if nav else 0.0),
                    shares=shares_delta, price=price, value=notional,
                )
            )
            report.costs += commission
            report.turnover += abs(delta)
            self.submitted.append(
                {"ticker": ticker, "action": action, "shares": abs(shares_delta),
                 "order_type": self.order_type, "tif": self.tif, "price": price}
            )
            if shares_delta > 0:
                self.bought_today[ticker] = self.bought_today.get(ticker, 0.0) + shares_delta

        self.shares = {t: s for t, s in self.shares.items() if abs(s) > 1e-12}
        report.nav_after = self.last_nav
        return report
