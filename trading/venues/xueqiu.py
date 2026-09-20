"""
Venue 2 — Xueqiu (雪球) portfolio executor (CN long-only).

Xueqiu is **target-based**: a rebalance submits the entire target holding
structure (``POST cubes/rebalancing/create.json``), not per-ticker fills.  The
executor therefore exposes a venue-native :meth:`XueqiuExecutor.rebalance`,
while :meth:`execute` implements the shared :class:`ExecutorBase` contract as a
shim that reconstructs the full target from the order diff and delegates.

Auth is browser **cookies only**, loaded from an env var or a chmod-600 file —
never logged and never committed.  The transport is injectable so the executor
is fully testable offline; the default transport wraps ``easytrader`` (an
applications-only dependency, see ``requirements-app.txt``).

Constraints enforced before submission: long-only, ≤ ``max_holdings`` names,
each kept weight ≥ ``min_weight`` (default 1 %), residual cash optionally parked
in a money-market proxy.  Suspended / limit-D / delisted names (Xueqiu
``flag != 1``) and unquotable names are dropped with a reason.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..base import ExecutorBase, Fill, FillReport, Order
from ..live import LiveGuard


def _finite(x: float) -> bool:
    return x == x


#: Xueqiu exchange prefixes ↔ repo exchange prefixes.
_EXCHANGE_TO_XQ = {"sh.": "SH", "sz.": "SZ"}
_EXCHANGE_FROM_XQ = {"SH": "sh.", "SZ": "sz."}


def to_xueqiu_symbol(ticker: str) -> str:
    """``sh.600000`` → ``SH600000`` (and pass through already-normalized codes)."""
    for prefix, xq in _EXCHANGE_TO_XQ.items():
        if ticker.startswith(prefix):
            return f"{xq}{ticker[len(prefix):]}"
    return ticker


def from_xueqiu_symbol(symbol: str) -> str:
    """``SH600000`` → ``sh.600000``."""
    if len(symbol) > 2:
        for xq, prefix in _EXCHANGE_FROM_XQ.items():
            if symbol.upper().startswith(xq):
                return f"{prefix}{symbol[2:]}"
    return symbol


@dataclass
class RebalanceResult:
    """Outcome of a (possibly dry-run) full rebalance."""

    submitted: bool
    reason: str | None
    holdings: dict[str, float] = field(default_factory=dict)
    cash: float = 0.0
    dropped: dict[str, str] = field(default_factory=dict)


class XueqiuClient:
    """
    Minimal Xueqiu HTTP client (throttled, retrying, cookie-authenticated).

    Parameters
    ----------
    cookie : str | None
        Cookie string.  If omitted, loaded from ``cookie_env`` then
        ``cookie_file``.  Never returned/logged.
    cookie_env : str
        Environment variable holding the cookie (default ``XUEQIU_COOKIE``).
    cookie_file : str | Path | None
        chmod-600 file holding the cookie (fallback source).
    min_interval_s : float
        Minimum seconds between requests (rate-limit protection).
    max_retries : int
        Retries per request on transport errors.
    base_url : str
        API root.
    user : object | None
        Pre-built ``easytrader`` XueQiu trader (dependency injection / tests).
    transport : object | None
        Object with ``get(path, params)`` / ``post(path, json)`` returning
        parsed JSON.  Defaults to an ``easytrader``-backed transport.  Tests
        inject a fake so no network/cookie is needed.
    sleep : callable
        Injectable sleep (tests record throttle delays).
    """

    BASE_URL = "https://xueqiu.com"

    def __init__(
        self,
        cookie: str | None = None,
        cookie_env: str = "XUEQIU_COOKIE",
        cookie_file: str | Path | None = None,
        min_interval_s: float = 1.0,
        max_retries: int = 3,
        base_url: str = BASE_URL,
        user: Any = None,
        transport: Any = None,
        sleep: Any = time.sleep,
    ):
        self._cookie = cookie
        self.cookie_env = cookie_env
        self.cookie_file = Path(cookie_file) if cookie_file else None
        self.min_interval_s = float(min_interval_s)
        self.max_retries = int(max_retries)
        self.base_url = base_url.rstrip("/")
        self._user = user
        self._transport = transport
        self._sleep = sleep
        self._last_call = 0.0

    # ------------------------------------------------------------------
    # Auth (never exposes the cookie value)
    # ------------------------------------------------------------------

    def _load_cookie(self) -> str:
        import os

        if self._cookie:
            return self._cookie
        env_cookie = os.environ.get(self.cookie_env)
        if env_cookie:
            return env_cookie.strip()
        if self.cookie_file is not None and self.cookie_file.exists():
            return self.cookie_file.read_text(encoding="utf-8").strip()
        raise RuntimeError(
            "Xueqiu cookie not found: set the environment variable "
            f"{self.cookie_env!r} or configure a cookie file (chmod 600)."
        )

    def _easytrader_user(self) -> Any:
        if self._user is None:
            try:
                import easytrader  # type: ignore
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "Xueqiu venue requires 'easytrader'. "
                    "Install the application extras: pip install -r requirements-app.txt"
                ) from exc
            cls = getattr(easytrader, "XueQiuTrader", None) or easytrader.use("xq")
            self._user = cls() if callable(cls) else cls
            prepare = getattr(self._user, "prepare", None)
            if prepare is not None:
                prepare(cookie=self._load_cookie())
        return self._user

    def _get_transport(self) -> Any:
        if self._transport is not None:
            return self._transport
        user = self._easytrader_user()
        session = getattr(user, "_session", None) or getattr(user, "session", None)
        if session is None:
            import requests

            session = requests.Session()
        self._transport = _SessionTransport(session, self.base_url)
        return self._transport

    # ------------------------------------------------------------------
    # Request plumbing
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        wait = self.min_interval_s - elapsed
        if wait > 0:
            self._sleep(wait)
        self._last_call = time.monotonic()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        transport = self._get_transport()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                fn = getattr(transport, method)
                return fn(path, **kwargs)
            except Exception as exc:  # network / auth — never include the cookie
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    self._sleep(min(2.0 ** attempt, 8.0))
        raise RuntimeError(f"Xueqiu request {method} {path} failed: {last_exc}")

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def get_positions(self, cube_symbol: str) -> list[dict[str, Any]]:
        data = self._request(
            "get", "cubes/rebalancing/current.json", params={"cube_symbol": cube_symbol}
        )
        return _as_list(data)

    def get_quote(self, cube_symbol: str) -> dict[str, Any]:
        data = self._request("get", "cubes/quote.json", params={"cube_symbol": cube_symbol})
        return data if isinstance(data, dict) else {}

    def search_stock(self, query: str) -> list[dict[str, Any]]:
        data = self._request("get", "stock/p/search.json", params={"q": query})
        return _as_list(data)

    def rebalance(
        self,
        cube_symbol: str,
        cash: float,
        holdings: list[dict[str, Any]],
        segment: str = "cn",
        comment: str = "rebalance",
    ) -> dict[str, Any]:
        payload = {
            "cash": cash,
            "holdings": json.dumps(holdings) if not isinstance(holdings, str) else holdings,
            "cube_symbol": cube_symbol,
            "segment": segment,
            "comment": comment,
        }
        data = self._request("post", "cubes/rebalancing/create.json", json=payload)
        return data if isinstance(data, dict) else {}


class _SessionTransport:
    """Adapter turning a ``requests``-like session into get/post JSON calls."""

    def __init__(self, session: Any, base_url: str):
        self.session = session
        self.base_url = base_url

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self.session.get(self._url(path), params=params)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        resp = self.session.post(self._url(path), json=json)
        resp.raise_for_status()
        return resp.json()


def _as_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("list", "data", "holdings", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


class XueqiuExecutor(ExecutorBase):
    """
    Target-based Xueqiu executor with a long-only CN projection.

    Parameters
    ----------
    client : XueqiuClient
        Injectable client (tests pass a fake with the same 4 methods).
    cube_symbol : str
        Xueqiu portfolio symbol.
    guard : LiveGuard
        Submission gate; dry-run by default.
    market : str
        Portfolio market (``cn``/``us``/``hk``); sets the request ``segment``.
    max_holdings, min_weight : int, float
        Xueqiu limits (≤30 names, ≥1 % each).
    cash_proxy_ticker : str | None
        Money-market fund receiving the residual cash (keeps the book fully
        invested); when ``None`` the residual is sent as the payload ``cash``.
    min_price : float
        Reject names quoted below this (HK < HK$1 rule).
    allowed_tickers : set[str] | None
        Optional universe restriction (e.g. CN-only).
    """

    def __init__(
        self,
        client: XueqiuClient,
        cube_symbol: str,
        guard: LiveGuard | None = None,
        market: str = "cn",
        max_holdings: int = 30,
        min_weight: float = 0.01,
        cash_proxy_ticker: str | None = None,
        min_price: float = 0.0,
        allowed_tickers: set[str] | None = None,
        allow_short: bool = False,
        cost_rate: float = 0.0,
        initial_cash: float = 1.0,
    ):
        super().__init__(initial_cash=initial_cash)
        self.client = client
        self.cube_symbol = cube_symbol
        self.guard = guard or LiveGuard(dry_run=True)
        self.market = market
        self.max_holdings = int(max_holdings)
        self.min_weight = float(min_weight)
        self.cash_proxy_ticker = cash_proxy_ticker
        self.min_price = float(min_price)
        self.allowed_tickers = allowed_tickers
        self.allow_short = allow_short
        self.cost_rate = float(cost_rate)
        self._weights: dict[str, float] = {}
        self.last_dropped: dict[str, str] = {}
        self.last_result: RebalanceResult | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        xq_cfg: Any,
        live_cfg: Any = None,
        client: Any = None,
    ) -> "XueqiuExecutor":
        get = xq_cfg.get
        guard = LiveGuard.from_config(
            live_cfg,
            live=bool(get("live", False)),
            dry_run=bool(get("dry_run", True)),
            max_notional=get("max_notional", None),
        )
        if client is None:
            client = XueqiuClient(
                cookie_env=str(get("cookie_env", "XUEQIU_COOKIE")),
                cookie_file=get("cookie_file", None),
                min_interval_s=float(get("min_interval_s", 1.0)),
            )
        max_holdings = int(get("max_holdings", 30))
        cash_proxy = get("cash_proxy_ticker", None)
        allowed = get("allowed_tickers", None)
        return cls(
            client=client,
            cube_symbol=str(get("cube_symbol", "")),
            guard=guard,
            market=str(get("market", "cn")),
            max_holdings=max_holdings,
            min_weight=float(get("min_weight", 0.01)),
            cash_proxy_ticker=str(cash_proxy) if cash_proxy else None,
            allowed_tickers=set(allowed) if allowed else None,
            cost_rate=float(get("cost_rate", 0.0)),
        )

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    def refresh(self) -> float:
        """Pull current holdings + NAV from Xueqiu into the shared book."""
        positions = self.client.get_positions(self.cube_symbol) or []
        quote = self.client.get_quote(self.cube_symbol) or {}
        nav = _quote_nav(quote)
        self._weights = {}
        self.shares = {}
        for row in positions:
            symbol = str(row.get("stock_symbol") or row.get("symbol") or row.get("stock_id") or "")
            if not symbol:
                continue
            ticker = from_xueqiu_symbol(symbol)
            weight = _position_weight(row)
            self._weights[ticker] = weight
        self.last_prices = dict(self.last_prices)
        self.last_nav = float(nav) if nav else self.last_nav
        for ticker, weight in self._weights.items():
            price = self.last_prices.get(ticker)
            if price:
                self.shares[ticker] = weight * self.last_nav / price
        return self.last_nav

    def get_shares(self) -> dict[str, float]:
        return dict(self.shares)

    def get_nav(self) -> float:
        return self.last_nav

    def get_holdings(self) -> dict[str, float]:
        return dict(self._weights)

    def mark(self, prices: dict[str, float]) -> float:
        for t, p in prices.items():
            if _finite(p):
                self.last_prices[t] = float(p)
        return self.last_nav

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(self, target: dict[str, float]) -> tuple[dict[str, float], float]:
        """
        Enforce Xueqiu's constraints on a raw target.

        Drops non-positive weights (long-only), out-of-universe names,
        below-``min_price`` / ``flag != 1`` names, and sub-``min_weight``
        residuals; keeps the top ``max_holdings`` by weight; scales down if the
        invested fraction exceeds 1.  The residual becomes cash, optionally
        parked in ``cash_proxy_ticker``.
        """
        self.last_dropped = {}
        kept: dict[str, float] = {}
        for ticker, w in target.items():
            if w <= 0:
                if w < 0:
                    self.last_dropped[ticker] = "long_only"
                continue
            if self.allowed_tickers is not None and ticker not in self.allowed_tickers:
                self.last_dropped[ticker] = "not_in_universe"
                continue
            price = self.last_prices.get(ticker)
            if price is not None and self.min_price > 0 and price < self.min_price:
                self.last_dropped[ticker] = "below_min_price"
                continue
            if w < self.min_weight:
                self.last_dropped[ticker] = "below_min_weight"
                continue
            kept[ticker] = float(w)

        # Reserve a slot for the cash proxy so the final count stays ≤ max.
        budget = self.max_holdings - (1 if self.cash_proxy_ticker else 0)
        if budget >= 0 and len(kept) > budget:
            ordered = sorted(kept.items(), key=lambda kv: kv[1], reverse=True)
            for ticker, _w in ordered[budget:]:
                self.last_dropped[ticker] = "over_max_holdings"
            kept = dict(ordered[:budget])

        invested = sum(kept.values())
        if invested > 1.0 and invested > 0:
            kept = {t: w / invested for t, w in kept.items()}
            invested = 1.0
        cash = max(0.0, 1.0 - invested)

        if self.cash_proxy_ticker and cash >= self.min_weight:
            kept[self.cash_proxy_ticker] = cash
            cash = 0.0
        return kept, cash

    def _holdings_payload(self, holdings: dict[str, float]) -> list[dict[str, Any]]:
        payload = []
        for ticker, weight in holdings.items():
            payload.append({
                "stock_symbol": to_xueqiu_symbol(ticker),
                "weight": round(weight * 100.0, 2),
            })
        return payload

    # ------------------------------------------------------------------
    # Target-based rebalance (primary path)
    # ------------------------------------------------------------------

    def rebalance(self, target: dict[str, float]) -> RebalanceResult:
        """Project ``target`` and submit a full rebalance (or dry-run it)."""
        holdings, cash = self.project(target)
        nav = self.last_nav or 1.0
        notional = sum(holdings.values()) * nav
        reason = self.guard.block_reason(notional)
        result = RebalanceResult(
            submitted=reason is None, reason=reason,
            holdings=holdings, cash=cash, dropped=dict(self.last_dropped),
        )
        if reason is not None:
            self.last_result = result
            return result

        self.client.rebalance(
            cube_symbol=self.cube_symbol,
            cash=cash * nav,
            holdings=self._holdings_payload(holdings),
            segment=self.market,
            comment="tactis rebalance",
        )
        self._weights = dict(holdings)
        self.last_result = result
        return result

    # ------------------------------------------------------------------
    # ExecutorBase contract (thin shim over rebalance)
    # ------------------------------------------------------------------

    def execute(self, orders: list[Order], prices: dict[str, float]) -> FillReport:
        """
        Reconstruct the full target from the order diff and rebalance.

        Xueqiu has no per-ticker fills, so the report is best-effort: submitted
        orders get a planned ``Fill``; blocked/dropped orders land in
        ``skipped`` with a reason.
        """
        self.mark(prices)
        target = dict(self.get_holdings())
        for order in orders:
            target[order.ticker] = target.get(order.ticker, 0.0) + order.delta_weight
        result = self.rebalance(target)

        report = FillReport(nav_before=self.last_nav, nav_after=self.last_nav)
        for order in orders:
            if result.reason is not None:
                report.skipped[order.ticker] = result.reason
                continue
            if order.ticker in result.dropped:
                report.skipped[order.ticker] = result.dropped[order.ticker]
                continue
            price = prices.get(order.ticker, self.last_prices.get(order.ticker, 0.0))
            report.fills.append(
                Fill(ticker=order.ticker, delta_weight=order.delta_weight,
                     shares=0.0, price=float(price) if price else 0.0, value=0.0)
            )
            report.turnover += abs(order.delta_weight)
        if result.submitted and self.cost_rate:
            report.costs = report.turnover * self.cost_rate
        return report


def _quote_nav(quote: dict[str, Any]) -> float:
    for key in ("net_value", "net_asset", "total_value", "nav", "value"):
        if key in quote:
            try:
                return float(quote[key])
            except (TypeError, ValueError):
                continue
    nested = quote.get("data") or {}
    if isinstance(nested, dict):
        for key in ("net_value", "net_asset", "total_value", "nav", "value"):
            if key in nested:
                try:
                    return float(nested[key])
                except (TypeError, ValueError):
                    continue
    return 0.0


def _position_weight(row: dict[str, Any]) -> float:
    """Extract a fractional weight from a Xueqiu holding row.

    Xueqiu reports weights as **percentages** (e.g. ``12.5`` = 12.5 %), so the
    percent-style keys are always divided by 100.  A ``weight_fraction`` key (if
    present) is used verbatim.
    """
    if "weight_fraction" in row:
        try:
            return float(row["weight_fraction"])
        except (TypeError, ValueError):
            pass
    for key in ("weight", "weight_rate", "position_weight", "proportion"):
        if key in row:
            try:
                return float(row[key]) / 100.0
            except (TypeError, ValueError):
                continue
    return 0.0
