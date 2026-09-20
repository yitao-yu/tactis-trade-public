"""
Venue 0 backtest runner (mirrors eval.py conventions).

Exercises the full application stack end-to-end on out-of-sample data:

    strategy (span-forecast rolling partitions)
        → PositionManager (diff + market risk filters)
        → BacktestExecutor (fills at the next open, shares/cash book, NAV)

Usage:
    python backtest.py backtest.checkpoint=outputs/.../custom_tactis_medium.pth
    python backtest.py backtest.mock_model=true backtest.end_date=2025-02-01
    python backtest.py backtest.n_partitions=2 backtest.allocator.return_upper=0.06

No-look-ahead by construction: the signal at close(t) uses only data through t;
orders are queued and filled at open(t+1); NAV is marked at close(t+1).
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from eval import _load_config
from metrics import _build_allocators
from model import build_model
from trading.base import MarketRule, PositionManager
from trading.data import BacktestData
from trading.samplers import ModelSpanSampler, RandomSpanSampler
from trading.strategy import RollingPartitionStrategy
from trading.costs import build_cost_model
from trading.venues.backtest import BacktestExecutor

CUSTOM_MERGED = Path("data/custom-data/raw/all_stocks.csv")


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def _max_drawdown(navs: np.ndarray) -> float:
    peak = np.maximum.accumulate(navs)
    return float(((navs - peak) / peak).min())


def _daily_stats(navs: np.ndarray) -> dict[str, Any]:
    navs = np.asarray(navs, dtype=np.float64)
    if navs.size < 2:
        return {}
    log_ret = np.diff(np.log(navs))
    std = log_ret.std()
    stats: dict[str, Any] = {
        "final_nav": float(navs[-1]),
        "total_return": float(navs[-1] / navs[0] - 1.0),
        "n_days": int(navs.size - 1),
        "max_drawdown": _max_drawdown(navs),
        "annualized_vol": float(std * np.sqrt(252)),
    }
    stats["daily_sharpe"] = float(log_ret.mean() / std) if std > 1e-12 else 0.0
    stats["annualized_sharpe"] = stats["daily_sharpe"] * np.sqrt(252)
    return stats


def _equal_weight_benchmark(
    data: BacktestData,
    rows: np.ndarray,
    allowed: np.ndarray | None = None,
) -> float:
    """Equal-weight, daily-rebalanced universe final NAV (no costs).

    ``allowed`` optionally restricts the basket (e.g. CN-only for a CN backtest).
    """
    nav = 1.0
    for r in rows:
        if r < 1:
            continue
        valid = data.valid[r] if allowed is None else (data.valid[r] & allowed)
        if valid.sum() == 0:
            continue
        nav *= 1.0 + float(data.ret[r][valid].mean())
    return nav


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _price_dict(prices: np.ndarray, row: int, tickers: list[str]) -> dict[str, float]:
    """Map a price row to {ticker: price} for finite entries only."""
    vals = prices[row]
    finite = np.isfinite(vals)
    return {t: float(vals[i]) for i, t in enumerate(tickers) if finite[i]}


def _carried_close_prices(data: BacktestData, executor: BacktestExecutor, row: int) -> dict[str, float]:
    """
    Close prices for held names, carrying forward the last known mark where a
    name has no price today (suspended / missing) so it is frozen, not dropped.
    """
    vals = data.close[row]
    out: dict[str, float] = {}
    for t in executor.shares:
        i = data.index_of(t)
        if np.isfinite(vals[i]):
            out[t] = float(vals[i])
        elif t in executor.last_prices:
            out[t] = executor.last_prices[t]
    return out


def _build_market_rules(cfg: Any) -> dict[str, MarketRule]:
    return {
        "us": MarketRule(allow_short=bool(cfg.backtest.allow_short), t_plus_1=False),
        "cn": MarketRule(allow_short=bool(cfg.backtest.get("cn_allow_short", False)),
                         t_plus_1=True, price_limit=0.10),
    }


def _build_cost_model(b: Any):
    """Build the transaction-cost model from config (see ``trading.costs``)."""
    return build_cost_model(b)


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------


def run_backtest(
    cfg: Any,
    model,
    data: BacktestData,
    device: torch.device,
    history_dir: Path | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    b = cfg.backtest
    hist_len = int(cfg.dataset.hist_len)
    pred_len = int(cfg.dataset.pred_len)
    n_partitions = int(b.n_partitions)
    if n_partitions > pred_len:
        raise ValueError(f"n_partitions={n_partitions} exceeds pred_len={pred_len}")

    market_filter = b.get("market_filter", None)
    allowed = data.market_mask(str(market_filter)) if market_filter else None
    long_only = bool(b.get("long_only", False))

    allocators = _build_allocators([b.allocator], float(b.var_alpha))
    allocator = allocators[b.allocator.name]
    # Ban shorts inside the allocator so its risk model matches the executed book.
    if long_only:
        # Long-only everywhere: per-name lower bound 0 (replaces the CN-only mask).
        if hasattr(allocator, "weight_lower"):
            allocator.weight_lower = 0.0
        if hasattr(allocator, "long_only_mask"):
            allocator.long_only_mask = None
    elif hasattr(allocator, "long_only_mask"):
        # Default: ban shorts on CN A-shares only.
        allocator.long_only_mask = data.market_mask("cn").tolist()

    if model is None:
        sampler = RandomSpanSampler(
            n_series=data.num_series, n_samples=int(b.n_samples),
            seed=int(b.seed), device=device,
        )
    else:
        sampler = ModelSpanSampler(
            model=model, data=data, hist_len=hist_len, pred_len=pred_len,
            n_samples=int(b.n_samples), device=device,
        )

    def _valid(row: int) -> np.ndarray:
        m = data.window_valid(row, hist_len) & data.tradeable(row)
        if allowed is not None:
            m = m & allowed
        return m

    strategy = RollingPartitionStrategy(
        allocator=allocator,
        tickers=data.tickers,
        n_partitions=n_partitions,
        share_reset=b.share_reset,
        sampler=sampler,
        valid_fn=_valid,
        min_valid_series=int(b.get("min_valid_series", 25)),
        max_gross_exposure=b.get("max_gross_exposure", 1.0),
        max_net_exposure=b.get("max_net_exposure", 1.0),
        prev_weight_clip=float(b.get("prev_weight_clip", b.allocator.get("weight_upper", 0.05))),
        history_dir=history_dir,
    )

    costs = _build_cost_model(b)
    cn_tickers = {t for t, is_cn in zip(data.tickers, data.market_mask("cn")) if is_cn}
    executor = BacktestExecutor(
        initial_cash=1.0, costs=costs, allow_short=bool(b.allow_short),
        long_only_tickers=cn_tickers,
    )
    manager = PositionManager(executor, market_rules=_build_market_rules(cfg))

    rows = data.trading_rows(b.start_date, b.end_date)
    rows = rows[rows >= max(hist_len, 1)]
    if max_rows is not None:
        rows = rows[:max_rows]
    if rows.size == 0:
        raise ValueError("no trading rows in the requested range")

    print(
        f"Backtest: {rows.size} days  {pd.Timestamp(data.dates[rows[0]]).date()} → "
        f"{pd.Timestamp(data.dates[rows[-1]]).date()}  |  {data.num_series} series  |  "
        f"{n_partitions} partitions  |  mock={model is None}"
    )
    print(
        f"  allocator={b.allocator.name}  turnover_coef="
        f"{b.allocator.get('turnover_coef', 0.0)}  commission_rate="
        f"{b.allocator.get('commission_rate', 3e-4)}  share_reset={b.share_reset}"
    )

    nav_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    pending: list[Any] = []
    total_costs = 0.0
    t0 = time.time()

    for r in rows:
        date = pd.Timestamp(data.dates[r])

        # 1. Morning fills (queued after close(r-1)) at the chosen execution price.
        if pending:
            if b.execution_price == "next_open":
                fill_prices = _price_dict(data.open, r, data.tickers)
            elif b.execution_price == "next_close":
                fill_prices = _price_dict(data.close, r, data.tickers)
            else:
                raise NotImplementedError(f"execution_price={b.execution_price}")
            report = executor.execute(pending, fill_prices)
            total_costs += report.costs
            strategy.charge_due_costs(report.costs)   # costs hit the rolling partition's cash
            for f in report.fills:
                trades.append(
                    dict(date=str(date.date()), ticker=f.ticker, delta_weight=f.delta_weight,
                         shares=f.shares, price=f.price, value=f.value)
                )
        else:
            report = None

        # 2. Evening mark at close(r); suspended names are carried at last mark.
        nav = executor.mark(_carried_close_prices(data, executor, r))

        # 3. Signal after close: reconcile the partition books to the actual
        #    account, then roll the due partition with a fresh sample.
        account_dollars = np.zeros(data.num_series, dtype=np.float64)
        for t, shares_qty in executor.shares.items():
            if t in executor.last_prices:
                account_dollars[data.index_of(t)] = shares_qty * executor.last_prices[t]

        # Reconcile books to the actual account, then snapshot the (exact-tiling)
        # partition claims before the due partition re-deploys.
        strategy.reconcile(account_dollars, executor.cash)
        part_columns = {}
        for p, part in enumerate(strategy.partitions):
            part_columns[f"partition_{p}_nav"] = part.nav
            part_columns[f"partition_{p}_gross"] = part.gross
            part_columns[f"partition_{p}_return"] = part.period_return

        target = strategy.on_close(
            str(date.date()), r, nav, account_dollars, executor.cash, device,
        )

        gross = executor.gross_exposure
        if gross > 1.5:
            print(f"  [WARN] leverage guard: gross_exposure={gross:.2f} on {date.date()}")

        # 4. Queue filtered orders for tomorrow's open; snapshot partition state.
        pending = manager.plan(target)
        strategy.snapshot_all(str(date.date()))
        tile_gap = float(sum(p.nav for p in strategy.partitions) - nav)

        nav_rows.append(
            dict(
                date=str(date.date()),
                weekday=str(date.day_name()),
                nav=nav,
                cash_weight=executor.cash_weight,
                cash_unengaged=1.0 - executor.gross_exposure,
                gross_exposure=executor.gross_exposure,
                net_exposure=executor.net_exposure,
                turnover=report.turnover if report else 0.0,
                costs=report.costs if report else 0.0,
                skip_reasons=";".join(f"{t}:{why}" for t, why in
                                      (report.skipped.items() if report else {})) or "",
                partition_tile_gap=tile_gap,
                **part_columns,
            )
        )

        if len(nav_rows) % 100 == 0:
            print(f"  day {len(nav_rows)}/{rows.size}  nav={nav:.4f}  ({time.time() - t0:.0f}s)")

    navs = np.array([x["nav"] for x in nav_rows], dtype=np.float64)
    stats = _daily_stats(navs)
    stats["total_costs"] = float(total_costs)
    stats["avg_turnover_per_day"] = float(np.mean([x["turnover"] for x in nav_rows]))
    stats["equal_weight_benchmark"] = float(_equal_weight_benchmark(data, rows, allowed))
    stats["start_date"] = str(pd.Timestamp(data.dates[rows[0]]).date())
    stats["end_date"] = str(pd.Timestamp(data.dates[rows[-1]]).date())
    cash = np.array([x["cash_weight"] for x in nav_rows])
    gross = np.array([x["gross_exposure"] for x in nav_rows])
    tile = np.abs(np.array([x["partition_tile_gap"] for x in nav_rows]))
    stats["neg_cash_days"] = int((cash < 0).sum())
    stats["over_gross_days"] = int((gross > 1.0 + 1e-9).sum())
    stats["max_gross_exposure"] = float(gross.max())
    stats["max_abs_tile_gap"] = float(tile.max())
    for p, part in enumerate(strategy.partitions):
        stats[f"partition_{p}/n_rolls"] = int(len(part.rolls))
        stats[f"partition_{p}/total_return"] = float(part.cumulative_return())
        stats[f"partition_{p}/final_nav"] = float(part.nav)
        stats[f"partition_{p}/cost_total"] = float(part.cost_total)
    if "SPY" in data.tickers:
        spy = data.tickers.index("SPY")
        seg = data.ret[rows[1:], spy]
        m = data.valid[rows[1:], spy]
        stats["spy_buy_hold"] = float(np.prod(1.0 + seg[m])) if m.any() else 0.0

    strategy.close()
    return {"nav": nav_rows, "trades": trades, "stats": stats}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _print_summary(stats: dict[str, float]) -> None:
    print("\n" + "=" * 60)
    print("Backtest Summary")
    print("=" * 60)
    if not stats:
        print("No stats (need >= 2 days).")
        return
    for key in ["final_nav", "total_return", "daily_sharpe", "annualized_sharpe",
                "annualized_vol", "max_drawdown", "avg_turnover_per_day", "total_costs",
                "equal_weight_benchmark", "spy_buy_hold", "days"]:
        if key in stats:
            print(f"  {key:<22}: {stats[key]:.6f}")


def main() -> None:
    cfg = _load_config()
    b = cfg.backtest
    sys.stdout.reconfigure(errors="replace")  # cp1252 consoles + unicode arrows

    seed = int(b.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mock = bool(b.mock_model)
    if not mock and (b.checkpoint is None or not Path(b.checkpoint).exists()):
        print("Error: backtest.checkpoint is required unless backtest.mock_model=true.")
        print("Example: python backtest.py backtest.checkpoint=outputs/.../custom_tactis_medium.pth")
        sys.exit(1)

    merged = Path(b.get("merged_path", CUSTOM_MERGED))
    if not merged.exists():
        print(f"Error: merged data not found at {merged}. Run the data download/merge first.")
        sys.exit(1)

    print(f"Loading data: {merged}")
    data = BacktestData.from_csv(merged, base_end_date=str(b.base_end_date))
    print(
        f"  {data.num_series} series, {data.n_rows} price rows "
        f"({pd.Timestamp(data.dates[0]).date()} → {pd.Timestamp(data.dates[-1]).date()})"
    )

    model = None
    if not mock:
        checkpoint_path = Path(b.checkpoint)
        model = build_model(cfg, data.num_series)
        model.to(device)
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        ckpt_stage = ckpt.get("stage", 2)
        if ckpt_stage == 2:
            model.initialize_stage2()
            model.to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        model.set_stage(ckpt_stage if ckpt_stage in (1, 2) else 2)
        print(f"Model stage: {ckpt_stage}  |  checkpoint: {checkpoint_path}")

    out_dir = Path(b.output_dir) if b.output_dir else \
        (Path(b.checkpoint).parent / "backtest" if b.checkpoint else Path("."))
    out_dir.mkdir(parents=True, exist_ok=True)
    history_dir = out_dir / "ledger"
    history_dir.mkdir(parents=True, exist_ok=True)

    result = run_backtest(cfg, model, data, device, history_dir=history_dir)
    navs, trades, stats = result["nav"], result["trades"], result["stats"]

    pd.DataFrame(navs).to_csv(out_dir / "nav.csv", index=False)
    if trades:
        pd.DataFrame(trades).to_csv(out_dir / "trades.csv", index=False)
    with open(out_dir / "backtest_metrics.json", "w") as f:
        json.dump(stats, f, indent=2, default=float)
    print(f"\nSaved to {out_dir}")
    _print_summary(stats)


if __name__ == "__main__":
    main()
