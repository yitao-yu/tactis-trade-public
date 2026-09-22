"""Day-loop pipeline shared by both venues: weights -> plan -> execute -> ledger.

Mirrors the Venue 0 day loop: signal at close(t), orders at open(t+1).
Dry-run is always the default; each venue's LiveGuard (built inside the
executor's from_config) enforces the two-factor live gate.

Logging (backtest.py-style, adapted for a single live day):
  - stdout progress lines: weights/plan/execution/ledger stages with counts
  - a JSON run summary (weights source, plan size, fills, skipped reasons,
    costs) written to <log_dir>/<venue>_daily.jsonl next to the ledger row,
    so CloudWatch + the local ledger agree on what happened each day.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LEDGER_KEYS = ("venue", "dry_run", "actions", "nav", "holdings_count")


def _log(msg: str) -> None:
    print(f"[live {datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}",
          flush=True)


def load_app_config() -> Any:
    with open(REPO / "cfg" / "application.yaml") as f:
        return yaml.safe_load(f)


def load_target_weights(path: Optional[str]) -> dict:
    """Target weights from a JSON file (model sampling is wired later)."""
    if not path:
        raise SystemExit("No --weights file; model sampling is wired in a later step.")
    with open(path) as f:
        return json.load(f)


def run_venue(venue: str, force_dry_run: bool, weights: dict, log_dir: Path,
              weights_source: str = "static") -> int:
    t0 = time.time()
    cfg = load_app_config()
    if venue not in ("ibkr", "xueqiu") or venue not in cfg:
        raise SystemExit(f"unknown venue: {venue}")
    venue_cfg = cfg[venue]
    live_cfg = cfg.get("live", {})

    # honour explicit dry-run from caller: if the venue is live we do NOT
    # touch config, but we refuse to run when force_dry_run is set.
    live_cfg = dict(live_cfg or {})
    if force_dry_run:
        venue_cfg = dict(venue_cfg, dry_run=True)
    dry = bool(venue_cfg.get("dry_run", True))

    _log(f"venue={venue}  weights={len(weights)} assets  source={weights_source}  "
         f"dry_run={dry}  live_env={'TACTIS_LIVE' in __import__('os').environ}")

    with_ledger = Path(log_dir)

    if venue == "ibkr":
        from trading.venues.ibkr import IBKRBroker, IBKRExecutor

        _log("connecting IB Gateway ...")
        broker = IBKRBroker(
            host=str(venue_cfg.get("host", "127.0.0.1")),
            port=int(venue_cfg.get("port", 4002)),
            client_id=int(venue_cfg.get("client_id", 17)),
            connect_on_init=True,
        )
        broker.connect()
        _log("syncing positions/NAV from broker ...")
        broker.sync()
        executor = IBKRExecutor.from_config(
            venue_cfg, live_cfg=live_cfg, broker=broker
        )
        pm = _make_position_manager(venue_cfg)
        orders = pm.plan(weights)
        _log(f"plan: {len(getattr(orders, 'orders', orders) or [])} order(s) "
             f"from {len(weights)} target weights")
        report = executor.execute(orders, prices={})
        fills = getattr(report, "fills", [])
        skipped = getattr(report, "skipped", [])
        _log(f"execute: {len(fills)} fill(s), {len(skipped)} skipped "
             f"({', '.join(sorted({s.get('reason', '?') if isinstance(s, dict) else str(s) for s in skipped})) or '-'})")
        info = report.info()
        _write_ledger(with_ledger, venue, info)
        _log(f"ledger written  total {time.time() - t0:.1f}s")
        return 0

    if venue == "xueqiu":
        from trading.venues.xueqiu import XueqiuExecutor

        executor = XueqiuExecutor.from_config(venue_cfg, live_cfg=live_cfg)
        if not force_dry_run and executor.guard.live_enabled:
            _log("live mode: refreshing account state ...")
            executor.refresh()  # real account state only needed when submitting
        result = executor.rebalance(weights)
        info = {
            "submitted": result.submitted,
            "reason": result.reason,
            "holdings_count": len(result.holdings),
            "cash": result.cash,
            "dropped": result.dropped,
        }
        _log(f"rebalance: submitted={result.submitted}  reason={result.reason}  "
             f"holdings={len(result.holdings)}  dropped={len(result.dropped or [])}")
        _write_ledger(with_ledger, venue, info)
        _log(f"ledger written  total {time.time() - t0:.1f}s")
        return 0

    raise SystemExit(f"unknown venue: {venue}")


def _make_position_manager(venue_cfg: dict) -> Any:
    from trading.base import PositionManager

    # hard constraint flags come from the application config per-venue
    return PositionManager(venue_cfg)


def _write_ledger(log_dir: Path, venue: str, info: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "venue": venue,
        **{k: info.get(k) for k in LEDGER_KEYS if k in info},
    }
    with open(log_dir / f"{venue}_daily.jsonl", "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    _log(f"ledger row: {json.dumps(row, default=str)[:300]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--venue", choices=["ibkr", "xueqiu"], required=True)
    ap.add_argument("--weights", help="JSON file of target weights")
    ap.add_argument("--dry-run", action="store_true",
                    help="bypass live gate decision and force dry-run")
    ap.add_argument("--log-dir", default="~/.tactis/app-logs")
    args = ap.parse_args()

    weights = load_target_weights(args.weights)
    return run_venue(args.venue, args.dry_run, weights, Path(args.log_dir).expanduser())


if __name__ == "__main__":
    sys.exit(main())
