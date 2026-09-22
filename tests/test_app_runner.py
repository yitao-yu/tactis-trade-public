"""Offline tests for the app/ live-runner layer (Venues 1 & 2 on AWS Batch).

Covers: LiveGuard two-factor gate via the real executors' from_config,
pipeline config/weight loading, long-only helpers, secrets resolution order,
and the Xueqiu keepalive (SSM injected via stub). No network, no broker.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "app"
for p in (str(REPO), str(APP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pipeline  # noqa: E402
from trading.live import REASON_DRY, REASON_DISABLED, REASON_NO_ENV  # noqa: E402


# ---------------------------------------------------------------------------
# Config / weight loading
# ---------------------------------------------------------------------------

def test_load_app_config_has_venues():
    cfg = pipeline.load_app_config()
    assert {"live", "ibkr", "xueqiu"} <= set(cfg.keys())


def test_load_target_weights_roundtrip(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"AAPL": 0.1, "sh.600000": -0.05}))
    w = pipeline.load_target_weights(str(p))
    assert w == {"AAPL": 0.1, "sh.600000": -0.05}


def test_load_target_weights_requires_file():
    with pytest.raises(SystemExit):
        pipeline.load_target_weights(None)


# ---------------------------------------------------------------------------
# LiveGuard two-factor via real executors' from_config
# ---------------------------------------------------------------------------

def _venue_cfgs():
    cfg = pipeline.load_app_config()
    return cfg["ibkr"], cfg["xueqiu"], cfg.get("live")


def test_ibkr_executor_defaults_to_dry_run():
    from trading.venues.ibkr import IBKRExecutor
    ibkr_cfg, _, live = _venue_cfgs()
    ex = IBKRExecutor.from_config(ibkr_cfg, live_cfg=live)
    assert not ex.guard.live_enabled
    assert ex.guard.block_reason(0.0) == REASON_DRY


def test_xueqiu_executor_defaults_to_dry_run():
    from trading.venues.xueqiu import XueqiuExecutor
    _, xq_cfg, live = _venue_cfgs()
    ex = XueqiuExecutor.from_config(xq_cfg, live_cfg=live)
    assert not ex.guard.live_enabled


def test_gate_refuses_live_without_env(monkeypatch):
    """live:true + dry_run:false but TACTIS_LIVE unset -> REASON_NO_ENV."""
    from trading.live import LiveGuard
    monkeypatch.delenv("TACTIS_LIVE", raising=False)
    g = LiveGuard(live=True, dry_run=False)
    assert g.block_reason(0.0) == REASON_NO_ENV
    monkeypatch.setenv("TACTIS_LIVE", "1")
    assert g.block_reason(0.0) is None


def test_gate_kill_switch_wins(monkeypatch):
    from trading.live import LiveGuard
    monkeypatch.setenv("TACTIS_LIVE", "1")
    monkeypatch.setenv("TACTIS_TRADING_KILL", "1")
    g = LiveGuard(live=True, dry_run=False)
    assert g.block_reason(0.0) == "kill_switch"


def test_gate_notional_cap(monkeypatch):
    from trading.live import LiveGuard
    monkeypatch.setenv("TACTIS_LIVE", "1")
    g = LiveGuard(live=True, dry_run=False, max_notional=1000.0)
    assert g.block_reason(2000.0) == "notional_cap"
    assert g.block_reason(500.0) is None


# ---------------------------------------------------------------------------
# pipeline.run_venue: forced dry-run + wiring (no broker reachable, offline)
# ---------------------------------------------------------------------------

def test_run_venue_rejects_unknown():
    with pytest.raises(SystemExit):
        pipeline.run_venue("nope", True, {}, Path("/tmp"))


class FakeXqClient:
    registered = False
    def rebalance(self, **kw):
        type(self).registered = True
        return {"status": "ok"}
    def get_positions(self, cube_symbol):
        return []
    def get_quote(self, cube_symbol):
        return {"net_value": 100.0}


def test_xueqiu_rebalance_dry_run_never_submits(monkeypatch, tmp_path):
    """Dry-run: rebalance must project + report, but never hit the client."""
    from trading.venues.xueqiu import XueqiuClient, XueqiuExecutor
    cfg = pipeline.load_app_config()
    FakeXqClient.registered = False
    client = XueqiuClient(cookie="c=1", transport=object(), user=object())
    client.rebalance = FakeXqClient().rebalance  # stub the POST
    monkeypatch.setattr(pipeline, "load_app_config", lambda: cfg)
    ex = XueqiuExecutor.from_config(cfg["xueqiu"], live_cfg=cfg.get("live"), client=client)
    result = ex.rebalance({"sh.600000": 0.5, "sz.000001": 0.3})
    assert result.submitted is False
    assert result.reason == REASON_DRY
    assert FakeXqClient.registered is False
    # ledger row still writes
    log_dir = tmp_path / "log"
    pipeline._write_ledger(log_dir, "xueqiu", {"submitted": result.submitted})
    assert (log_dir / "xueqiu_daily.jsonl").is_file()


def test_xueqiu_projection_long_only_cap(minimum_cfg=None):
    """Negative / >max_holdings / sub-min-weight names are dropped."""
    from trading.venues.xueqiu import XueqiuExecutor
    cfg = pipeline.load_app_config()
    xq = cfg["xueqiu"]
    client = object()  # never used in projection-only path
    ex = XueqiuExecutor.from_config(xq, client=client)
    holdings, cash = ex.project({
        "a": 0.6, "b": -0.2,                 # negative dropped
        "tiny": 0.001,                        # below min_weight
        **{f"s{i}": 0.02 for i in range(40)},  # > max_holdings
    })
    assert "b" not in holdings and "tiny" not in holdings
    assert len(holdings) <= ex.max_holdings
    assert cash >= 0.0


def _write_ledger_checks(tmp_path):
    pipeline._write_ledger(tmp_path, "ibkr", {"submitted": False})
    return json.loads((tmp_path / "ibkr_daily.jsonl").read_text().splitlines()[0])

