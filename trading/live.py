"""
Live-trading guardrails shared by every live venue.

Local development is the default and is **always dry-run**: submitting a real
order requires *both* ``live: true`` in the venue/application config *and* the
environment variable named by ``require_env`` set to ``1`` (default
``TACTIS_LIVE``).  Two independent factors means a stray env var on a laptop
cannot trade, and a config typo alone cannot trade.

A kill-switch (env ``TACTIS_TRADING_KILL=1`` or a sentinel file) and a
per-venue notional cap are enforced on top, and are the *only* things the
executors must consult before dispatching an order.

The gate is deliberately pure (no I/O beyond an env lookup and an ``exists``
check) so tests can drive every branch offline.
"""

import os
from pathlib import Path
from typing import Any

#: Reason codes surfaced in ``FillReport.skipped`` and the trade log.
REASON_KILL = "kill_switch"
REASON_DRY = "dry_run"
REASON_DISABLED = "live_disabled"
REASON_NO_ENV = "no_live_env"
REASON_NOTIONAL = "notional_cap"


def load_application_config(path: str | Path = "cfg/application.yaml") -> Any:
    """
    Load ``cfg/application.yaml`` (live: ibkr: xueqiu:) with OmegaConf.

    Kept separate from ``cfg/config.yaml`` so training/backtest configs stay
    untouched by live-trading settings.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    return cfg


class LiveGuard:
    """
    Decide whether an order may leave the machine.

    Parameters
    ----------
    live : bool
        Venue-level live flag (config).  ``False`` (default) forces dry-run.
    dry_run : bool
        Explicit dry-run flag (config).  ``True`` (default) forces dry-run even
        when ``live`` is set — clear, redundant intent.
    require_env : str
        Environment variable that must equal ``"1"`` to allow live submission.
    max_notional : float | None
        Absolute per-order notional cap (account currency).  ``None`` disables.
    kill_switch_env : str
        Env var; when set to ``"1"`` all submissions are blocked.
    kill_switch_file : str | Path | None
        Sentinel file; when it exists all submissions are blocked.
    env : mapping | None
        Injectable environment (defaults to ``os.environ``) — tests only.
    """

    def __init__(
        self,
        live: bool = False,
        dry_run: bool = True,
        require_env: str = "TACTIS_LIVE",
        max_notional: float | None = None,
        kill_switch_env: str = "TACTIS_TRADING_KILL",
        kill_switch_file: str | Path | None = None,
        env: Any = None,
    ):
        self.live = bool(live)
        self.dry_run = bool(dry_run)
        self.require_env = str(require_env)
        self.max_notional = float(max_notional) if max_notional is not None else None
        self.kill_switch_env = str(kill_switch_env)
        self.kill_switch_file = Path(kill_switch_file) if kill_switch_file else None
        self._env = env

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        live_cfg: Any = None,
        *,
        live: bool | None = None,
        dry_run: bool | None = None,
        max_notional: float | None = None,
        env: Any = None,
    ) -> "LiveGuard":
        """
        Build from the shared ``live:`` config node (``application.yaml``).

        Per-venue values (``live``, ``dry_run``, ``max_notional``) override the
        shared node so a venue can be stricter without editing the shared block.
        """
        live_cfg = live_cfg or {}
        get = live_cfg.get
        return cls(
            live=bool(live) if live is not None else bool(get("live", False)),
            dry_run=bool(dry_run) if dry_run is not None else bool(get("dry_run", True)),
            require_env=str(get("require_env", "TACTIS_LIVE")),
            max_notional=max_notional if max_notional is not None
            else get("max_notional_default", None),
            kill_switch_env=str(get("kill_switch_env", "TACTIS_TRADING_KILL")),
            kill_switch_file=get("kill_switch_file", None),
            env=env,
        )

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    def _environ(self) -> Any:
        return os.environ if self._env is None else self._env

    def _kill_active(self) -> bool:
        if str(self._environ().get(self.kill_switch_env, "0")) == "1":
            return True
        return self.kill_switch_file is not None and self.kill_switch_file.exists()

    @property
    def live_enabled(self) -> bool:
        """True only when every live precondition is satisfied."""
        return self.block_reason(0.0) is None

    def block_reason(self, notional: float = 0.0) -> str | None:
        """
        ``None`` when the order may be submitted live, else a reason code.

        Precedence: kill-switch → dry-run → live disabled → missing env →
        notional cap.  In dry-run every order reports :data:`REASON_DRY`.
        """
        if self._kill_active():
            return REASON_KILL
        if self.dry_run:
            return REASON_DRY
        if not self.live:
            return REASON_DISABLED
        if str(self._environ().get(self.require_env, "0")) != "1":
            return REASON_NO_ENV
        if self.max_notional is not None and abs(float(notional)) > self.max_notional:
            return REASON_NOTIONAL
        return None

    def allows(self, notional: float = 0.0) -> bool:
        return self.block_reason(notional) is None
