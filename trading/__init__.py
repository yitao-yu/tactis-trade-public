"""
Application layer — shared venue contract, strategy, and venue executors.

- ``base.py``      : ExecutorBase, PositionManager, FillReport, TargetWeights
- ``data.py``      : BacktestData (date-aware backtest price/return data)
- ``strategy.py``  : RollingPartitionStrategy (span-forecast allocation)
- ``costs.py``     : CostModel / MarketCost / build_cost_model (shared)
- ``live.py``      : LiveGuard (dry-run default, live gates, kill-switch)
- ``venues/``      : venue executors (backtest, IBKR, Xueqiu)
"""

from .base import (
    ExecutorBase,
    FillReport,
    MarketRule,
    Order,
    PositionManager,
    TargetWeights,
)
from .live import LiveGuard, load_application_config

__all__ = [
    "ExecutorBase",
    "FillReport",
    "LiveGuard",
    "MarketRule",
    "Order",
    "PositionManager",
    "TargetWeights",
    "load_application_config",
]
