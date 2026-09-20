"""
Portfolio allocation package.

Registry of named allocator constructors for use in config-driven
metric computation.
"""

from .base import BaseAllocator
from .constraints import PortfolioConstraints
from .mv_long import MeanVarianceLongAllocator
from .mv_neutral import MeanVarianceNeutralCVaRAllocator
from .proportional import ProportionalAllocator
from .var_aware import VaRAwareAllocator
from .var_deployment import VarDeploymentAllocator
from .var_optimizer import VaROptimizer


ALLOCATOR_REGISTRY: dict[str, type[BaseAllocator]] = {
    "proportional": ProportionalAllocator,
    "var_aware": VaRAwareAllocator,
    "var_optimizer": VaROptimizer,
    "mv_neutral_cvar": MeanVarianceNeutralCVaRAllocator,
    "mv_long": MeanVarianceLongAllocator,
    "var_deployment": VarDeploymentAllocator,
}


__all__ = [
    "BaseAllocator",
    "PortfolioConstraints",
    "ProportionalAllocator",
    "VaRAwareAllocator",
    "VaROptimizer",
    "MeanVarianceNeutralCVaRAllocator",
    "MeanVarianceLongAllocator",
    "VarDeploymentAllocator",
    "ALLOCATOR_REGISTRY",
]
