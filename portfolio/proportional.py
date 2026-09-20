"""
Proportional allocator — weights proportional to predicted mean return.

w_i  ∝  max(0, E[r_i])

This is the simplest long-only baseline: take the expected return from the
Monte Carlo samples, threshold at zero, and normalise.
"""

import torch

from .base import BaseAllocator


class ProportionalAllocator(BaseAllocator):
    """Weights proportional to positive predicted mean return."""

    def compute_raw_weights(
        self,
        samples: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return samples.mean(dim=-1).clamp(min=0)
