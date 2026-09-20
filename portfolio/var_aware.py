"""
VaR-aware allocator — filter by tail risk then allocate proportionally.

Only invests in series whose VaR (α-quantile of predicted returns) exceeds
a threshold.  Among survivors, weights are proportional to predicted mean
return.

Parameters
----------
var_threshold : float
    Minimum VaR_α for a series to be included (e.g. 0.0 = tail-case return
    must be positive; -0.01 = allows a 1 % loss in the tail).
"""

import torch

from .base import BaseAllocator


class VaRAwareAllocator(BaseAllocator):
    """Heuristic: filter by tail risk, allocate proportionally to mean."""

    def __init__(self, *args, var_threshold: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_threshold = var_threshold

    def compute_raw_weights(
        self,
        samples: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        n_samples = samples.shape[-1]
        idx = self._var_alpha_idx(n_samples)
        sorted_s, _ = samples.sort(dim=-1)
        var_a = sorted_s[:, :, idx]                         # VaR_α per series  [batch, series]

        means = samples.mean(dim=-1)
        eligible = var_a > self.var_threshold
        return means * eligible.float()
