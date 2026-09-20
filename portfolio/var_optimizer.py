"""
VaR / CVaR optimizer — gradient-descent portfolio allocation.

Solves, for each batch item:

    max_w   mean(w · samples)
    s.t.    CVaR_α(w · samples)  >=  -max_loss

The CVaR constraint is a soft penalty.  Hard structural constraints
(max_assets, max_weight, …) are projected after convergence.

Optimizes over all series (including cash if provided) — no pre-filter.
"""

import torch

from .base import BaseAllocator
from .constraints import PortfolioConstraints


class VaROptimizer(BaseAllocator):
    """
    Gradient-descent allocation with CVaR soft constraint.

    Parameters
    ----------
    n_steps : int
        Gradient descent steps per batch.
    optimizer_lr : float
        Learning rate for the Adam optimizer over logits.
    soft_cvar_weight : float
        Penalty multiplier λ for CVaR constraint violation.
    max_loss : float
        Maximum acceptable loss in the α-tail (CVaR_α ≥ -max_loss).
    """

    def __init__(
        self,
        *args,
        n_steps: int = 100,
        optimizer_lr: float = 0.05,
        soft_cvar_weight: float = 1.0,
        max_loss: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.optimizer_lr = optimizer_lr
        self.soft_cvar_weight = soft_cvar_weight
        self.max_loss = max_loss

    def compute_raw_weights(
        self,
        samples: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run gradient descent over all series (including cash) with soft
        CVaR penalty, returning raw (pre-constraint) weights.
        """
        device = samples.device
        batch, num_series, n_samples = samples.shape

        logits = torch.zeros(batch, num_series, device=device, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=self.optimizer_lr)

        tail_size = max(1, int(n_samples * self.var_alpha))

        for _step in range(self.n_steps):
            w = logits.masked_fill(~mask, -1e9).softmax(dim=1)  # [batch, num_series]
            port = (w.unsqueeze(-1) * samples).sum(dim=1)        # [batch, n_samples]

            mean_ret = port.mean(dim=1)

            sorted_ret, _ = port.sort(dim=1)
            cvar = sorted_ret[:, :tail_size].mean(dim=1)

            violation = torch.relu(-self.max_loss - cvar)
            loss = -mean_ret.mean() + self.soft_cvar_weight * violation.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        return logits.masked_fill(~mask, -1e9).softmax(dim=1).detach()
