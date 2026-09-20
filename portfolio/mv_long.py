"""
Mean-variance long-only allocator — gradient-descent with CVaR constraint,
beta neutral penalty, and position limits.

Solves, for each batch item:

    max  μᵀw - (λ/2) Var(w·r) - γ · relu(-max_loss - CVaR_α)
    s.t. w_i ≥ 0,  Σw_i = 1
         w_i ≤ b      (position limits)
         βᵀw → 0      (beta neutral, soft penalty)

Portfolio variance ``Var(w·r)`` and betas ``β_i = Cov(r_i, r_mkt) / Var(r_mkt)``
are computed directly from MC samples — no Σ matrix, no external market data.

Dollar neutrality is not enforced (long-only precludes it).  Normalisation and
other structural constraints are applied by the shared constraint pipeline.
"""

import torch

from .base import BaseAllocator
from .constraints import PortfolioConstraints


class MeanVarianceLongAllocator(BaseAllocator):
    """
    Gradient-descent long-only mean-variance optimisation with CVaR constraint
    and optional beta neutrality.

    Parameters
    ----------
    risk_aversion : float
        λ — penalty weight on portfolio variance (higher = more shrinkage).
    soft_cvar_weight : float
        γ — penalty multiplier for CVaR constraint violation.
    max_loss : float
        Maximum acceptable loss in the α-tail (CVaR_α ≥ -max_loss).
    n_steps : int
        Gradient descent steps per batch.
    optimizer_lr : float
        Learning rate for the Adam optimizer over logits.
    beta_penalty_weight : float
        Soft penalty multiplier for ``(βᵀw)²``.  Set to 0 to disable.
    position_limit : float | None
        Hard cap ``w_i ≤ position_limit``.  Enforced via soft penalty in
        the loss (the constraint pipeline applies a backup clamp).
    """

    def __init__(
        self,
        *args,
        risk_aversion: float = 1.0,
        soft_cvar_weight: float = 1.0,
        max_loss: float = 0.01,
        n_steps: int = 200,
        optimizer_lr: float = 0.01,
        beta_penalty_weight: float = 0.1,
        position_limit: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.risk_aversion = risk_aversion
        self.soft_cvar_weight = soft_cvar_weight
        self.max_loss = max_loss
        self.n_steps = n_steps
        self.optimizer_lr = optimizer_lr
        self.beta_penalty_weight = beta_penalty_weight
        self.position_limit = position_limit

    def compute_raw_weights(
        self,
        samples: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gradient descent with long-only (logits → softmax) and position limits.

        Assumes the last column of ``samples`` is the cash asset.  Cash is
        included in the softmax and the constraint pipeline handles its
        special treatment (exempt from position_limit and cardinality).
        """
        device = samples.device
        batch, num_series, n_samples = samples.shape
        num_assets = num_series - 1
        tail_size = max(1, int(n_samples * self.var_alpha))

        asset_mask = mask[:, :num_assets]

        r_market = samples[:, :num_assets, :].mean(dim=1)
        mkt_mean = r_market.mean(dim=1, keepdim=True)
        market_var = r_market.var(dim=1, keepdim=True, unbiased=False)
        mkt_centered = r_market - mkt_mean
        asset_centered = samples[:, :num_assets, :] - samples[:, :num_assets, :].mean(
            dim=1, keepdim=True
        )
        cov_market = (asset_centered * mkt_centered.unsqueeze(1)).mean(dim=2)
        betas = cov_market / (market_var + 1e-8)
        betas[~asset_mask] = 0.0

        logits = torch.zeros(batch, num_series, device=device, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=self.optimizer_lr)

        for _step in range(self.n_steps):
            w = logits.masked_fill(~mask, -1e9).softmax(dim=1)

            port_ret = (w.unsqueeze(-1) * samples).sum(dim=1)
            mu_port = port_ret.mean(dim=1)
            var_port = port_ret.var(dim=1, unbiased=False)

            sorted_ret, _ = port_ret.sort(dim=1)
            cvar = sorted_ret[:, :tail_size].mean(dim=1)
            cvar_violation = torch.relu(-self.max_loss - cvar)

            loss = (
                -mu_port
                + (self.risk_aversion / 2) * var_port
                + self.soft_cvar_weight * cvar_violation
            )

            if self.beta_penalty_weight > 0:
                beta_exp = (w[:, :num_assets] * betas).sum(dim=1)
                loss = loss + self.beta_penalty_weight * (beta_exp ** 2)

            if self.position_limit is not None:
                excess = torch.relu(w[:, :num_assets] - self.position_limit)
                loss = loss + 100.0 * excess.sum(dim=1)

            opt.zero_grad()
            loss.mean().backward()
            opt.step()

        w = logits.masked_fill(~mask, -1e9).softmax(dim=1).detach()
        return w
