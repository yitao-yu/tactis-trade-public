"""
Mean-variance neutral allocator with CVaR constraint — gradient-descent with
dollar/beta neutral constraints and soft CVaR penalty.

Solves, for each batch item:

    max  μᵀw - (λ/2) Var(w·r) - γ · relu(-max_loss - CVaR_α)
    s.t. 1ᵀw = 0       (dollar neutral, hard projection)
         |w_i| ≤ b      (position limits, clamp)
         βᵀw → 0       (beta neutral, soft penalty)

Portfolio variance ``Var(w·r)`` and betas ``β_i = Cov(r_i, r_mkt) / Var(r_mkt)``
are computed directly from MC samples — no Σ matrix, no external market data.
"""

import torch

from .base import BaseAllocator
from .constraints import PortfolioConstraints


class MeanVarianceNeutralCVaRAllocator(BaseAllocator):
    """
    Gradient-descent mean-variance optimisation with neutral constraints
    and soft CVaR penalty.

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
        Learning rate for the Adam optimizer over raw weights.
    beta_penalty_weight : float
        Soft penalty multiplier for ``(βᵀw)²``.  Set to 0 to disable.
    position_limit : float
        Hard cap ``|w_i| ≤ position_limit`` applied inside the gradient loop.
    max_assets : int | None
        Maximum number of non-zero asset positions (excluding cash).
        Truncated by ``|w_i|`` magnitude after the gradient loop, then
        re-zero-centered to maintain dollar neutrality.
    max_gross_exposure : float | None
        Target gross exposure ``Σ|w_i|``.  Weights are scaled to this
        target after zero-centering, then re-zero-centered.
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
        max_assets: int | None = None,
        max_gross_exposure: float | None = 1.0,
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
        self.max_assets = max_assets
        self.max_gross_exposure = max_gross_exposure

    def compute_raw_weights(
        self,
        samples: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gradient descent with hard dollar-neutral projection and position limits.

        Assumes the last column of ``samples`` is the cash asset (zero return,
        zero variance).  Cash is excluded from dollar-neutral and position-limit
        constraints.
        """
        device = samples.device
        batch, num_series, n_samples = samples.shape
        num_assets = num_series - 1
        dtype = samples.dtype
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

        w = torch.zeros(batch, num_series, device=device, dtype=dtype)
        w.requires_grad_(True)
        opt = torch.optim.Adam([w], lr=self.optimizer_lr)

        for _step in range(self.n_steps):
            with torch.no_grad():
                w.data[~mask] = 0.0

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

            opt.zero_grad()
            loss.mean().backward()
            opt.step()

            with torch.no_grad():
                w[:, :num_assets].clamp_(-self.position_limit, self.position_limit)
                valid_count = asset_mask.sum(dim=1, keepdim=True).clamp(min=1)
                imbalance = (
                    w[:, :num_assets].sum(dim=1, keepdim=True) / valid_count
                )
                w[:, :num_assets] -= imbalance * asset_mask.float()
                w.data[~mask] = 0.0

        with torch.no_grad():
            # Position limits
            w[:, :num_assets].clamp_(-self.position_limit, self.position_limit)

            # Dollar neutral (zero-center)
            valid_count = asset_mask.sum(dim=1, keepdim=True).clamp(min=1)
            imbalance = w[:, :num_assets].sum(dim=1, keepdim=True) / valid_count
            w[:, :num_assets] -= imbalance * asset_mask.float()

            # Normalise gross exposure to target
            if self.max_gross_exposure is not None:
                gross = w[:, :num_assets].abs().sum(dim=1, keepdim=True)
                scale = (self.max_gross_exposure / gross.clamp(min=1e-8)).clamp(max=1.0)
                w[:, :num_assets] *= scale

                # Re-zero-center after gross scaling
                valid_count = asset_mask.sum(dim=1, keepdim=True).clamp(min=1)
                imbalance = w[:, :num_assets].sum(dim=1, keepdim=True) / valid_count
                w[:, :num_assets] -= imbalance * asset_mask.float()

            # Position limits (belt-and-suspenders after rescaling)
            w[:, :num_assets].clamp_(-self.position_limit, self.position_limit)

            # Cardinality — keep top-K by |weight|, then re-neutralise
            if self.max_assets is not None:
                K = min(self.max_assets, num_assets)
                _, topk_idx = w[:, :num_assets].abs().topk(K, dim=1)
                keep = torch.zeros(batch, num_assets, device=device)
                keep.scatter_(1, topk_idx, 1.0)
                w[:, :num_assets] *= keep

                kept_count = keep.sum(dim=1, keepdim=True).clamp(min=1)
                imbalance = (
                    w[:, :num_assets].sum(dim=1, keepdim=True) / kept_count
                )
                w[:, :num_assets] = (w[:, :num_assets] - imbalance) * keep

                w[:, :num_assets].clamp_(-self.position_limit, self.position_limit)

            w.data[~mask] = 0.0

        return w.detach()
