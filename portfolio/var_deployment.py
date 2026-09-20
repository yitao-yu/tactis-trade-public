"""
Deployment allocator — long-short mean-variance optimizer with a soft
market-neutral (beta) penalty, a soft CVaR penalty, and a hard central
return band enforced by de-leveraging into cash.

Solves, for each batch item:

    max  μᵀw - (λ/2) Var(w·r)
             - γ · relu(return_lower - CVaR_α)       (downside tail, soft)
             - γ · relu(upperCVaR_α - return_upper)  (upside tail, soft)
    s.t. weight_lower ≤ w_i ≤ weight_upper    (shorts when weight_lower < 0)
         βᵀw → 0                              (beta neutral, soft penalty)
         τ′ · c · |w - w_prev|₁               (turnover penalty, soft — the
                                               coef τ′ scales with the real
                                               commission rate c ≥ 3e-4, so the
                                               optimizer anticipates fees)

After convergence the asset weights are de-leveraged by a single scalar
``s ∈ [0, 1]`` so the central ``band_confidence`` interval of the return
distribution lies inside ``[return_lower, return_upper]``, and the
residual ``1 - Σ|s·w|`` is placed in cash.  The final gross book is always
``Σ|assets| + cash = 1`` with ``cash ≥ 0`` — cash is the variance-control dial.

Assumes the last column of ``samples`` is the cash asset (zero return,
zero variance).  The shared constraint projection is bypassed: this allocator
owns its own position bounds, cardinality, market neutrality, hard return
band and gross normalisation.
"""

import torch

from .base import BaseAllocator


class VarDeploymentAllocator(BaseAllocator):
    """
    Long-short mean-variance deployment optimizer.

    Parameters
    ----------
    risk_aversion : float
        λ — penalty weight on portfolio variance (higher = more shrinkage).
    soft_cvar_weight : float
        γ — penalty multiplier for both tail constraints: the downside
        ``relu(return_lower - CVaR_α)`` and the upside
        ``relu(upperCVaR_α - return_upper)``.
    n_steps : int
        Gradient descent steps per batch.
    optimizer_lr : float
        Learning rate for the Adam optimizer over raw asset weights.
    beta_penalty_weight : float
        Soft penalty multiplier for ``(βᵀw)²`` (market neutrality).  0 disables.
    weight_lower : float
        Per-asset lower bound.  ``weight_lower < 0`` enables short selling.
    weight_upper : float
        Per-asset upper bound.
    max_assets : int | None
        Maximum number of non-zero asset positions (cardinality, by |w|).
    return_lower : float
        Lower edge of the hard central return band, and the soft CVaR floor
        (CVaR_α ≥ return_lower).
    return_upper : float
        Upper edge of the hard central return band, and the soft gain ceiling
        (upperCVaR_α ≤ return_upper).
    band_confidence : float
        Confidence of the symmetric central return band (e.g. 0.95).  The
        band edges are the ``(1 - band_confidence) / 2`` and
        ``1 - (1 - band_confidence) / 2`` quantiles.
    turnover_coef : float
        Penalty coefficient (τ′) on the L1 distance to ``prev_weights``,
        scaled by the commission rate: effective term is
        ``τ′ · max(commission_rate, 3e-4) · |w - prev|₁``.  ``0`` disables.
        With the default ``commission_rate = 3e-4`` and ``turnover_coef ≈ 16.7``
        the effective weight is ≈ ``0.005`` (the former fixed ``turnover_weight``).
    commission_rate : float
        Actual per-trade commission rate (fraction, e.g. ``3e-4`` = 3 bp).  The
        penalty uses ``max(commission_rate, 3e-4)`` so it never models a fee
        below the real CN cost.
    long_only_mask : list[bool] | None
        Per-series flag (length = num_series, assets only): True names cannot be
        shorted (their per-name lower bound is clamped to 0).  ``None`` keeps
        ``weight_lower`` for every name (eval/val/RL default — unchanged).
    use_hard_band : bool
        If True (default), enforce the hard central return band by
        de-leveraging into cash.  If False, skip the band and normalise
        gross exposure to 1 (full investment, no cash buffer).
    """

    def __init__(
        self,
        *args,
        risk_aversion: float = 1.0,
        soft_cvar_weight: float = 1.0,
        n_steps: int = 200,
        optimizer_lr: float = 0.01,
        beta_penalty_weight: float = 0.1,
        weight_lower: float = -0.05,
        weight_upper: float = 0.05,
        max_assets: int | None = 50,
        return_lower: float = -0.01,
        return_upper: float = 0.02,
        band_confidence: float = 0.95,
        turnover_coef: float = 0.0,
        commission_rate: float = 3e-4,
        long_only_mask: list[bool] | None = None,
        use_hard_band: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.risk_aversion = risk_aversion
        self.soft_cvar_weight = soft_cvar_weight
        self.n_steps = n_steps
        self.optimizer_lr = optimizer_lr
        self.beta_penalty_weight = beta_penalty_weight
        self.weight_lower = weight_lower
        self.weight_upper = weight_upper
        self.max_assets = max_assets
        self.return_lower = return_lower
        self.return_upper = return_upper
        self.band_confidence = band_confidence
        self.turnover_coef = turnover_coef
        self.commission_rate = commission_rate
        self._turnover_floor = 3e-4  # CN 3 bp — never model a cheaper fee
        self.long_only_mask = long_only_mask
        self.use_hard_band = use_hard_band

    @property
    def effective_turnover_coef(self) -> float:
        """Penalty weight actually applied: ``τ′ · max(c, 3e-4)`` (0 disables)."""
        if self.turnover_coef <= 0:
            return 0.0
        return self.turnover_coef * max(self.commission_rate, self._turnover_floor)

    def allocate(
        self,
        samples: torch.Tensor,
        mask: torch.Tensor,
        cash_index: int | None = None,
        prev_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Full pipeline for the deployment allocator.

        Bypasses the shared constraint projection — position bounds, market
        neutrality, the hard return band, cash de-leveraging and gross
        normalisation are all handled inside ``compute_raw_weights``.
        """
        return self.compute_raw_weights(samples, mask, prev_weights=prev_weights)

    def compute_raw_weights(
        self,
        samples: torch.Tensor,
        mask: torch.Tensor,
        prev_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Optimize long-short asset weights, then de-lever into cash to satisfy
        the hard central return band.

        ``prev_weights``: ``[batch, num_assets]`` (or ``[batch, num_series]``)
        previous portfolio used for the turnover penalty.  Ignored when
        ``turnover_coef == 0``.
        """
        device = samples.device
        dtype = samples.dtype
        batch, num_series, n_samples = samples.shape
        num_assets = num_series - 1
        samples_assets = samples[:, :num_assets, :]
        asset_mask = mask[:, :num_assets]
        tail_size = max(1, int(n_samples * self.var_alpha))

        # Per-name short ban (e.g. CN A-shares): flagged names get lower bound 0
        # instead of ``weight_lower``, so the optimizer itself never plans shorts
        # — the risk model (CVaR/band/beta) sees the book that will actually
        # execute, not phantom CN shorts.
        lower = torch.full((num_assets,), self.weight_lower, dtype=dtype, device=device)
        upper = torch.full((num_assets,), self.weight_upper, dtype=dtype, device=device)
        if self.long_only_mask is not None:
            lo = torch.as_tensor(list(self.long_only_mask), dtype=torch.bool, device=device)
            if lo.numel() != num_assets:
                raise ValueError(
                    f"long_only_mask length {lo.numel()} must equal num_assets {num_assets}"
                )
            lower[lo] = 0.0

        r_market = samples_assets.mean(dim=1)
        mkt_mean = r_market.mean(dim=1, keepdim=True)
        market_var = r_market.var(dim=1, keepdim=True, unbiased=False)
        mkt_centered = r_market - mkt_mean
        asset_centered = samples_assets - samples_assets.mean(dim=1, keepdim=True)
        cov_market = (asset_centered * mkt_centered.unsqueeze(1)).mean(dim=2)
        betas = cov_market / (market_var + 1e-8)
        betas[~asset_mask] = 0.0

        w = torch.zeros(batch, num_assets, device=device, dtype=dtype)
        w.requires_grad_(True)
        opt = torch.optim.Adam([w], lr=self.optimizer_lr)

        for _step in range(self.n_steps):
            with torch.no_grad():
                w.data[~asset_mask] = 0.0
                w.data.clamp_(lower, upper)

            port_ret = (w.unsqueeze(-1) * samples_assets).sum(dim=1)
            mu_port = port_ret.mean(dim=1)
            var_port = port_ret.var(dim=1, unbiased=False)

            sorted_ret, _ = port_ret.sort(dim=1)
            cvar = sorted_ret[:, :tail_size].mean(dim=1)
            cvar_violation = torch.relu(self.return_lower - cvar)
            upper_cvar = sorted_ret[:, -tail_size:].mean(dim=1)
            gain_violation = torch.relu(upper_cvar - self.return_upper)

            loss = (
                -mu_port
                + (self.risk_aversion / 2) * var_port
                + self.soft_cvar_weight * cvar_violation
                + self.soft_cvar_weight * gain_violation
            )

            if self.beta_penalty_weight > 0:
                beta_exp = (w * betas).sum(dim=1)
                loss = loss + self.beta_penalty_weight * (beta_exp ** 2)

            if self.effective_turnover_coef > 0 and prev_weights is not None:
                prev = prev_weights.detach().to(device=device, dtype=dtype)
                prev = prev[:, :num_assets] if prev.shape[1] > num_assets else prev
                turnover = (w - prev).abs().sum(dim=1)
                # Anticipated-commission term: coef scaled by the real fee rate
                # (never below the 3 bp CN floor).  Eff. ≈ 0.167 at 1 % fees and
                # ≈ 0.005 at 3 bp — tune via ``turnover_coef`` if that is too big.
                loss = loss + self.effective_turnover_coef * turnover

            opt.zero_grad()
            loss.mean().backward()
            opt.step()

        with torch.no_grad():
            w.clamp_(lower, upper)
            w[~asset_mask] = 0.0

            if self.max_assets is not None and self.max_assets < num_assets:
                K = min(self.max_assets, num_assets)
                _, topk_idx = w.abs().topk(K, dim=1)
                keep = torch.zeros_like(w)
                keep.scatter_(1, topk_idx, 1.0)
                w = w * keep
                w[~asset_mask] = 0.0

            gross = w.abs().sum(dim=1)                 # [batch]

            if self.use_hard_band:
                port_ret = (w.unsqueeze(-1) * samples_assets).sum(dim=1)  # [batch, n_samples]

                tail_q = (1.0 - self.band_confidence) / 2.0
                q_lo = torch.quantile(port_ret, tail_q, dim=1)
                q_hi = torch.quantile(port_ret, 1.0 - tail_q, dim=1)

                scale = torch.ones(batch, device=device, dtype=dtype)
                scale = torch.minimum(scale, 1.0 / gross.clamp(min=1e-8))
                lo_ratio = self.return_lower / q_lo.clamp(max=-1e-8)
                hi_ratio = self.return_upper / q_hi.clamp(min=1e-8)
                scale = torch.where(q_lo < self.return_lower, torch.minimum(scale, lo_ratio), scale)
                scale = torch.where(q_hi > self.return_upper, torch.minimum(scale, hi_ratio), scale)
                scale = scale.clamp(0.0, 1.0)

                w_scaled = w * scale.unsqueeze(1)
                cash = (1.0 - scale * gross).clamp(min=0.0)
            else:
                gross_safe = gross.clamp(min=1e-8)
                w_scaled = w / gross_safe.unsqueeze(1)
                cash = 1.0 - gross / gross_safe

        return torch.cat([w_scaled, cash.unsqueeze(1)], dim=1).detach()
