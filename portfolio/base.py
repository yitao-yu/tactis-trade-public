"""
Abstract base class for portfolio allocators.

Each allocator takes TACTiS Monte Carlo samples for one timestep and
returns a weight vector.  Constraint projection is shared via
``BaseAllocator`` so every allocator gets the same post-processing.

Cash asset support: if ``cash_index`` is provided, that column is
exempt from ``max_assets`` and ``max_weight`` constraints.
"""

from abc import ABC, abstractmethod

import torch

from .constraints import PortfolioConstraints


class BaseAllocator(ABC):
    """
    Allocator interface.

    Parameters
    ----------
    constraints : PortfolioConstraints
        Structural constraints applied after the allocator produces raw weights.
    var_alpha : float
        Tail quantile for VaR / CVaR computations (e.g. 0.05 = 5 % worst-case).
    """

    def __init__(
        self,
        constraints: PortfolioConstraints | None = None,
        var_alpha: float = 0.05,
    ):
        self.constraints = constraints or PortfolioConstraints()
        self.var_alpha = var_alpha
        #: Per-series flag: True = this name cannot be shorted (e.g. CN A-shares).
        #: ``None`` (default) keeps the allocator's own ``weight_lower`` for all
        #: names — eval/val/RL never set this, so their behavior is unchanged.
        self.long_only_mask: list[bool] | None = None

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_raw_weights(
        self,
        samples: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce raw (pre-constraint) weights.

        ``samples``:  ``[batch, series, num_samples]``
        ``mask``:     ``[batch, series]`` bool — True = valid.

        Returns ``[batch, series]`` — not necessarily non-negative or
        summing to 1 (constraint projection handles that).
        """
        ...

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def allocate(
        self,
        samples: torch.Tensor,
        mask: torch.Tensor,
        cash_index: int | None = None,
    ) -> torch.Tensor:
        """
        Full pipeline: raw weights → constraint projection.

        Returns ``[batch, series]`` final weights (non-negative, sum=1
        unless ``allow_short`` is enabled).
        """
        raw = self.compute_raw_weights(samples, mask)
        return self._apply_constraints(raw, mask, self.constraints, cash_index)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def compute_return(
        self,
        weights: torch.Tensor,
        true_returns: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Realised return: ``sum(w_i * r_true_i)`` per batch item.  ``[batch]``."""
        return (weights * true_returns.float()).sum(dim=1)

    def compute_var_cvar(
        self,
        samples: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ex-ante VaR and CVaR from Monte Carlo samples.

        Returns ``(var: [batch], cvar: [batch])``.
        """
        port = (weights.unsqueeze(-1) * samples.float()).sum(dim=1)   # [batch, num_samples]
        n_samples = port.shape[1]
        tail_size = max(1, int(n_samples * self.var_alpha))

        sorted_ret, _ = port.sort(dim=1)
        var = sorted_ret[:, self._var_alpha_idx(n_samples)]
        cvar = sorted_ret[:, :tail_size].mean(dim=1)
        return var, cvar

    def _var_alpha_idx(self, n_samples: int) -> int:
        """Index for the α-quantile in sorted returns."""
        return max(0, int(n_samples * self.var_alpha) - 1)

    # ------------------------------------------------------------------
    # Constraint projection
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_constraints(
        w: torch.Tensor,
        mask: torch.Tensor,
        constraints: PortfolioConstraints,
        cash_index: int | None = None,
    ) -> torch.Tensor:
        """
        Apply structural constraints to weight matrix in-place.

        Pipeline:
        1. Zero invalid positions.
        2. Cardinality — keep top-K by |weight| (cash always kept).
           **Skipped when ``dollar_neutral=True``**.
        3. Per-asset cap (cash exempt).
           **Skipped when ``dollar_neutral=True``**.
        4. Long-only clamp (if not allow_short).
        5. Gross-exposure cap.
           **Skipped when ``dollar_neutral=True``**.
        6. Normalise to sum=1 (long-only) or gross=1 (long-short).
           **Skipped when ``dollar_neutral=True``**.
        """
        w = w.clone()
        num_cols = w.shape[1]
        neutral = constraints.dollar_neutral
        limit = constraints.position_limit or constraints.max_weight

        # 1. Zero masked positions
        w[~mask] = 0.0

        # 2. Cardinality — keep top-K by |weight|; cash always kept
        if constraints.max_assets is not None and not neutral:
            if cash_index is not None:
                K = min(constraints.max_assets, num_cols)
                K_others = K - 1
                if K_others > 0:
                    abs_w = w.abs().clone()
                    abs_w[:, cash_index] = -1.0
                    _, topk_others = abs_w.topk(K_others, dim=1)
                    keep = torch.zeros_like(w)
                    keep.scatter_(1, topk_others, 1.0)
                    keep[:, cash_index] = 1.0
                    w = w * keep
                else:
                    keep = torch.zeros_like(w)
                    keep[:, cash_index] = 1.0
                    w = w * keep
            else:
                K = min(constraints.max_assets, num_cols)
                _, topk_idx = w.abs().topk(K, dim=1)
                keep = torch.zeros_like(w).scatter(1, topk_idx, 1.0)
                w = w * keep

        # 3. Per-asset cap — cash exempt.  Skipped for dollar-neutral
        #    (allocator handles projection internally).
        if limit is not None and not neutral:
            cash_col = w[:, cash_index].clone() if cash_index is not None else None
            w = w.clamp(-limit, limit)
            if cash_col is not None:
                w[:, cash_index] = cash_col

        # 4. Long-only
        if not constraints.allow_short:
            w = w.clamp(min=0)

        # 5. Gross-exposure cap
        gross = w.abs().sum(dim=1, keepdim=True)
        if constraints.max_gross_exposure is not None and not neutral:
            scale = (constraints.max_gross_exposure / gross.clamp(min=1e-8)).clamp(max=1.0)
            w = w * scale
            gross = w.abs().sum(dim=1, keepdim=True)

        # 6. Normalise — skip for dollar-neutral (gross exposure is free)
        if not neutral:
            if constraints.allow_short:
                denom = gross.clamp(min=1e-8)
            else:
                denom = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
            w = w / denom

        return w
