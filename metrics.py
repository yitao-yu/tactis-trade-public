"""
Validation metrics — recall@k and portfolio performance.

``compute_validation_metrics`` draws Monte Carlo samples from the model
on the validation set and reports recall@k (mean-based and quantile-based)
plus portfolio returns for every configured allocator.
"""

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from portfolio import ALLOCATOR_REGISTRY, PortfolioConstraints


@torch.no_grad()
def compute_validation_metrics(
    model,
    val_loader,
    hist_len: int,
    pred_len: int,
    device: torch.device,
    norm_stats: list[tuple[float, float]],
    num_samples: int = 20,
    recall_k: int = 10,
    var_alpha: float = 0.05,
    allocator_configs: list[dict] | None = None,
    sample_micro_batch: int = 4,
    show_progress: bool = False,
) -> dict[str, float]:
    """
    Compute recall@k and per-allocator portfolio metrics from TACTiS samples.

    Parameters
    ----------
    model : TACTiSModel
        Model in eval mode (caller must set).
    val_loader : DataLoader
        Validation set — batches of ``(hist, pred, pred_mask)``.
    hist_len : int
    pred_len : int
    device : torch.device
    norm_stats : list[tuple[float, float]]
        Per-series ``(mean, std)`` from training-split normalization.
        Used to de-normalize returns into raw daily return space.
    num_samples : int
        Monte Carlo samples to draw per prediction window.
    recall_k : int
        ``k`` for recall@k.
    var_alpha : float
        Tail quantile for VaR / CVaR (e.g. 0.05 = 5 % worst-case).
    allocator_configs : list[dict] | None
        Each dict has ``name`` (key in ALLOCATOR_REGISTRY), ``constraints``,
        and per-allocator kwargs.  Defaults to proportional-only if None.
    sample_micro_batch : int
        Number of batch items per ``model.sample()`` call.  Controls GPU
        memory (1 for safe fallback, 4–8 for speed on large models).
    show_progress : bool
        If True, display a tqdm progress bar over the val loader.

    Returns
    -------
    dict[str, float]
        Flat dict of metric name → average value over all validation batches.
        All returns are in raw daily arithmetic return space (e.g. 0.0023 = 0.23 %).
    """
    from trainer import make_time_tensors

    if allocator_configs is None:
        allocator_configs = [{"name": "proportional"}]

    allocators = _build_allocators(allocator_configs, var_alpha)
    mean_tsr, std_tsr = _build_de_normalize(norm_stats, device)

    agg: dict[str, list[float]] = defaultdict(list)

    val_iter = val_loader
    if show_progress:
        from tqdm import tqdm

        val_iter = tqdm(val_loader, desc="  Validating", leave=True)

    for hist, pred, pred_mask in val_iter:
        hist = hist.to(device)
        pred = pred.to(device)
        pred_mask = pred_mask.to(device)

        batch_size = hist.shape[0]
        hist_time, pred_time = make_time_tensors(hist_len, pred_len, batch_size, device)

        samples_all = []
        for start in range(0, batch_size, sample_micro_batch):
            end = min(start + sample_micro_batch, batch_size)
            mb = end - start
            ht, pt = make_time_tensors(hist_len, pred_len, mb, device)
            s = model.sample(
                num_samples=num_samples,
                hist_time=ht,
                hist_value=hist[start:end],
                pred_time=pt,
            )
            samples_all.append(s)
        samples = torch.cat(samples_all, dim=0)
        samples_pred = samples[:, :, -pred_len:, :]          # [batch, series, pred_len, n_samples]

        # De-normalize to raw daily returns
        samples_pred = samples_pred * std_tsr[None, :, None, None] + mean_tsr[None, :, None, None]
        pred_raw = pred * std_tsr[None, :, None] + mean_tsr[None, :, None]

        # ---- recall@k (per timestep, averaged) ----
        for t in range(pred_len):
            s_t = samples_pred[:, :, t, :]                    # [batch, series, n_samples]
            m_t = pred_mask[:, :, t]                           # [batch, series]

            _recall_mean = _recall_at_k(s_t, pred_raw[:, :, t], m_t, recall_k, quantile=None)
            _recall_q95 = _recall_at_k(s_t, pred_raw[:, :, t], m_t, recall_k, quantile=0.95)

            agg["recall@k_mean"].append(_recall_mean)
            agg["recall@k_q95"].append(_recall_q95)

        # ---- portfolio metrics per allocator ----
        for name, allocator in allocators.items():
            for t in range(pred_len):
                s_t = samples_pred[:, :, t, :]
                m_t = pred_mask[:, :, t]
                t_t = pred_raw[:, :, t]

                n_valid = m_t.sum(dim=1)                       # [batch]
                usable = n_valid >= 2
                if usable.sum() == 0:
                    continue

                s_aug, m_aug, cash_idx = _augment_with_cash(s_t[usable], m_t[usable])
                t_aug = torch.cat([t_t[usable], torch.zeros(usable.sum(), 1, device=device)], dim=1)

                with torch.enable_grad():
                    w = allocator.allocate(s_aug, m_aug, cash_index=cash_idx)

                ret = allocator.compute_return(w, t_aug, m_aug)
                var, cvar = allocator.compute_var_cvar(s_aug, w, m_aug)

                agg[f"portfolio/{name}/return"].extend(ret.cpu().tolist())
                agg[f"portfolio/{name}/var"].extend(var.cpu().tolist())
                agg[f"portfolio/{name}/cvar"].extend(cvar.cpu().tolist())

    result: dict[str, float] = {}
    for k, v in agg.items():
        if not k.startswith("portfolio/") or not (
            "/" in k[len("portfolio/"):]
            and k.split("/")[-1] in ("sharpe", "mean_return", "var_violation")
        ):
            result[k] = float(np.mean(v))

    for name in allocators:
        all_ret = np.array(agg.get(f"portfolio/{name}/return", []), dtype=np.float64)
        all_var = np.array(agg.get(f"portfolio/{name}/var", []), dtype=np.float64)
        if len(all_ret) == 0:
            result[f"portfolio/{name}/mean_return"] = 0.0
            result[f"portfolio/{name}/sharpe"] = 0.0
            result[f"portfolio/{name}/var_violation"] = 0.0
            continue
        global_mu = float(np.mean(all_ret))
        global_sigma = float(np.std(all_ret)) if len(all_ret) > 1 else 1.0
        result[f"portfolio/{name}/mean_return"] = global_mu
        result[f"portfolio/{name}/sharpe"] = float(global_mu / global_sigma) if global_sigma > 1e-8 else 0.0
        if len(all_var) > 0:
            result[f"portfolio/{name}/var_violation"] = float(np.mean(all_ret < all_var))
        else:
            result[f"portfolio/{name}/var_violation"] = 0.0

    return result


# ------------------------------------------------------------------
# De-normalization
# ------------------------------------------------------------------


def _build_de_normalize(
    norm_stats: list[tuple[float, float]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-series ``(mean, std)`` tensors for de-normalizing z-scored returns
    back to raw daily arithmetic return space.
    """
    means = torch.tensor([s[0] for s in norm_stats], dtype=torch.float32, device=device)
    stds = torch.tensor([s[1] for s in norm_stats], dtype=torch.float32, device=device)
    return means, stds


# ------------------------------------------------------------------
# Recalls
# ------------------------------------------------------------------


@torch.no_grad()
def _recall_at_k(
    samples: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    quantile: float | None = None,
) -> float:
    """
    Recall@k for a single timestep.

    ``samples``:  ``[batch, series, n_samples]``
    ``true``:     ``[batch, series]``
    ``mask``:     ``[batch, series]`` bool — True = valid
    ``quantile``: ``None`` → rank by sample mean; ``0.95`` → rank by 95th percentile.

    Returns scalar recall averaged over batch items that have a valid true top-1.
    """
    if quantile is not None:
        ranking_value = torch.quantile(samples, quantile, dim=-1)   # [batch, series]
    else:
        ranking_value = samples.mean(dim=-1)

    # Zero out invalid series so they never rank
    ranking_value[~mask] = float("-inf")
    true_rank = true.clone()
    true_rank[~mask] = float("-inf")

    _, topk_idx = ranking_value.topk(k, dim=1)                      # [batch, k]
    _, true_top1 = true_rank.topk(1, dim=1)                         # [batch, 1]

    hits = (topk_idx == true_top1).any(dim=1)                       # [batch]
    valid_top1 = mask.gather(1, true_top1).squeeze(-1)              # [batch]

    weight = valid_top1.sum()
    if weight == 0:
        return 0.0
    return (hits & valid_top1).float().sum().item() / weight.item()


# ------------------------------------------------------------------
# Cash augmentation
# ------------------------------------------------------------------


def _augment_with_cash(
    samples: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Append a cash asset (zero return, zero variance, independent).

    ``samples``:  ``[batch, series, n_samples]``
    ``mask``:     ``[batch, series]`` bool

    Returns ``(aug_samples, aug_mask, cash_index)``.
    """
    device = samples.device
    batch = samples.shape[0]
    n_samples = samples.shape[2]
    cash_samples = torch.zeros(batch, 1, n_samples, device=device)
    cash_mask = torch.ones(batch, 1, dtype=torch.bool, device=device)
    return (
        torch.cat([samples, cash_samples], dim=1),
        torch.cat([mask, cash_mask], dim=1),
        samples.shape[1],                                    # index of cash column
    )


# ------------------------------------------------------------------
# Allocator factory
# ------------------------------------------------------------------


def _build_allocators(
    configs: list[dict],
    var_alpha: float,
) -> dict[str, Any]:
    """
    Instantiate allocators from config dicts.

    Each config dict:
        name        : str             — key in ALLOCATOR_REGISTRY
        constraints : dict | None     — kwargs for PortfolioConstraints
        kwargs      : dict            — per-allocator __init__ kwargs
    """
    allocators: dict[str, Any] = {}
    for entry in configs:
        name = entry["name"]
        cls = ALLOCATOR_REGISTRY[name]
        constraints = PortfolioConstraints(**(entry.get("constraints") or {}))
        extra = {k: v for k, v in entry.items() if k not in ("name", "constraints")}
        allocators[name] = cls(
            constraints=constraints,
            var_alpha=var_alpha,
            **extra,
        )
    return allocators
