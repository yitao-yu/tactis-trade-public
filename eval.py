"""
Test-set evaluation for trained TACTiS models.

Reports:
  - Per-timestep recall@k (mean and q95) at multiple k values
  - Portfolio performance (return, VaR, CVaR, Sharpe) per allocator
  - Var_optimizer diagnostics (weight concentration, cash allocation, effective N)

Usage:
    python eval.py eval.checkpoint=outputs/2026-08-05/12-08-01-708369/custom_tactis_medium.pth
    python eval.py eval.checkpoint=<path> eval.n_samples=200 eval.sample_micro_batch=2
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import CustomReturnDataset, MitsuiReturnDataset
from metrics import _augment_with_cash, _build_allocators, _build_de_normalize, _recall_at_k
from model import build_model
from trainer import make_time_tensors


def _load_config() -> Any:
    """Load config with Hydra defaults merged, apply CLI overrides."""
    cfg = OmegaConf.load("cfg/config.yaml")
    cfg.dataset = OmegaConf.load(f"cfg/dataset/{cfg.defaults[1]['dataset']}.yaml")
    cfg.model = OmegaConf.load(f"cfg/model/{cfg.defaults[2]['model']}.yaml")

    overrides = [arg for arg in sys.argv[1:] if "=" in arg]
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def _print_results(results: dict[str, float], cfg: Any, elapsed: float) -> None:
    """Pretty-print evaluation results."""
    recall_k = list(cfg.eval.recall_k)
    pred_len = cfg.dataset.pred_len

    print("=" * 70)
    print(f"{cfg.eval.split.capitalize()} Evaluation Results")
    print("=" * 70)
    print(f"Dataset: {cfg.dataset.name}  |  Model: {cfg.dataset.name}_{cfg.model.name}")
    print(f"Checkpoint: {cfg.eval.checkpoint}")
    print(f"n_samples: {cfg.eval.n_samples}  |  micro_batch: {cfg.eval.sample_micro_batch}")
    print(f"Time: {elapsed:.1f}s")
    print("-" * 70)

    print("\nPer-timestep recall@k (mean-based):")
    header = "  " + "".join(f"   t={t + 1}  " for t in range(pred_len)) + "     avg"
    print(header)
    for k in recall_k:
        vals = [results.get(f"recall@{k}_mean_t{t + 1}", 0) * 100 for t in range(pred_len)]
        avg = results.get(f"recall@{k}_mean", 0) * 100
        row = f"  k={k:<4d}" + "".join(f"  {v:5.1f}%" for v in vals) + f"  {avg:5.1f}%"
        print(row)

    print("\nPer-timestep recall@k (q95-based):")
    print(header)
    for k in recall_k:
        vals = [results.get(f"recall@{k}_q95_t{t + 1}", 0) * 100 for t in range(pred_len)]
        avg = results.get(f"recall@{k}_q95", 0) * 100
        row = f"  k={k:<4d}" + "".join(f"  {v:5.1f}%" for v in vals) + f"  {avg:5.1f}%"
        print(row)

    for entry in cfg.eval.allocators:
        name = entry["name"]
        print(f"\nPortfolio: {name}")
        mr = results.get(f"portfolio/{name}/return_mean", 0)
        sh = results.get(f"portfolio/{name}/sharpe", 0)
        var = results.get(f"portfolio/{name}/var_mean", 0)
        cvar = results.get(f"portfolio/{name}/cvar_mean", 0)
        vi = results.get(f"portfolio/{name}/var_violation_mean", 0)
        print(f"  mean_return: {mr:.6f}  sharpe: {sh:.4f}  var: {var:.6f}  cvar: {cvar:.6f}  var_vio: {vi:.3f}")

        if "var_optimizer" in name:
            nw = results.get(f"portfolio/{name}/nonzero_weights_mean", 0)
            ca = results.get(f"portfolio/{name}/cash_allocation_mean", 0)
            en = results.get(f"portfolio/{name}/effective_n_mean", 0)
            mw = results.get(f"portfolio/{name}/max_weight_mean", 0)
            print(f"  nonzero_weights: {nw:.1f}  cash_alloc: {ca:.3f}  "
                  f"effective_n: {en:.1f}  max_w: {mw:.3f}")

        if "mv_neutral_cvar" in name:
            nw = results.get(f"portfolio/{name}/nonzero_weights_mean", 0)
            ca = results.get(f"portfolio/{name}/cash_allocation_mean", 0)
            en = results.get(f"portfolio/{name}/effective_n_mean", 0)
            ge = results.get(f"portfolio/{name}/gross_exposure_mean", 0)
            db = results.get(f"portfolio/{name}/dollar_balance_mean", 0)
            print(f"  nonzero_weights: {nw:.0f}  cash_alloc: {ca:.4f}  "
                  f"effective_n: {en:.0f}  gross_exp: {ge:.3f}  dollar_bal: {db:.6f}")

        if "mv_long" in name:
            nw = results.get(f"portfolio/{name}/nonzero_weights_mean", 0)
            ca = results.get(f"portfolio/{name}/cash_allocation_mean", 0)
            en = results.get(f"portfolio/{name}/effective_n_mean", 0)
            print(f"  nonzero_weights: {nw:.1f}  cash_alloc: {ca:.3f}  "
                  f"effective_n: {en:.1f}")

        if "var_deployment" in name:
            nw = results.get(f"portfolio/{name}/nonzero_weights_mean", 0)
            ca = results.get(f"portfolio/{name}/cash_allocation_mean", 0)
            en = results.get(f"portfolio/{name}/effective_n_mean", 0)
            ge = results.get(f"portfolio/{name}/gross_exposure_mean", 0)
            db = results.get(f"portfolio/{name}/dollar_balance_mean", 0)
            q025 = results.get(f"portfolio/{name}/return_q025_mean", 0)
            q975 = results.get(f"portfolio/{name}/return_q975_mean", 0)
            print(f"  nonzero_weights: {nw:.0f}  cash_alloc: {ca:.3f}  "
                  f"effective_n: {en:.1f}  gross_exp: {ge:.3f}  net_exp: {db:.3f}  "
                  f"q025: {q025:.5f}  q975: {q975:.5f}")

    print("=" * 70)


@torch.no_grad()
def evaluate_test(
    model,
    test_loader,
    hist_len: int,
    pred_len: int,
    device: torch.device,
    norm_stats: list[tuple[float, float]],
    eval_cfg: Any,
) -> dict[str, float]:
    """
    Full test evaluation: per-timestep recall at multiple k + portfolio metrics
    + var_optimizer diagnostics.

    All sampling is done in one pass over the test set.
    """
    allocators = _build_allocators(list(eval_cfg.allocators), eval_cfg.var_alpha)
    mean_tsr, std_tsr = _build_de_normalize(norm_stats, device)

    recall_k_values = list(eval_cfg.recall_k)
    n_samples = eval_cfg.n_samples
    micro_batch = eval_cfg.sample_micro_batch

    per_t_recall_mean = {k: {t: [] for t in range(pred_len)} for k in recall_k_values}
    per_t_recall_q95 = {k: {t: [] for t in range(pred_len)} for k in recall_k_values}

    agg: dict[str, list[float]] = defaultdict(list)

    for hist, pred, pred_mask in test_loader:
        hist = hist.to(device)
        pred = pred.to(device)
        pred_mask = pred_mask.to(device)
        batch_size = hist.shape[0]

        samples_all = []
        for start in range(0, batch_size, micro_batch):
            end = min(start + micro_batch, batch_size)
            mb = end - start
            ht, pt = make_time_tensors(hist_len, pred_len, mb, device)
            s = model.sample(
                num_samples=n_samples,
                hist_time=ht,
                hist_value=hist[start:end],
                pred_time=pt,
            )
            samples_all.append(s)
        samples = torch.cat(samples_all, dim=0)
        samples_pred = samples[:, :, -pred_len:, :]

        samples_pred = samples_pred * std_tsr[None, :, None, None] + mean_tsr[None, :, None, None]
        pred_raw = pred * std_tsr[None, :, None] + mean_tsr[None, :, None]

        for t in range(pred_len):
            s_t = samples_pred[:, :, t, :]
            m_t = pred_mask[:, :, t]
            true_t = pred_raw[:, :, t]

            for k in recall_k_values:
                r_mean = _recall_at_k(s_t, true_t, m_t, k, quantile=None)
                r_q95 = _recall_at_k(s_t, true_t, m_t, k, quantile=0.95)
                per_t_recall_mean[k][t].append(r_mean)
                per_t_recall_q95[k][t].append(r_q95)

            for name, allocator in allocators.items():
                n_valid = m_t.sum(dim=1)
                usable = n_valid >= 2
                if usable.sum() == 0:
                    continue

                s_aug, m_aug, cash_idx = _augment_with_cash(s_t[usable], m_t[usable])
                t_aug = torch.cat(
                    [true_t[usable], torch.zeros(usable.sum(), 1, device=device)], dim=1
                )

                with torch.enable_grad():
                    w = allocator.allocate(s_aug, m_aug, cash_index=cash_idx)

                ret = allocator.compute_return(w, t_aug, m_aug)
                var, cvar = allocator.compute_var_cvar(s_aug, w, m_aug)

                agg[f"portfolio/{name}/return"].extend(ret.cpu().tolist())
                agg[f"portfolio/{name}/var"].extend(var.cpu().tolist())
                agg[f"portfolio/{name}/cvar"].extend(cvar.cpu().tolist())
                violation = (ret < var).float()
                agg[f"portfolio/{name}/var_violation"].extend(violation.cpu().tolist())

                if "var_optimizer" in name:
                    nonzero = (w.abs() > 1e-8).sum(dim=1).float()
                    agg[f"portfolio/{name}/nonzero_weights"].extend(nonzero.cpu().tolist())

                    cash_w = w[:, cash_idx]
                    agg[f"portfolio/{name}/cash_allocation"].extend(cash_w.cpu().tolist())

                    max_w = w[:, :-1].max(dim=1)[0] if cash_idx == w.shape[1] - 1 else w.max(dim=1)[0]
                    agg[f"portfolio/{name}/max_weight"].extend(max_w.cpu().tolist())

                    hhi = (w ** 2).sum(dim=1)
                    eff_n = 1.0 / hhi.clamp(min=1e-8)
                    agg[f"portfolio/{name}/effective_n"].extend(eff_n.cpu().tolist())

                if "mv_neutral_cvar" in name:
                    nonzero = (w.abs() > 1e-8).sum(dim=1).float()
                    agg[f"portfolio/{name}/nonzero_weights"].extend(nonzero.cpu().tolist())

                    cash_w = w[:, cash_idx]
                    agg[f"portfolio/{name}/cash_allocation"].extend(cash_w.cpu().tolist())

                    hhi = (w[:, :-1] ** 2).sum(dim=1)
                    eff_n = 1.0 / hhi.clamp(min=1e-8)
                    agg[f"portfolio/{name}/effective_n"].extend(eff_n.cpu().tolist())

                    gross = w[:, :-1].abs().sum(dim=1)
                    agg[f"portfolio/{name}/gross_exposure"].extend(gross.cpu().tolist())

                    dollar_balance = w[:, :-1].sum(dim=1).abs()
                    agg[f"portfolio/{name}/dollar_balance"].extend(dollar_balance.cpu().tolist())

                if "mv_long" in name:
                    nonzero = (w.abs() > 1e-8).sum(dim=1).float()
                    agg[f"portfolio/{name}/nonzero_weights"].extend(nonzero.cpu().tolist())

                    cash_w = w[:, cash_idx]
                    agg[f"portfolio/{name}/cash_allocation"].extend(cash_w.cpu().tolist())

                    hhi = (w[:, :-1] ** 2).sum(dim=1)
                    eff_n = 1.0 / hhi.clamp(min=1e-8)
                    agg[f"portfolio/{name}/effective_n"].extend(eff_n.cpu().tolist())

                if "var_deployment" in name:
                    nonzero = (w.abs() > 1e-8).sum(dim=1).float()
                    agg[f"portfolio/{name}/nonzero_weights"].extend(nonzero.cpu().tolist())

                    cash_w = w[:, cash_idx]
                    agg[f"portfolio/{name}/cash_allocation"].extend(cash_w.cpu().tolist())

                    hhi = (w[:, :-1] ** 2).sum(dim=1)
                    eff_n = 1.0 / hhi.clamp(min=1e-8)
                    agg[f"portfolio/{name}/effective_n"].extend(eff_n.cpu().tolist())

                    gross = w[:, :-1].abs().sum(dim=1)
                    agg[f"portfolio/{name}/gross_exposure"].extend(gross.cpu().tolist())

                    dollar_balance = w[:, :-1].sum(dim=1)
                    agg[f"portfolio/{name}/dollar_balance"].extend(dollar_balance.cpu().tolist())

                    port_ret = (w.unsqueeze(-1) * s_aug).sum(dim=1)
                    q025 = torch.quantile(port_ret, 0.025, dim=1)
                    q975 = torch.quantile(port_ret, 0.975, dim=1)
                    agg[f"portfolio/{name}/return_q025"].extend(q025.cpu().tolist())
                    agg[f"portfolio/{name}/return_q975"].extend(q975.cpu().tolist())

    results: dict[str, float] = {}

    for k in recall_k_values:
        for t in range(pred_len):
            results[f"recall@{k}_mean_t{t + 1}"] = float(
                np.mean(per_t_recall_mean[k][t])
            )
            results[f"recall@{k}_q95_t{t + 1}"] = float(
                np.mean(per_t_recall_q95[k][t])
            )
        results[f"recall@{k}_mean"] = float(
            np.mean([np.mean(per_t_recall_mean[k][t]) for t in range(pred_len)])
        )
        results[f"recall@{k}_q95"] = float(
            np.mean([np.mean(per_t_recall_q95[k][t]) for t in range(pred_len)])
        )

    for key, values in agg.items():
        if any(x in key for x in ("return", "var", "cvar", "nonzero", "cash", "max_weight", "effective", "var_violation", "gross_exposure", "dollar_balance")):
            results[f"{key}_mean"] = float(np.mean(values))
        else:
            results[key] = float(np.mean(values))

    for name, _allocator in allocators.items():
        rets = agg.get(f"portfolio/{name}/return")
        if rets and len(rets) > 1:
            rets_arr = np.array(rets)
            std = rets_arr.std()
            results[f"portfolio/{name}/sharpe"] = float(rets_arr.mean() / std) if std > 1e-8 else 0.0
        else:
            results[f"portfolio/{name}/sharpe"] = 0.0

    return results


def main() -> None:
    cfg = _load_config()

    seed = cfg.eval.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if cfg.eval.checkpoint is None:
        print("Error: eval.checkpoint is required.  Specify a checkpoint path.")
        print("Example: python eval.py eval.checkpoint=outputs/.../custom_tactis_medium.pth")
        sys.exit(1)

    checkpoint_path = Path(cfg.eval.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cls = CustomReturnDataset if cfg.dataset.name == "custom" else MitsuiReturnDataset
    base_kwargs = dict(
        hist_len=cfg.dataset.hist_len,
        pred_len=cfg.dataset.pred_len,
        splits=tuple(cfg.dataset.splits),
        price_column=cfg.dataset.price_column,
    )

    eval_split = cfg.eval.get("split", "test")
    test_ds = cls(split=eval_split, **base_kwargs)
    print(
        f"Dataset: {cfg.dataset.name} | "
        f"num_series={test_ds.num_series} | "
        f"split={eval_split} | "
        f"windows={len(test_ds)}"
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(cfg, test_ds.num_series)
    model.to(device)

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    ckpt_stage = ckpt.get("stage", 2)
    if ckpt_stage == 2:
        model.initialize_stage2()
        model.to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    model.set_stage(ckpt_stage if ckpt_stage in (1, 2) else 2)
    print(f"Model stage: {ckpt_stage}  |  best_val_loss: {ckpt.get('best_val_loss', '?')}")

    import time
    t0 = time.time()
    assert test_ds.norm_stats is not None, f"{eval_split} dataset must have norm_stats"
    results = evaluate_test(
        model=model,
        test_loader=test_loader,
        hist_len=cfg.dataset.hist_len,
        pred_len=cfg.dataset.pred_len,
        device=device,
        norm_stats=test_ds.norm_stats,
        eval_cfg=cfg.eval,
    )
    elapsed = time.time() - t0

    _print_results(results, cfg, elapsed)

    output_dir = Path(cfg.eval.output_dir) if cfg.eval.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{eval_split}_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
