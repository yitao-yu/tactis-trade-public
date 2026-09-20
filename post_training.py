"""
Post-training RL fine-tuning — maximizes portfolio reward by tuning
TACTiS model parameters.

REINFORCE with an advantage computed against a frozen reference model's
reward (the reference model's own portfolio under the same allocator), a
symmetric squared log-density constraint to that reference (evaluated on
samples drawn from both the current and reference models), and a symmetric
hinge keeping log-probability near a sweet-spot target in both directions
(resisting over-concentration and over-flattening alike).

Usage:
    python post_training.py rl.resume_ckpt=outputs/.../custom_tactis_medium.pth
    python post_training.py rl.resume_ckpt=<path> rl.epochs=100 rl.reward=sharpe
"""

import random
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np

from tqdm import tqdm

import torch
from omegaconf import DictConfig, OmegaConf

from data.dataset import CustomReturnDataset, MitsuiReturnDataset
from metrics import compute_validation_metrics, _build_de_normalize, _augment_with_cash, _build_allocators
from model import build_model, subset_scope
from trainer import build_dataloader, make_time_tensors


def _apply_freeze(model, rl_cfg) -> list[str]:
    """Returns trainable parameter names."""
    freeze_map = {
        "freeze_embeddings": ["series_encoder"],
        "freeze_encoder": [
            "input_encoder",
            "time_encoding",
            "flow_encoder.",
            "copula_encoder.",
        ],
        "freeze_decoder": ["decoder."],
    }

    trainable = []
    for name, param in model.named_parameters():
        should_freeze = False
        for cfg_key, patterns in freeze_map.items():
            if rl_cfg.get(cfg_key, False):
                if any(pattern in name for pattern in patterns):
                    should_freeze = True
                    break
        if should_freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable.append(name)
    return trainable


@hydra.main(version_base="1.3", config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    rl_cfg = cfg.rl

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if rl_cfg.resume_ckpt is None:
        print("Error: rl.resume_ckpt is required.")
        print("Example: python post_training.py rl.resume_ckpt=outputs/.../model.pth")
        return

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if cfg.wandb.enabled:
        import wandb

        original_cwd = hydra.utils.get_original_cwd()
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"posttraining_{datetime.now():%Y-%m-%d}",
            mode=cfg.wandb.get("mode", "online"),
            dir=Path.cwd(),
        )

    cls = CustomReturnDataset if cfg.dataset.name == "custom" else MitsuiReturnDataset
    base_kwargs = dict(
        hist_len=cfg.dataset.hist_len,
        pred_len=cfg.dataset.pred_len,
        splits=tuple(cfg.dataset.splits),
        price_column=cfg.dataset.price_column,
    )

    hist_len = cfg.dataset.hist_len
    pred_len = cfg.dataset.pred_len

    train_ds = cls(split="train", **base_kwargs)
    val_ds = cls(split="val", **base_kwargs)
    _, val_loader = build_dataloader(train_ds, val_ds, rl_cfg.batch_size)
    num_series = train_ds.num_series

    print(f"Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")
    print(f"Num series: {num_series}")

    model = build_model(cfg, num_series)
    model.to(device)

    original_cwd = hydra.utils.get_original_cwd()
    ckpt_path = Path(original_cwd) / rl_cfg.resume_ckpt
    ckpt = torch.load(str(ckpt_path), map_location=device)
    ckpt_stage = ckpt.get("stage", 2)
    if ckpt_stage == 2:
        model.initialize_stage2()
        model.to(device)
    model.load_state_dict(ckpt["model"])

    model.set_stage(ckpt_stage)
    print(f"Loaded checkpoint stage={ckpt_stage}  best_val_loss={ckpt.get('best_val_loss', '?')}")

    ref_model = build_model(cfg, num_series)
    ref_model.to(device)
    if ckpt_stage == 2:
        ref_model.initialize_stage2()
        ref_model.to(device)
    ref_model.load_state_dict(ckpt["model"])
    ref_model.set_stage(ckpt_stage)
    ref_model.eval()
    ref_model.tactis.bagging_size = None
    for param in ref_model.parameters():
        param.requires_grad = False

    trainable = _apply_freeze(model, rl_cfg)
    total_trainable = sum(dict(model.named_parameters())[n].numel() for n in trainable)
    frozen_flags = []
    if rl_cfg.freeze_embeddings: frozen_flags.append("embeddings")
    if rl_cfg.freeze_encoder: frozen_flags.append("encoder")
    if rl_cfg.freeze_decoder: frozen_flags.append("decoder")
    print(f"Trainable params: {total_trainable:,}  (frozen: {', '.join(frozen_flags) if frozen_flags else 'none'})")

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=rl_cfg.lr,
    )

    reward_allocators = _build_allocators(
        list(rl_cfg.allocator),
        var_alpha=0.05,
    )
    reward_alloc = list(reward_allocators.values())[0]
    print(f"Reward allocator: {list(reward_allocators.keys())[0]}")

    mean_tsr, std_tsr = _build_de_normalize(train_ds.norm_stats, device)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_sharpe = -float("inf")

    def _run_validation(step: int, epoch: int) -> float:
        model.eval()
        tqdm.write(f"\n  [{step}] Running validation ...")
        t0 = time.time()
        val_metrics = compute_validation_metrics(
            model=model,
            val_loader=val_loader,
            hist_len=hist_len,
            pred_len=pred_len,
            device=device,
            norm_stats=val_ds.norm_stats,
            num_samples=rl_cfg.n_samples,
            recall_k=cfg.training.metrics.recall_k,
            var_alpha=cfg.training.metrics.var_alpha,
            sample_micro_batch=cfg.training.metrics.sample_micro_batch,
            allocator_configs=list(cfg.training.metrics.allocators),
            show_progress=True,
        )
        val_elapsed = time.time() - t0
        tqdm.write(f"  Validation done ({val_elapsed:.0f}s)")

        val_sharpe = 0.0
        wandb_log = {"val/global_step": step}
        for k, v in val_metrics.items():
            wandb_log[f"val/{k}"] = v

        for entry in cfg.training.metrics.allocators:
            name = entry["name"]
            sh = val_metrics.get(f"portfolio/{name}/sharpe", 0)
            mr = val_metrics.get(f"portfolio/{name}/mean_return", 0)
            vv = val_metrics.get(f"portfolio/{name}/var_violation", 0)
            print(f"    {name}: sharpe={sh:.4f}  mean_ret={mr:.6f}  var_vio={vv:.4f}")
            if name == "var_deployment":
                val_sharpe = sh

        print(f"    recall@10_mean: {val_metrics.get('recall@k_mean', 0):.4f}  "
              f"recall@10_q95: {val_metrics.get('recall@k_q95', 0):.4f}")

        nonlocal best_val_sharpe  # noqa: will be bound in outer scope

        if cfg.wandb.enabled:
            wandb.log(wandb_log)

        ckpt_name = f"rl_step{step:06d}_sharpe{val_sharpe:.4f}.pt"
        ckpt_path = checkpoint_dir / ckpt_name
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "global_step": step,
                "val_sharpe": val_sharpe,
                "val_metrics": val_metrics,
                "rl_cfg": OmegaConf.to_container(rl_cfg, resolve=True),
            },
            ckpt_path,
        )
        print(f"  Saved: {ckpt_name}")

        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_path = Path(f"{cfg.dataset.name}_{cfg.model.name}_rl.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch,
                    "global_step": step,
                    "val_sharpe": val_sharpe,
                },
                best_path,
            )
            print(f"  Best model saved: {best_path}")

        topk_ckpts = sorted(
            checkpoint_dir.glob("rl_step*_sharpe*.pt"),
            key=lambda p: float(p.stem.split("sharpe")[-1]),
            reverse=True,
        )
        for extra in topk_ckpts[rl_cfg.topk_checkpoints:]:
            extra.unlink()
            print(f"  Removed old checkpoint: {extra.name}")

        return val_sharpe

    # Validate the starting checkpoint to seed the reference-baseline sharpe.
    print("Validating the initial checkpoint ...")
    ref_val_sharpe = _run_validation(0, 0)

    global_step = 0
    for epoch in range(1, rl_cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_steps = 0

        train_loader, _ = build_dataloader(train_ds, None, rl_cfg.batch_size)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{rl_cfg.epochs}")

        for hist, pred, pred_mask in pbar:
            hist = hist.to(device)
            pred = pred.to(device)
            pred_mask = pred_mask.to(device)
            batch_size = hist.shape[0]

            for start in range(0, batch_size, rl_cfg.micro_batch):
                end = min(start + rl_cfg.micro_batch, batch_size)
                mb = end - start

                subset = random.sample(range(num_series), min(rl_cfg.series_subset, num_series))

                ht, pt = make_time_tensors(hist_len, pred_len, mb, device)

                # Both the reward samples and the score-function log-density come
                # from the SAME subset-conditioned distribution, so sample,
                # reward and REINFORCE gradient are on-policy-consistent, and the
                # encoder/copula attention cost drops from O(1302²) to
                # O(len(subset)²) instead of sampling the full joint and slicing
                # afterwards (the old code also scored the sliced columns under
                # sequential 0..len(subset)-1 embeddings, i.e. wrong identities).
                with subset_scope((model, ref_model), subset):
                    # ---- REINFORCE: sample for reward (no grad) ----
                    with torch.no_grad():
                        samples = model.sample_subset(
                            num_samples=rl_cfg.n_samples,
                            hist_time=ht,
                            hist_value=hist[start:end],
                            pred_time=pt,
                            series_idx=subset,
                        )
                        samples_pred_z = samples[:, :, -pred_len:, :]

                        samples_pred = (
                            samples_pred_z * std_tsr[None, subset, None, None]
                            + mean_tsr[None, subset, None, None]
                        )

                        ref_samples = ref_model.sample_subset(
                            num_samples=rl_cfg.n_samples,
                            hist_time=ht,
                            hist_value=hist[start:end],
                            pred_time=pt,
                            series_idx=subset,
                        )
                        ref_samples_pred_z = ref_samples[:, :, -pred_len:, :]

                        ref_samples_pred = (
                            ref_samples_pred_z * std_tsr[None, subset, None, None]
                            + mean_tsr[None, subset, None, None]
                        )

                        true_pred = (
                            pred[start:end] * std_tsr[None, :, None] + mean_tsr[None, :, None]
                        )

                    def _reward_fn(ret, var=None):
                        if rl_cfg.reward == "sharpe":
                            return ret.mean() / (ret.std() + 1e-8)
                        elif rl_cfg.reward == "mean_return":
                            return ret.mean()
                        elif rl_cfg.reward == "custom_reward_product":
                            vv = (ret < var).float().mean()
                            s_val = ret.mean() / (ret.std() + 1e-8)
                            is_pos = (s_val > 0).float()
                            return rl_cfg.reward_scale * s_val * ((1.0 - vv) ** is_pos)
                        else:
                            s_val = ret.mean() / (ret.std() + 1e-8)
                            return rl_cfg.reward_alpha * s_val + (1 - rl_cfg.reward_alpha) * ret.mean()

                    timestep_rewards = []
                    timestep_baselines = []
                    timestep_var_violations = []
                    for t in range(pred_len):
                        s_t = samples_pred[:, :, t, :]
                        ref_s_t = ref_samples_pred[:, :, t, :]
                        m_t = pred_mask[start:end][:, subset, t]
                        t_t = true_pred[:, subset, t]

                        n_valid = m_t.sum(dim=1)
                        usable = n_valid >= 2
                        if usable.sum() == 0:
                            continue

                        s_aug, m_aug, cash_idx = _augment_with_cash(s_t[usable], m_t[usable])
                        ref_s_aug, _, _ = _augment_with_cash(ref_s_t[usable], m_t[usable])
                        t_aug = torch.cat(
                            [t_t[usable], torch.zeros(usable.sum(), 1, device=device)], dim=1
                        )

                        w = reward_alloc.allocate(s_aug, m_aug, cash_index=cash_idx)
                        ret = reward_alloc.compute_return(w, t_aug, m_aug)
                        var, _ = reward_alloc.compute_var_cvar(s_aug, w, m_aug)
                        timestep_rewards.append(_reward_fn(ret, var))
                        timestep_var_violations.append((ret < var).float().mean())

                        w_ref = reward_alloc.allocate(ref_s_aug, m_aug, cash_index=cash_idx)
                        ret_ref = reward_alloc.compute_return(w_ref, t_aug, m_aug)
                        var_ref, _ = reward_alloc.compute_var_cvar(ref_s_aug, w_ref, m_aug)
                        timestep_baselines.append(_reward_fn(ret_ref, var_ref))

                    if not timestep_rewards:
                        continue

                    reward = torch.stack(timestep_rewards).mean()
                    baseline = torch.stack(timestep_baselines).mean()
                    advantage = reward - baseline.detach()
                    var_violation = torch.stack(timestep_var_violations).mean()

                    # ---- Score function: joint log P of all sampled actions ----
                    # ∇J = A · Σᵢ ∇log π_θ(a_i|s)  −  β · Σᵢ ∇(log π_θ − log π_ref)²
                    # Fold sample dim into batch, chunked to control peak GPU memory.
                    actions = samples_pred_z  # [mb, len(subset), pred_len, N]
                    ref_actions = ref_samples_pred_z  # [mb, len(subset), pred_len, N]
                    hist_subset = hist[start:end][:, subset, :]

                    # --- Reference log-probs on current + ref samples (no grad) ---
                    actions_folded_all = actions.permute(0, 3, 1, 2).reshape(
                        mb * rl_cfg.n_samples, len(subset), pred_len
                    )
                    ref_actions_folded_all = ref_actions.permute(0, 3, 1, 2).reshape(
                        mb * rl_cfg.n_samples, len(subset), pred_len
                    )
                    hist_exp_all = hist_subset.repeat_interleave(rl_cfg.n_samples, dim=0)
                    ht_exp_all = ht.repeat_interleave(rl_cfg.n_samples, dim=0)
                    pt_exp_all = pt.repeat_interleave(rl_cfg.n_samples, dim=0)

                    with torch.no_grad():
                        ref_m_l, ref_c_l = ref_model.loss(
                            hist_time=ht_exp_all,
                            hist_value=hist_exp_all,
                            pred_time=pt_exp_all,
                            pred_value=actions_folded_all,
                            nan_pred_mask=None,
                        )
                        ref_m_l_r, ref_c_l_r = ref_model.loss(
                            hist_time=ht_exp_all,
                            hist_value=hist_exp_all,
                            pred_time=pt_exp_all,
                            pred_value=ref_actions_folded_all,
                            nan_pred_mask=None,
                        )
                    ref_log_probs = (ref_m_l - ref_c_l).reshape(mb, rl_cfg.n_samples)
                    ref_log_probs_ref = (ref_m_l_r - ref_c_l_r).reshape(mb, rl_cfg.n_samples)

                    # --- Trainable score function (chunked) ---
                    opt.zero_grad()

                    log_prob = 0.0
                    sq_total = 0.0
                    hinge_total = 0.0
                    chunk_size = max(1, rl_cfg.n_samples // rl_cfg.n_chunks)
                    prev_bagging = model.tactis.bagging_size
                    model.tactis.bagging_size = None

                    for c in range(rl_cfg.n_chunks):
                        c_start = c * chunk_size
                        c_end = c_start + chunk_size if c < rl_cfg.n_chunks - 1 else rl_cfg.n_samples
                        c_n = c_end - c_start

                        chunk_actions = actions[:, :, :, c_start:c_end]
                        actions_folded = chunk_actions.permute(0, 3, 1, 2).reshape(
                            mb * c_n, len(subset), pred_len
                        )
                        ref_chunk_actions = ref_actions[:, :, :, c_start:c_end]
                        ref_actions_folded = ref_chunk_actions.permute(0, 3, 1, 2).reshape(
                            mb * c_n, len(subset), pred_len
                        )

                        hist_expanded = hist_subset.repeat_interleave(c_n, dim=0)
                        ht_expanded = ht.repeat_interleave(c_n, dim=0)
                        pt_expanded = pt.repeat_interleave(c_n, dim=0)

                        # Trainable model log-probs on current samples
                        m_l, c_l = model.loss(
                            hist_time=ht_expanded,
                            hist_value=hist_expanded,
                            pred_time=pt_expanded,
                            pred_value=actions_folded,
                            nan_pred_mask=None,
                        )
                        # Trainable model log-probs on reference samples
                        m_l_ref, c_l_ref = model.loss(
                            hist_time=ht_expanded,
                            hist_value=hist_expanded,
                            pred_time=pt_expanded,
                            pred_value=ref_actions_folded,
                            nan_pred_mask=None,
                        )

                        logp_ba = (m_l - c_l).reshape(mb, c_n)          # [mb, c_n]
                        ref_ba = ref_log_probs[:, c_start:c_end]        # [mb, c_n]

                        logp_ref_ba = (m_l_ref - c_l_ref).reshape(mb, c_n)
                        ref_ref_ba = ref_log_probs_ref[:, c_start:c_end]

                        chunk_logp = logp_ba.sum(dim=1).mean()
                        log_prob = log_prob + chunk_logp.item()

                        # Symmetric squared log-density ratio, averaged over
                        # current + reference sample sets.
                        sq_term = 0.25 * (
                            ((logp_ba - ref_ba.detach()) ** 2).mean()
                            + ((logp_ref_ba - ref_ref_ba.detach()) ** 2).mean()
                        )
                        sq_total = sq_total + sq_term.item()

                        # Euclidean hinge: keep log-probability near the sweet-spot
                        # target in BOTH directions — over-concentration (logp too
                        # high) and over-flattening (logp too low) are both penalised,
                        # so the bound holds regardless of the sign of the advantage.
                        # Quadratic in the deviation so its gradient (2·w·Δ) is
                        # hands-off at typical steps but overtakes the REINFORCE
                        # driving force during a collapse (the linear |Δ| hinge lost
                        # to it by ~9× per nat, letting logp run +28 → −225).
                        logp_sum = logp_ba.sum(dim=1)                       # [mb]
                        logp_hinge = (logp_sum - rl_cfg.logp_target).pow(2).mean()
                        hinge_total = hinge_total + logp_hinge.item()

                        chunk_loss = (
                            -advantage.detach() * chunk_logp
                            + rl_cfg.kl_beta * sq_term
                            + rl_cfg.logp_penalty_weight * logp_hinge
                        )
                        chunk_loss.backward()

                    model.tactis.bagging_size = prev_bagging
                # scope ends: embeddings restored before clip/grad/step

                trainable_params = [p for p in model.parameters() if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, rl_cfg.clip_grad)
                opt.step()

                step_loss = (
                    -advantage.item() * log_prob
                    + rl_cfg.kl_beta * sq_total
                    + rl_cfg.logp_penalty_weight * hinge_total
                )

                global_step += 1
                epoch_steps += 1
                epoch_loss += step_loss
                epoch_reward += reward.item()

                pbar.set_postfix(
                    reward=reward.item(),
                    base=baseline.item(),
                    adv=advantage.item(),
                    vv=var_violation.item(),
                    loss=step_loss,
                    logp=log_prob,
                    sq=sq_total,
                    hinge=hinge_total,
                    grad=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    step=global_step,
                )

                if cfg.wandb.enabled and global_step % rl_cfg.log_every_steps == 0:
                    wandb.log({
                        "rl/loss": step_loss,
                        "rl/reward": reward.item(),
                        "rl/baseline": baseline.item(),
                        "rl/advantage": advantage.item(),
                        "rl/var_violation": var_violation.item(),
                        "rl/logp": log_prob,
                        "rl/sq": sq_total,
                        "rl/logp_hinge": hinge_total,
                        "rl/grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
                        "rl/global_step": global_step,
                    })

            if rl_cfg.steps_per_epoch is not None and epoch_steps >= rl_cfg.steps_per_epoch:
                break

        pbar.close()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_reward = epoch_reward / max(epoch_steps, 1)
        print(f"Epoch {epoch:3d}/{rl_cfg.epochs}  "
              f"loss={avg_loss:.6f}  reward={avg_reward:.6f}  steps={epoch_steps}")

        if cfg.wandb.enabled:
            import wandb

            wandb.log(
                {
                    "rl/train/epoch": epoch,
                    "rl/train/loss": avg_loss,
                    "rl/train/reward": avg_reward,
                    "rl/train/global_step": global_step,
                }
            )

        # Per-epoch validation; promote the current model to be the frozen
        # reference baseline if val sharpe improves meaningfully over the
        # existing reference.
        val_sharpe = _run_validation(global_step, epoch)
        if val_sharpe > ref_val_sharpe + rl_cfg.baseline_refresh_margin:
            old_ref = ref_val_sharpe
            ref_model.load_state_dict(model.state_dict())
            ref_model.eval()
            ref_val_sharpe = val_sharpe
            print(f"  Reference baseline refreshed: val sharpe {val_sharpe:.4f} > "
                  f"{old_ref:.4f} + {rl_cfg.baseline_refresh_margin}")

    print(f"\nDone. Best val sharpe: {best_val_sharpe:.4f}")

    if cfg.wandb.enabled:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
