"""
TACTiS two-stage training loop with checkpointing.

Stage 1: train flow (marginal) components
Stage 2: freeze flow, train copula decoder
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def build_dataloader(
    train_ds: Dataset,
    val_ds: Dataset | None,
    batch_size: int,
    num_workers: int = 0,
):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return train_loader, val_loader


def make_time_tensors(
    hist_len: int,
    pred_len: int,
    batch_size: int,
    device: torch.device,
):
    hist_time = torch.arange(0, hist_len, device=device)[None, :].expand(batch_size, -1)
    pred_time = torch.arange(hist_len, hist_len + pred_len, device=device)[None, :].expand(batch_size, -1)
    return hist_time, pred_time


def train_step(
    model,
    hist_value,
    pred_value,
    pred_mask,
    hist_time,
    pred_time,
    stage,
):
    marginal_logdet, copula_loss = model.loss(
        hist_time,
        hist_value,
        pred_time,
        pred_value,
        nan_pred_mask=pred_mask if stage == 1 else None,
    )

    if stage == 1:
        total_loss = -marginal_logdet
    else:
        total_loss = copula_loss

    return total_loss.mean(), marginal_logdet.mean().detach(), copula_loss.mean().detach()


@torch.no_grad()
def validate(model, val_loader, hist_len, pred_len, stage, device):
    model.eval()
    marginal_vals = []
    copula_vals = []
    for hist, pred, pred_mask in val_loader:
        hist = hist.to(device)
        pred = pred.to(device)
        pred_mask = pred_mask.to(device)
        batch_size = hist.shape[0]
        hist_time, pred_time = make_time_tensors(hist_len, pred_len, batch_size, device)
        _ = model.loss(
            hist_time,
            hist_value=hist,
            pred_time=pred_time,
            pred_value=pred,
            nan_pred_mask=pred_mask if stage == 1 else None,
        )
        marginal_vals.append(model.marginal_logdet.mean().item())
        copula_vals.append(model.copula_loss.mean().item())
    model.train()
    return float(np.mean(marginal_vals)), float(np.mean(copula_vals))


def switch_to_stage2(model, lr, weight_decay, device):
    model.set_stage(2)
    model.initialize_stage2()
    model.to(device)

    copula_params = [
        "copula_series_encoder",
        "copula_time_encoding",
        "copula_input_encoder",
        "copula_encoder",
        "decoder.copula",
    ]
    params_to_optimize = []
    for name, param in model.named_parameters():
        if any(pname in name for pname in copula_params):
            params_to_optimize.append(param)
        else:
            param.requires_grad = False

    return torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=weight_decay)


def save_checkpoint(model, optimizer, epoch, path, best_val_loss, stage):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "stage": stage,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt["stage"]


def _update_topk_checkpoints(
    entries: list[tuple[float, Path]],
    val_loss: float,
    model,
    optimizer,
    epoch: int,
    stage: int,
    checkpoint_dir: Path,
    max_size: int = 10,
    var_violation: float | None = None,
    sharpe: float | None = None,
) -> tuple[list[tuple[float, Path]], bool]:
    suffix = ""
    if var_violation is not None:
        suffix += f"_varviol{var_violation:.4f}"
    if sharpe is not None:
        suffix += f"_sharpe{sharpe:.4f}"
    path = checkpoint_dir / f"top_stage{stage}_epoch{epoch:04d}{suffix}.pt"
    save_checkpoint(model, optimizer, epoch, path, val_loss, stage)
    entries.append((val_loss, path))
    entries.sort(key=lambda x: x[0])
    is_new_best = bool(entries and entries[0][1] == path)
    while len(entries) > max_size:
        _, rm_path = entries.pop()
        rm_path.unlink(missing_ok=True)
    return entries, is_new_best


def train_stage(
    model,
    stage,
    train_ds,
    val_ds,
    cfg,
    device,
    optimizer,
    checkpoint_dir,
    resume_epoch=0,
    best_val_loss=float("inf"),
):
    batch_size = cfg.training.batch_size
    hist_len = cfg.dataset.hist_len
    pred_len = cfg.dataset.pred_len

    train_loader, val_loader = build_dataloader(train_ds, val_ds, batch_size)

    val_every = cfg.training.val_every_epochs
    total_epochs = cfg.training.epochs_stage1 if stage == 1 else cfg.training.epochs_stage2

    best_epoch = resume_epoch - 1 if resume_epoch > 0 else -1
    topk_entries: list[tuple[float, Path]] = []
    topk_size = cfg.training.get("topk_checkpoints", 10)

    for epoch in range(resume_epoch, total_epochs):
        epoch_losses = []
        epoch_marginal = []
        epoch_copula = []

        for hist, pred, pred_mask in train_loader:
            hist = hist.to(device)
            pred = pred.to(device)
            pred_mask = pred_mask.to(device)
            hist_time, pred_time = make_time_tensors(hist_len, pred_len, hist.shape[0], device)

            optimizer.zero_grad()
            loss, marginal, copula = train_step(
                model, hist, pred, pred_mask, hist_time, pred_time, stage,
            )
            loss.backward()

            if cfg.training.get("clip_grad"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad)

            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_marginal.append(marginal.item())
            epoch_copula.append(copula.item())

        avg_loss = float(np.mean(epoch_losses))

        val_loss = float("inf")
        val_marginal = float("inf")
        val_copula = float("inf")
        metric_vals: dict[str, float] = {}
        did_validate = False
        if val_loader is not None and ((epoch + 1) % val_every == 0 or epoch == total_epochs - 1):
            val_marginal, val_copula = validate(model, val_loader, hist_len, pred_len, stage, device)
            val_loss = -val_marginal if stage == 1 else val_copula
            did_validate = True

            metrics_cfg = cfg.training.get("metrics", {})
            metric_vals: dict[str, float] = {}
            if metrics_cfg.get("enabled", False):
                from metrics import compute_validation_metrics
                model.eval()
                metric_vals = compute_validation_metrics(
                    model=model,
                    val_loader=val_loader,
                    hist_len=hist_len,
                    pred_len=pred_len,
                    device=device,
                    norm_stats=val_loader.dataset.norm_stats,
                    num_samples=metrics_cfg.get("n_samples", 20),
                    recall_k=metrics_cfg.get("recall_k", 10),
                    var_alpha=metrics_cfg.get("var_alpha", 0.05),
                    allocator_configs=metrics_cfg.get("allocators"),
                    sample_micro_batch=metrics_cfg.get("sample_micro_batch", 4),
                )
                model.train()

            var_violation = metric_vals.get("portfolio/var_deployment/var_violation")
            var_sharpe = metric_vals.get("portfolio/var_deployment/sharpe")

            topk_entries, is_new_best = _update_topk_checkpoints(
                topk_entries, val_loss, model, optimizer,
                epoch, stage, checkpoint_dir, topk_size,
                var_violation=var_violation,
                sharpe=var_sharpe,
            )
            if is_new_best:
                best_val_loss = val_loss
                best_epoch = epoch

        rank_str = ""
        if topk_entries and val_loss != float("inf"):
            rank = next((i for i, (v, _) in enumerate(topk_entries) if v == val_loss), -1)
            if rank >= 0:
                rank_str = f" rank={rank + 1}/{len(topk_entries)}"

        if did_validate:
            val_str = (
                f"val_marginal={val_marginal:.4f} val_copula={val_copula:.4f} "
                f"val_loss={val_loss:.4f}"
            )
        else:
            val_str = "val_marginal=- val_copula=- val_loss=-"

        print(
            f"Stage {stage} Epoch {epoch + 1}/{total_epochs} | "
            f"train_loss={avg_loss:.4f} marginal={np.mean(epoch_marginal):.4f} "
            f"copula={np.mean(epoch_copula):.4f} | "
            f"{val_str}{rank_str}"
        )

        if cfg.get("wandb", {}).get("enabled", False):
            import wandb
            metrics = {
                f"stage{stage}/train_loss": avg_loss,
                f"stage{stage}/train_marginal": float(np.mean(epoch_marginal)),
                f"stage{stage}/train_copula": float(np.mean(epoch_copula)),
                "epoch": epoch,
            }
            if val_loss != float("inf"):
                metrics.update({
                    f"stage{stage}/val_marginal": val_marginal,
                    f"stage{stage}/val_copula": val_copula,
                    f"stage{stage}/val_loss": val_loss,
                })
            for k, v in metric_vals.items():
                metrics[f"stage{stage}/val/{k}"] = v
            wandb.log(metrics)

        save_checkpoint(
            model, optimizer, epoch,
            checkpoint_dir / f"last_stage{stage}.pt",
            best_val_loss, stage,
        )

    return best_val_loss, best_epoch


@torch.no_grad()
def predict(model, hist_value, pred_len, pred_time, device, num_samples=100):
    model.eval()
    hist_time = torch.arange(0, hist_value.shape[2], device=device)[None, :]
    samples = model.sample(
        num_samples=num_samples,
        hist_time=hist_time,
        hist_value=hist_value,
        pred_time=pred_time,
    )
    return samples[:, :, -pred_len:, :]
