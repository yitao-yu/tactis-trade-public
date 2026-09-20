"""
TACTiS training entry point with Hydra config.

Usage:
    python train.py                                  # custom + tactis_medium
    python train.py dataset=mitsui                   # mitsui dataset
    python train.py dataset=mitsui model=tactis_medium training.epochs_stage1=100
"""

import hydra
import random

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from data.dataset import MitsuiReturnDataset, CustomReturnDataset
from model import build_model
from trainer import train_stage, switch_to_stage2, save_checkpoint
from datetime import datetime


def create_datasets(cfg):
    config = cfg.dataset
    cls = CustomReturnDataset if config.name == "custom" else MitsuiReturnDataset
    base_kwargs = dict(
        hist_len=config.hist_len,
        pred_len=config.pred_len,
        splits=tuple(config.splits),
        price_column=config.price_column,
    )
    is_mitsui = config.name == "mitsui"
    use_official = config.get("use_official_test", False) and is_mitsui

    if is_mitsui:
        train_ds = cls(split="train", use_official_test=use_official, **base_kwargs)
        val_ds = cls(split="val", use_official_test=use_official, **base_kwargs)
        if use_official:
            test_ds = cls(split="test", use_official_test=True, norm_stats=train_ds.norm_stats, **base_kwargs)
        else:
            test_ds = cls(split="test", use_official_test=False, **base_kwargs)
    else:
        train_ds = cls(split="train", **base_kwargs)
        val_ds = cls(split="val", **base_kwargs)
        test_ds = cls(split="test", **base_kwargs)
    return train_ds, val_ds, test_ds


@hydra.main(version_base="1.3", config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if cfg.wandb.enabled:
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.dataset.name}_{cfg.model.name}_{datetime.now():%Y-%m-%d}",
            mode=cfg.wandb.get("mode", "online"),
            dir=Path.cwd(),
        )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = create_datasets(cfg)
    print(
        f"Dataset: {cfg.dataset.name} | "
        f"num_series={train_ds.num_series} | "
        f"timesteps={train_ds.timesteps} | "
        f"price_column={cfg.dataset.price_column}"
    )

    model = build_model(cfg, train_ds.num_series)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")

    if cfg.training.resume_from:
        print(f"Resuming from {cfg.training.resume_from}")
        ckpt = torch.load(cfg.training.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        resumed_stage = ckpt.get("stage", 1)
        print(f"Loaded stage {resumed_stage} checkpoint")

        if resumed_stage == 1:
            print("Switching to stage 2 ...")
            optimizer = switch_to_stage2(
                model, cfg.training.lr_stage2, cfg.training.weight_decay, device
            )
            best_val_loss, best_epoch = train_stage(
                model=model, stage=2, train_ds=train_ds, val_ds=val_ds,
                cfg=cfg, device=device, optimizer=optimizer,
                checkpoint_dir=checkpoint_dir,
            )
            save_checkpoint(
                model, optimizer, best_epoch,
                Path(f"{cfg.dataset.name}_{cfg.model.name}.pth"),
                best_val_loss=best_val_loss, stage=2,
            )
            print(f"\nModel saved to {cfg.dataset.name}_{cfg.model.name}.pth")
        else:
            print(f"Stage {resumed_stage} resume not yet implemented")
    else:

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.lr_stage1,
            weight_decay=cfg.training.weight_decay,
        )

        print("\n=== Stage 1: Training marginals ===")
        train_stage(
            model=model,
            stage=1,
            train_ds=train_ds,
            val_ds=val_ds,
            cfg=cfg,
            device=device,
            optimizer=optimizer,
            checkpoint_dir=checkpoint_dir,
        )

        print("\n=== Stage 2: Training copula ===")
        optimizer = switch_to_stage2(
            model, cfg.training.lr_stage2, cfg.training.weight_decay, device
        )
        best_val_loss, best_epoch = train_stage(
            model=model,
            stage=2,
            train_ds=train_ds,
            val_ds=val_ds,
            cfg=cfg,
            device=device,
            optimizer=optimizer,
            checkpoint_dir=checkpoint_dir,
        )

        save_checkpoint(
            model, optimizer, best_epoch,
            Path(f"{cfg.dataset.name}_{cfg.model.name}.pth"),
            best_val_loss=best_val_loss, stage=2,
        )

        print(f"\nModel saved to {cfg.dataset.name}_{cfg.model.name}.pth")

    if cfg.wandb.enabled:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
