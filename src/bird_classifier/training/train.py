"""Final training run using the best known hyperparameters."""

import json
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.optim as optim
from datasets import DatasetDict

from bird_classifier.config import (
    BEST_PARAMS,
    MODELS_DIR,
    RMS_ALPHA,
    RMS_EPS,
    RMS_MOMENTUM,
    SGD_MOMENTUM,
    STEPLR_GAMMA,
)
from bird_classifier.data.dataloaders import build_dataloaders
from bird_classifier.data.dataset import build_label_mapping
from bird_classifier.training.engine import run_epoch
from bird_classifier.training.model import build_model, get_device


def build_optimizer(
    model: torch.nn.Module,
    hyperparams: dict[str, Any],
    stage: str,
) -> optim.Optimizer:
    """
    Build the optimizer for warmup or finetune stage.

    Warmup: single param group covering only unfrozen params, at lr_head.
    Finetune: two param groups — classifier at lr_head, backbone at lr_backbone.
    lr_head is also passed as the global default so PyTorch doesn't complain when
    all groups already have explicit per-group lr values.
    """
    name = hyperparams["optimizer"]
    lr_head = hyperparams["lr_head"]
    lr_backbone = hyperparams["lr_backbone"]
    weight_decay = hyperparams["weight_decay"]

    if stage == "warmup":
        param_groups = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = [
            {"params": list(model.classifier.parameters()), "lr": lr_head},
            {"params": list(model.features.parameters()), "lr": lr_backbone},
        ]

    if name == "RMSprop":
        return optim.RMSprop(
            param_groups,
            lr=lr_head,
            momentum=RMS_MOMENTUM,
            alpha=RMS_ALPHA,
            eps=RMS_EPS,
            weight_decay=weight_decay,
        )
    elif name == "Adam":
        return optim.Adam(param_groups, lr=lr_head, weight_decay=weight_decay)
    elif name == "AdamW":
        return optim.AdamW(param_groups, lr=lr_head, weight_decay=weight_decay)
    elif name == "SGD":
        return optim.SGD(
            param_groups, lr=lr_head, momentum=SGD_MOMENTUM, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {name!r}")


def build_scheduler(
    optimizer: optim.Optimizer,
    hyperparams: dict[str, Any],
    n_finetune: int,
) -> optim.lr_scheduler.LRScheduler:
    """Build the learning rate scheduler for the finetune stage."""
    name = hyperparams["scheduler"]

    if name == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, n_finetune // 3),
            gamma=STEPLR_GAMMA,
        )
    elif name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_finetune)
    else:
        raise ValueError(f"Unknown scheduler: {name!r}")


def run_training(
    dataset: DatasetDict,
    params: dict[str, Any] | None = None,
    mlflow_experiment: str = "bird_training",
    checkpoint_name: str = "EN_final.pth",
) -> Path:
    """
    Run the full two-stage training with the given hyperparameters.

    Reads BEST_PARAMS from config if params is None. Logs every epoch to MLflow
    and saves the final model checkpoint to MODELS_DIR.

    Returns:
        Path to the saved checkpoint.
    """
    if params is None:
        params = BEST_PARAMS

    device = get_device()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / checkpoint_name

    label_to_idx = build_label_mapping(dataset)
    train_loader, val_loader, _ = build_dataloaders(
        dataset=dataset,
        label_to_idx=label_to_idx,
        batch_size=params["batch_size"],
    )

    n_warmup = params["n_warmup_epochs"]
    n_finetune = params["n_finetune_epochs"]

    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name="final_training"):
        mlflow.log_params(params)

        model = build_model().to(device)

        # --- Stage 1: Warmup — backbone frozen, head only ---
        for p in model.features.parameters():
            p.requires_grad = False

        warmup_opt = build_optimizer(model, params, stage="warmup")

        for ep in range(1, n_warmup + 1):
            train_loss, train_top1, train_top5 = run_epoch(
                model, train_loader, device,
                optimizer=warmup_opt, training=True,
                epoch_label=f"Warmup {ep}/{n_warmup}",
            )
            val_loss, val_top1, val_top5 = run_epoch(
                model, val_loader, device,
                training=False,
                epoch_label=f"Val {ep}",
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss, "train_top1": train_top1,
                    "train_top5": train_top5, "val_loss": val_loss,
                    "val_top1": val_top1, "val_top5": val_top5,
                },
                step=ep,
            )

        # --- Stage 2: Finetune — full network, two learning rates ---
        for p in model.features.parameters():
            p.requires_grad = True

        finetune_opt = build_optimizer(model, params, stage="finetune")
        scheduler = build_scheduler(finetune_opt, params, n_finetune)

        for ep in range(1, n_finetune + 1):
            global_ep = n_warmup + ep
            train_loss, train_top1, train_top5 = run_epoch(
                model, train_loader, device,
                optimizer=finetune_opt, training=True,
                epoch_label=f"Finetune {ep}/{n_finetune}",
            )
            val_loss, val_top1, val_top5 = run_epoch(
                model, val_loader, device,
                training=False,
                epoch_label=f"Val {global_ep}",
            )
            scheduler.step()
            mlflow.log_metrics(
                {
                    "train_loss": train_loss, "train_top1": train_top1,
                    "train_top5": train_top5, "val_loss": val_loss,
                    "val_top1": val_top1, "val_top5": val_top5,
                },
                step=global_ep,
            )

        torch.save(model.state_dict(), checkpoint_path)
        mlflow.log_artifact(str(checkpoint_path))
        print(f"Final model saved to {checkpoint_path}")

    return checkpoint_path
