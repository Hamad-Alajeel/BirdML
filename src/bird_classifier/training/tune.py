"""Optuna hyperparameter search for the bird classifier."""

import json
from pathlib import Path
from typing import Any

import boto3
import mlflow
import optuna
import torch
from datasets import DatasetDict

from bird_classifier.config import (
    AWS_REGION,
    DEFAULT_HYPERPARAMS,
    MODELS_DIR,
    S3_BEST_PARAMS_KEY,
    S3_BUCKET,
    SEARCH_SPACE,
)
from bird_classifier.data.dataloaders import build_dataloaders
from bird_classifier.data.dataset import build_label_mapping
from bird_classifier.training.engine import run_epoch
from bird_classifier.training.model import build_model, get_device
from bird_classifier.training.train import build_optimizer, build_scheduler


def _sample_params(trial: optuna.Trial, tune_params: list[str]) -> dict[str, Any]:
    """
    Build a full hyperparameter dict for one Optuna trial.

    Params named in tune_params are sampled from SEARCH_SPACE.
    Everything else falls back to DEFAULT_HYPERPARAMS.
    """
    params = dict(DEFAULT_HYPERPARAMS)

    for name in tune_params:
        spec = SEARCH_SPACE[name]
        if "choices" in spec:
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec.get("type") == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )

    return params


def save_params_to_s3(
    params: dict[str, Any],
    bucket: str = S3_BUCKET,
    key: str = S3_BEST_PARAMS_KEY,
    region: str = AWS_REGION,
) -> str:
    """Serialize best params to JSON and upload to S3. Returns the S3 URI."""
    body = json.dumps(params, indent=2).encode("utf-8")
    boto3.client("s3", region_name=region).put_object(
        Body=body,
        Bucket=bucket,
        Key=key,
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


def run_tuning(
    dataset: DatasetDict,
    n_trials: int = 30,
    tune_params: list[str] | None = None,
    mlflow_experiment: str = "bird_finetuning",
    smoke_test: bool = False,
) -> tuple[dict[str, Any], Path, Path]:
    """
    Run Optuna hyperparameter search over tune_params.

    Each trial samples the named params from SEARCH_SPACE and uses DEFAULT_HYPERPARAMS
    for everything else. Runs the full two-stage training per trial and logs every
    epoch to MLflow.

    After all trials:
    - Tags the best top-1 and top-5 MLflow runs.
    - Saves best params to MODELS_DIR/best_params.json.

    Args:
        dataset: Hugging Face DatasetDict with train/validation/test splits.
        n_trials: Number of Optuna trials to run.
        tune_params: Which params from SEARCH_SPACE to search. Defaults to all.
        mlflow_experiment: MLflow experiment name.

    Returns:
        best_params, path to best top-1 checkpoint, path to best top-5 checkpoint.
    """
    if tune_params is None:
        tune_params = list(SEARCH_SPACE.keys())

    device = get_device()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    label_to_idx = build_label_mapping(dataset)

    best_top1, best_top5 = -1.0, -1.0
    best_top1_run_id, best_top5_run_id = None, None
    best_params: dict[str, Any] = {}

    top1_checkpoint = MODELS_DIR / "EN_best_top1.pth"
    top5_checkpoint = MODELS_DIR / "EN_best_top5.pth"

    mlflow.set_experiment(mlflow_experiment)

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_top1, best_top5, best_top1_run_id, best_top5_run_id, best_params

        params = _sample_params(trial, tune_params)

        if smoke_test:
            params = {
                **params,
                "batch_size": 4,
                "n_warmup_epochs": 1,
                "n_finetune_epochs": 1,
            }
            
        train_loader, val_loader, _ = build_dataloaders(
            dataset=dataset,
            label_to_idx=label_to_idx,
            batch_size=params["batch_size"],
        )

        n_warmup = params["n_warmup_epochs"]
        n_finetune = params["n_finetune_epochs"]

        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            run_id = mlflow.active_run().info.run_id
            mlflow.log_params({"trial_number": trial.number, **params})

            model = build_model().to(device)

            # --- Stage 1: Warmup ---
            for p in model.features.parameters():
                p.requires_grad = False

            warmup_opt = build_optimizer(model, params, stage="warmup")

            for ep in range(1, n_warmup + 1):
                train_loss, train_top1, train_top5 = run_epoch(
                    model, train_loader, device,
                    optimizer=warmup_opt, training=True,
                    epoch_label=f"Trial {trial.number} warmup {ep}/{n_warmup}",
                )
                val_loss, val_top1, val_top5 = run_epoch(
                    model, val_loader, device,
                    training=False,
                    epoch_label=f"Trial {trial.number} val {ep}",
                )
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss, "train_top1": train_top1,
                        "train_top5": train_top5, "val_loss": val_loss,
                        "val_top1": val_top1, "val_top5": val_top5,
                    },
                    step=ep,
                )

            # --- Stage 2: Finetune ---
            for p in model.features.parameters():
                p.requires_grad = True

            finetune_opt = build_optimizer(model, params, stage="finetune")
            scheduler = build_scheduler(finetune_opt, params, n_finetune)

            final_val_top1 = final_val_top5 = 0.0

            for ep in range(1, n_finetune + 1):
                global_ep = n_warmup + ep
                train_loss, train_top1, train_top5 = run_epoch(
                    model, train_loader, device,
                    optimizer=finetune_opt, training=True,
                    epoch_label=f"Trial {trial.number} finetune {ep}/{n_finetune}",
                )
                val_loss, val_top1, val_top5 = run_epoch(
                    model, val_loader, device,
                    training=False,
                    epoch_label=f"Trial {trial.number} val {global_ep}",
                )
                scheduler.step()
                final_val_top1, final_val_top5 = val_top1, val_top5
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss, "train_top1": train_top1,
                        "train_top5": train_top5, "val_loss": val_loss,
                        "val_top1": val_top1, "val_top5": val_top5,
                    },
                    step=global_ep,
                )

            # --- Save checkpoint if this trial beat the running best ---
            if final_val_top1 > best_top1:
                best_top1 = final_val_top1
                best_top1_run_id = run_id
                best_params = params
                torch.save(model.state_dict(), top1_checkpoint)
                print(f"  → New best top-1: {final_val_top1:.4f} (trial {trial.number})")

            if final_val_top5 > best_top5:
                best_top5 = final_val_top5
                best_top5_run_id = run_id
                torch.save(model.state_dict(), top5_checkpoint)
                print(f"  → New best top-5: {final_val_top5:.4f} (trial {trial.number})")

            trial.set_user_attr("val_top5", final_val_top5)
            trial.set_user_attr("run_id", run_id)

        return final_val_top1

    study = optuna.create_study(direction="maximize", study_name="bird_finetuning")
    study.optimize(objective, n_trials=n_trials)

    # Tag the best MLflow runs so they're easy to find in the UI
    client = mlflow.MlflowClient()
    if best_top1_run_id:
        client.set_tag(best_top1_run_id, "best_top1", "true")
    if best_top5_run_id:
        client.set_tag(best_top5_run_id, "best_top5", "true")

    # Persist best params locally
    params_path = MODELS_DIR / "best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2))
    print(f"\nBest params saved to {params_path}")
    print(f"Best top-1: {best_top1:.4f}  |  Best top-5: {best_top5:.4f}")

    return best_params, top1_checkpoint, top5_checkpoint
