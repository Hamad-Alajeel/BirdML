"""
Entry point for the SageMaker Hyperparameter Tuning Job.

Runs top-to-bottom inside the container:
  1. Load the dataset from S3.
  2. Run Optuna search over the requested hyperparameters.
  3. Upload best_params.json, both best checkpoints, and MLflow runs to S3.

The --tune-params and --n-trials arguments map directly to the TuneParams and
TuneTrials pipeline parameters injected by the SageMaker Pipeline definition.
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# SageMaker strips PYTHONPATH from the Dockerfile ENV; restore it here so
# bird_classifier (in /opt/ml/code/src) and infrastructure (in /opt/ml/code) import.
sys.path.insert(0, "/opt/ml/code/src")
sys.path.insert(0, "/opt/ml/code")

from bird_classifier.config import (
    AWS_REGION,
    MLRUNS_DIR,
    S3_BUCKET,
    S3_MLRUNS_PREFIX,
    S3_MODELS_PREFIX,
)
from bird_classifier.data.ingestion import load_bird_dataset, upload_directory_to_s3
from bird_classifier.training.tune import run_tuning, save_params_to_s3
from infrastructure.sagemaker.scripts._s3_utils import upload_file_to_s3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bird classifier hyperparameter tuning job.")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--tune-params",
        type=str,
        default="lr_head,lr_backbone,n_warmup_epochs",
        help="Comma-separated hyperparameter names to search (must exist in SEARCH_SPACE).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/ml/model"),
        help="Local path SageMaker uses for model output artifacts.",
    )
    parser.add_argument(
        "--retune",
        type=str,
        default="false",
        help="Pass 'true' to run Optuna search. If 'false', exits immediately without tuning.",
    )
    parser.add_argument(
        "--smoke-test",
        type=str,
        default="false",
        help="Pass 'true' to run a tiny smoke test: one trial, small batch, few epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    smoke_test = args.smoke_test.lower() == "true"

    if args.retune.lower() != "true":
        logger.info("RetuneHyperparameters=false — skipping tuning.")
        return

    tune_params = [p.strip() for p in args.tune_params.split(",")]

    if smoke_test:
        tune_params = [
            "lr_head",
            "lr_backbone",
            "weight_decay",
            "optimizer",
            "scheduler",
        ]
        args.n_trials = 1

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")

    logger.info("=== Step 1/4: Loading dataset from S3 ===")
    dataset = load_bird_dataset(test_bool=smoke_test)

    logger.info("=== Step 2/4: Running tuning — %d trials over %s ===", args.n_trials, tune_params)
    best_params, top1_path, top5_path = run_tuning(
        dataset=dataset,
        n_trials=args.n_trials,
        tune_params=tune_params,
        smoke_test=smoke_test,
    )

    logger.info("=== Step 3/4: Uploading best params and checkpoints to S3 ===")
    params_uri = save_params_to_s3(best_params)
    logger.info("Best params uploaded to %s", params_uri)

    top1_uri = upload_file_to_s3(top1_path, prefix=S3_MODELS_PREFIX)
    top5_uri = upload_file_to_s3(top5_path, prefix=S3_MODELS_PREFIX)
    logger.info("Best top-1 checkpoint: %s", top1_uri)
    logger.info("Best top-5 checkpoint: %s", top5_uri)

    logger.info("=== Step 4/4: Syncing MLflow runs to S3 ===")
    upload_directory_to_s3(
        local_dir=MLRUNS_DIR,
        bucket=S3_BUCKET,
        prefix=S3_MLRUNS_PREFIX,
        region=AWS_REGION,
    )
    logger.info("MLflow runs synced to S3.")


if __name__ == "__main__":
    main()
