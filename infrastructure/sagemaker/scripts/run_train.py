"""
Entry point for the SageMaker Final Training Job.

Runs top-to-bottom inside the container:
  1. Load the dataset from S3.
  2. Read best_params.json from S3; fall back to BEST_PARAMS from config if absent.
  3. Run the full two-stage training and log to MLflow.
  4. Upload the final model checkpoint and MLflow runs to S3.
"""

import argparse
import json
from pathlib import Path

import boto3
import mlflow
from botocore.exceptions import ClientError

from bird_classifier.config import (
    AWS_REGION,
    BEST_PARAMS,
    MLRUNS_DIR,
    MODELS_DIR,
    S3_BEST_PARAMS_KEY,
    S3_BUCKET,
    S3_MLRUNS_PREFIX,
)
from bird_classifier.data.ingestion import load_bird_dataset, upload_directory_to_s3
from bird_classifier.data.dataset import build_idx_to_name, save_idx_to_name
from bird_classifier.training.train import run_training
from infrastructure.sagemaker.scripts._s3_utils import upload_file_to_s3


def load_params_from_s3(
    bucket: str = S3_BUCKET,
    key: str = S3_BEST_PARAMS_KEY,
    region: str = AWS_REGION,
) -> dict | None:
    """Download and parse best_params.json from S3. Returns None if the key doesn't exist."""
    try:
        response = boto3.client("s3", region_name=region).get_object(
            Bucket=bucket, Key=key
        )
        return json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bird classifier final training job.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/ml/model"),
        help="Local path SageMaker uses for model output artifacts.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny subset for quick local testing.",
    )
    parser.add_argument(
        "--skip-training",
        type=str,
        default="false",
        help="Pass 'true' to skip training and use the existing EN_final.pth in S3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_training.lower() == "true":
        print("=== Skip-training mode: using existing EN_final.pth in S3 ===")
        print("Loading dataset to regenerate class_names.json...")
        dataset = load_bird_dataset()
        class_names_path = MODELS_DIR / "class_names.json"
        save_idx_to_name(build_idx_to_name(dataset), class_names_path)
        class_names_uri = upload_file_to_s3(class_names_path)
        print(f"Class names uploaded to {class_names_uri}")
        print("Skipping training and MLflow sync.")
        return

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")

    print("=== Step 1/4: Loading dataset from S3 ===")
    dataset = load_bird_dataset(test_bool=args.smoke_test)

    print("=== Step 2/4: Loading hyperparameters ===")
    params = load_params_from_s3()
    if params is None:
        print(f"No best_params.json found at s3://{S3_BUCKET}/{S3_BEST_PARAMS_KEY}")
        print("Falling back to BEST_PARAMS from config.")
        params = BEST_PARAMS
    else:
        print(f"Loaded params from s3://{S3_BUCKET}/{S3_BEST_PARAMS_KEY}")
    print(json.dumps(params, indent=2))

    if args.smoke_test:
        print("Smoke test mode enabled: overriding training params.")
        params = {
            **params,
            "batch_size": 4,
            "n_warmup_epochs": 1,
            "n_finetune_epochs": 1,
        }

    print("=== Step 3/4: Running two-stage training ===")
    checkpoint_path = run_training(dataset=dataset, params=params)

    print("=== Step 4/4: Uploading artifacts to S3 ===")
    model_uri = upload_file_to_s3(checkpoint_path)
    print(f"Checkpoint uploaded to {model_uri}")

    class_names_path = MODELS_DIR / "class_names.json"
    save_idx_to_name(build_idx_to_name(dataset), class_names_path)
    class_names_uri = upload_file_to_s3(class_names_path)
    print(f"Class names uploaded to {class_names_uri}")

    upload_directory_to_s3(
        local_dir=MLRUNS_DIR,
        bucket=S3_BUCKET,
        prefix=S3_MLRUNS_PREFIX,
        region=AWS_REGION,
    )
    print("MLflow runs synced to S3.")


if __name__ == "__main__":
    main()
