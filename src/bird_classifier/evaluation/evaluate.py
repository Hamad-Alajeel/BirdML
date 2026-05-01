"""Evaluation entry point — loads a checkpoint and runs it on the test set."""

from pathlib import Path

import boto3
import torch
from botocore.exceptions import ClientError

from bird_classifier.config import (
    AWS_REGION,
    MODELS_DIR,
    S3_BUCKET,
    S3_MODELS_PREFIX,
)
from bird_classifier.data.dataloaders import build_dataloaders
from bird_classifier.data.dataset import build_label_mapping
from bird_classifier.data.ingestion import load_bird_dataset
from bird_classifier.training.engine import run_epoch
from bird_classifier.training.model import load_checkpoint, get_device


def download_checkpoint_from_s3(
    filename: str,
    local_dir: Path = MODELS_DIR,
    bucket: str = S3_BUCKET,
    prefix: str = S3_MODELS_PREFIX,
    region: str = AWS_REGION,
) -> Path | None:
    """Download a checkpoint from S3 to local_dir. Returns the local path, or None if not found."""
    local_path = local_dir / filename
    if local_path.exists():
        return local_path

    local_dir.mkdir(parents=True, exist_ok=True)
    s3_key = f"{prefix}/{filename}"

    try:
        boto3.client("s3", region_name=region).download_file(bucket, s3_key, str(local_path))
        return local_path
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return None
        raise


def evaluate_checkpoint(
    checkpoint_path: Path,
    loader,
    label: str,
) -> tuple[float, float, float]:
    """Load a checkpoint and evaluate it on the given loader."""
    device = get_device()
    model, device = load_checkpoint(checkpoint_path, device=device)

    test_loss, test_top1, test_top5 = run_epoch(
        model, loader, device,
        training=False,
        epoch_label=f"Test [{label}]",
    )

    print(f"\n=== {label} ({checkpoint_path.name}) ===")
    print(f"Test loss:      {test_loss:.4f}")
    print(f"Test Top-1 acc: {test_top1:.4f}  ({test_top1 * 100:.2f}%)")
    print(f"Test Top-5 acc: {test_top5:.4f}  ({test_top5 * 100:.2f}%)")

    return test_loss, test_top1, test_top5


def main() -> None:
    dataset = load_bird_dataset()
    label_to_idx = build_label_mapping(dataset)
    _, _, test_loader = build_dataloaders(dataset=dataset, label_to_idx=label_to_idx)

    for filename, label in [
        ("EN_best_top1.pth", "Best Top-1 model"),
        ("EN_best_top5.pth", "Best Top-5 model"),
    ]:
        checkpoint_path = download_checkpoint_from_s3(filename)
        if checkpoint_path is None:
            print(f"Checkpoint {filename} not found in S3 or locally — skipping.")
            continue
        evaluate_checkpoint(checkpoint_path, test_loader, label)


if __name__ == "__main__":
    main()
