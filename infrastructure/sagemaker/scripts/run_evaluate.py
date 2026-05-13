"""
Entry point for the SageMaker Evaluation Processing Job.

Evaluates EN_final.pth (always produced by run_train.py) on the held-out test set.

Two gates stand between evaluation and deployment:
  1. Automatic (here): exits non-zero if top-1 < MIN_TEST_TOP1_FOR_REGISTRATION,
     which prevents the pipeline from registering a substandard model.
  2. Human (SageMaker Model Registry): if the automatic gate passes, the model is
     registered as PendingManualApproval. A human inspects the metrics and lineage
     in SageMaker Studio and explicitly Approves or Rejects before deployment proceeds.
"""

import logging
import sys

# SageMaker strips PYTHONPATH from the Dockerfile ENV; restore it here so
# bird_classifier (in /opt/ml/code/src) and infrastructure (in /opt/ml/code) import.
sys.path.insert(0, "/opt/ml/code/src")
sys.path.insert(0, "/opt/ml/code")

from bird_classifier.config import MIN_TEST_TOP1_FOR_REGISTRATION
from bird_classifier.data.dataloaders import build_dataloaders
from bird_classifier.data.dataset import build_label_mapping
from bird_classifier.data.ingestion import load_bird_dataset
from bird_classifier.evaluation.evaluate import download_checkpoint_from_s3, evaluate_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== Step 1/2: Loading dataset and building test loader ===")
    dataset = load_bird_dataset()
    label_to_idx = build_label_mapping(dataset)
    _, _, test_loader = build_dataloaders(dataset=dataset, label_to_idx=label_to_idx)

    logger.info("=== Step 2/2: Evaluating EN_final.pth on test set ===")
    checkpoint_path = download_checkpoint_from_s3("EN_final.pth")

    if checkpoint_path is None:
        logger.error("EN_final.pth not found in S3 or locally.")
        sys.exit(1)

    _, test_top1, _ = evaluate_checkpoint(checkpoint_path, test_loader, "Final model")

    logger.info("Top-1: %.4f | Threshold: %s", test_top1, MIN_TEST_TOP1_FOR_REGISTRATION)

    if test_top1 < MIN_TEST_TOP1_FOR_REGISTRATION:
        logger.error("Top-1 below threshold — model will not be registered.")
        sys.exit(1)

    logger.info("Automatic gate passed — model will be registered as PendingManualApproval.")
    logger.info("A human must approve in SageMaker Model Registry before deployment proceeds.")


if __name__ == "__main__":
    main()
