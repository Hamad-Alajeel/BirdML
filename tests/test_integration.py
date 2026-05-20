"""Integration tests — hit real AWS / external services.

Skipped by default. Run explicitly with:

    pytest -m integration

Requires AWS credentials in the environment.
"""

import json

import boto3
import pytest

from bird_classifier.config import AWS_REGION, S3_BUCKET, S3_MODELS_PREFIX
from bird_classifier.data.dataset import build_idx_to_name
from bird_classifier.data.ingestion import load_bird_dataset

pytestmark = pytest.mark.integration


def test_s3_class_names_match_current_dataset_mapping(tmp_path):
    """
    Regression test for the container-vs-notebook label-drift bug.

    `class_names.json` in S3 was uploaded by the training run. The dense index
    `i` in that file MUST equal the dense index `i` that build_idx_to_name
    produces from the current dataset on disk. If they ever diverge, the model
    in S3 is silently misaligned with the test labels — exactly the failure mode
    that caused the container to report ~0% accuracy while the notebook reported
    99.5%.
    """
    # 1. Pull class_names.json from S3 (what training pinned).
    local_path = tmp_path / "class_names.json"
    boto3.client("s3", region_name=AWS_REGION).download_file(
        S3_BUCKET, f"{S3_MODELS_PREFIX}/class_names.json", str(local_path)
    )
    s3_idx_to_name = {int(k): v for k, v in json.loads(local_path.read_text()).items()}

    # 2. Build the equivalent mapping from the current dataset.
    dataset = load_bird_dataset()
    current_idx_to_name = build_idx_to_name(dataset)

    # 3. Indices must align one-for-one. If this fails, retrain or re-sync the data.
    mismatches = {
        i: (s3_idx_to_name[i], current_idx_to_name.get(i))
        for i in s3_idx_to_name
        if s3_idx_to_name[i] != current_idx_to_name.get(i)
    }
    assert not mismatches, (
        f"Label mapping drift detected — the model in S3 is no longer aligned with the "
        f"current dataset. {len(mismatches)} mismatches out of {len(s3_idx_to_name)}. "
        f"First 5: {dict(list(mismatches.items())[:5])}"
    )
