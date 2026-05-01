"""Shared S3 helpers used by the SageMaker entry point scripts."""

from pathlib import Path

import boto3

from bird_classifier.config import AWS_REGION, S3_BUCKET, S3_MODELS_PREFIX


def upload_file_to_s3(
    local_path: Path,
    bucket: str = S3_BUCKET,
    prefix: str = S3_MODELS_PREFIX,
    region: str = AWS_REGION,
) -> str:
    """Upload a single file to S3. Returns the S3 URI."""
    s3_key = f"{prefix}/{local_path.name}"
    boto3.client("s3", region_name=region).upload_file(
        str(local_path), bucket, s3_key
    )
    return f"s3://{bucket}/{s3_key}"
