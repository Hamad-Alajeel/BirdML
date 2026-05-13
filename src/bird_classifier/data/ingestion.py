"""
Dataset ingestion utilities.

Responsibilities:
1. Check whether the raw Hugging Face parquet dataset exists in S3.
2. If missing, download it from Hugging Face.
3. Upload the downloaded raw dataset to S3.
4. Ensure the dataset exists locally.
5. Load train/validation/test splits from parquet.
"""

from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download

from bird_classifier.config import (
    AWS_REGION,
    DATA_DIR,
    S3_BUCKET,
    S3_DATA_PREFIX,
)


HF_REPO_ID = "yashikota/birds-525-species-image-classification"
HF_REPO_TYPE = "dataset"


def s3_prefix_exists(
    bucket: str = S3_BUCKET,
    prefix: str = S3_DATA_PREFIX,
    region: str = AWS_REGION,
) -> bool:
    """
    Check whether the dataset already exists in S3.

    We only need to know whether at least one object exists under the prefix.
    """

    s3 = boto3.client("s3", region_name=region)

    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        MaxKeys=1,
    )

    return "Contents" in response


def download_from_hugging_face(local_dir: Path = DATA_DIR) -> Path:
    """
    Download the Hugging Face dataset snapshot to a local directory.
    """

    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        local_dir=str(local_dir),
    )

    return local_dir


def upload_directory_to_s3(
    local_dir: Path = DATA_DIR,
    bucket: str = S3_BUCKET,
    prefix: str = S3_DATA_PREFIX,
    region: str = AWS_REGION,
) -> None:
    """
    Upload a local directory recursively to S3.
    """

    s3 = boto3.client("s3", region_name=region)

    for path in local_dir.rglob("*"):
        if path.is_file():
            relative_path = path.relative_to(local_dir)
            s3_key = f"{prefix}/{relative_path.as_posix()}"

            s3.upload_file(
                Filename=str(path),
                Bucket=bucket,
                Key=s3_key,
            )


def download_directory_from_s3(
    local_dir: Path = DATA_DIR,
    bucket: str = S3_BUCKET,
    prefix: str = S3_DATA_PREFIX,
    region: str = AWS_REGION,
) -> Path:
    """
    Download the raw dataset directory from S3 to local storage.
    """

    s3 = boto3.client("s3", region_name=region)
    local_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")

    found_any_object = False

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            found_any_object = True

            s3_key = obj["Key"]

            if s3_key.endswith("/"):
                continue

            relative_path = Path(s3_key).relative_to(prefix)
            local_path = local_dir / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            s3.download_file(
                Bucket=bucket,
                Key=s3_key,
                Filename=str(local_path),
            )

    if not found_any_object:
        raise FileNotFoundError(
            f"No objects found in s3://{bucket}/{prefix}. "
            "Dataset may not have been uploaded correctly."
        )

    return local_dir


def ensure_dataset_available(
    local_dir: Path = DATA_DIR,
    bucket: str = S3_BUCKET,
    prefix: str = S3_DATA_PREFIX,
    region: str = AWS_REGION,
) -> Path:
    """
    Ensure the raw dataset exists both in S3 and locally.

    Flow:
    - If S3 already has the dataset, download it locally if needed.
    - If S3 does not have the dataset, download from Hugging Face and upload to S3.
    """

    parquet_dir = local_dir / "data"

    if s3_prefix_exists(bucket=bucket, prefix=prefix, region=region):
        if not parquet_dir.exists():
            download_directory_from_s3(
                local_dir=local_dir,
                bucket=bucket,
                prefix=prefix,
                region=region,
            )

        return local_dir

    download_from_hugging_face(local_dir=local_dir)

    upload_directory_to_s3(
        local_dir=local_dir,
        bucket=bucket,
        prefix=prefix,
        region=region,
    )

    return local_dir


def load_bird_dataset(local_dir: Path = DATA_DIR, test_bool: bool = False) -> DatasetDict:
    """
    Load the bird dataset from local parquet files.

    Returns:
        DatasetDict with train/validation/test splits.
    """

    ensure_dataset_available(local_dir=local_dir)

    parquet_data_dir = local_dir / "data"

    if not parquet_data_dir.exists():
        raise FileNotFoundError(
            f"Expected parquet data directory at {parquet_data_dir}, "
            "but it does not exist."
        )

    dataset = load_dataset(
        "parquet",
        data_dir=str(parquet_data_dir),
    )

    if test_bool:
        dataset = dataset.filter(lambda row: int(row["label"]) in {0, 1})
        
    return dataset

def main() -> None:
    dataset = load_bird_dataset()
    print(dataset)


if __name__ == "__main__":
    main()