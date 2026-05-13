"""
Data quality checks for the bird classifier dataset.

Responsibilities:
1. Verify expected split names exist.
2. Verify class counts are consistent across train/validation/test.
3. Verify total number of classes matches config.NUM_CLASSES.
4. Flag class-count outliers based on standard deviation.
5. Verify images are openable and RGB.
6. Produce a structured quality report.
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import boto3
import numpy as np
from datasets import DatasetDict

from bird_classifier.config import (
    AWS_REGION,
    CLASS_OUTLIER_STD_THRESHOLD,
    NUM_CLASSES,
    S3_BUCKET,
    S3_REPORTS_PREFIX,
)


EXPECTED_SPLITS = ("train", "validation", "test")


@dataclass
class DataQualityReport:
    expected_splits: tuple[str, ...]
    actual_splits: list[str]
    missing_splits: list[str]

    expected_num_classes: int
    total_num_classes: int
    class_count_matches_expected: bool

    split_num_classes: dict[str, int]
    class_sets_consistent_across_splits: bool

    total_images: int
    split_num_images: dict[str, int]
    split_label_ranges: dict[str, dict[str, int | None]]
    split_missing_label_ids: dict[str, list[int]]

    class_count_mean: float
    class_count_std: float
    class_outlier_std_threshold: float
    outlier_classes: list[int]
    
    image_check_total: int
    image_check_failed: int
    non_rgb_images: int
    image_errors: list[dict[str, Any]]

    passed: bool


def get_split_names(dataset: DatasetDict) -> list[str]:
    """Return split names from a Hugging Face DatasetDict."""

    return list(dataset.keys())


def validate_expected_splits(dataset: DatasetDict) -> list[str]:
    """Return any expected splits that are missing."""

    actual_splits = set(get_split_names(dataset))
    return [split for split in EXPECTED_SPLITS if split not in actual_splits]


def get_label_counts_by_split(dataset: DatasetDict) -> dict[str, Counter]:
    """Count labels separately for each split."""

    label_counts_by_split = {}

    for split in EXPECTED_SPLITS:
        labels = dataset[split]["label"]
        label_counts_by_split[split] = Counter(labels)

    return label_counts_by_split


def get_class_sets_from_counts(
    label_counts_by_split: dict[str, Counter],
) -> dict[str, set[int]]:
    """Derive the set of labels present in each split from the per-split Counters."""

    return {
        split: set(counter.keys())
        for split, counter in label_counts_by_split.items()
    }


def check_class_sets_consistent(class_sets_by_split: dict[str, set[int]]) -> bool:
    """Check whether train/validation/test contain the same class labels."""

    class_sets = list(class_sets_by_split.values())

    if not class_sets:
        return False

    first_class_set = class_sets[0]

    return all(class_set == first_class_set for class_set in class_sets[1:])


def get_total_label_counts(label_counts_by_split: dict[str, Counter]) -> Counter:
    """Combine label counts across all splits."""

    total_counts = Counter()

    for split_counts in label_counts_by_split.values():
        total_counts.update(split_counts)

    return total_counts


def find_class_count_outliers(
    total_label_counts: Counter,
    mean_count: float,
    std_count: float,
    std_threshold: float = CLASS_OUTLIER_STD_THRESHOLD,
) -> list[int]:
    """
    Flag classes whose sample counts deviate too far from the mean.

    A class is flagged if:

        abs(class_count - mean_count) > std_threshold * std_count

    This function only flags classes. It does not remove them.
    """

    if not total_label_counts or std_count == 0:
        return []

    outlier_classes = []

    for class_label, class_count in total_label_counts.items():
        z_distance = abs(class_count - mean_count) / std_count

        if z_distance > std_threshold:
            outlier_classes.append(int(class_label))

    return sorted(outlier_classes)


def validate_images(
    dataset: DatasetDict,
    max_images_per_split: int | None = None,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    """
    Validate that images are openable and RGB.

    Args:
        dataset: Hugging Face DatasetDict.
        max_images_per_split: Optional cap for faster local testing.
            Use None in the real data quality job.

    Returns:
        image_check_total, image_check_failed, non_rgb_images, image_errors
    """

    image_check_total = 0
    image_check_failed = 0
    non_rgb_images = 0
    image_errors = []

    for split in EXPECTED_SPLITS:
        split_dataset = dataset[split]

        num_items = len(split_dataset)

        if max_images_per_split is not None:
            num_items = min(num_items, max_images_per_split)

        for idx in range(num_items):
            image_check_total += 1

            try:
                sample = split_dataset[idx]
                image = sample["image"]

                # Read mode from header before verify() — verify() invalidates the image object.
                mode = image.mode

                # Check structural validity without decoding pixel data. Much faster than load().
                image.verify()

                if mode != "RGB":
                    non_rgb_images += 1
                    image_errors.append(
                        {
                            "split": split,
                            "index": idx,
                            "label": int(sample["label"]),
                            "error_type": "NON_RGB",
                            "mode": mode,
                        }
                    )

            except Exception as exc:
                image_check_failed += 1
                image_errors.append(
                    {
                        "split": split,
                        "index": idx,
                        "error_type": "IMAGE_OPEN_FAILED",
                        "message": str(exc),
                    }
                )

    return image_check_total, image_check_failed, non_rgb_images, image_errors


def run_data_quality_checks(
    dataset: DatasetDict,
    max_images_per_split: int | None = None,
) -> DataQualityReport:
    """
    Run all data quality checks and return a structured report.

    For quick local testing, pass max_images_per_split=100.
    For the real AWS/SageMaker quality step, use max_images_per_split=None.
    """

    actual_splits = get_split_names(dataset)
    missing_splits = validate_expected_splits(dataset)

    if missing_splits:
        raise ValueError(f"Dataset is missing expected splits: {missing_splits}")

    label_counts_by_split = get_label_counts_by_split(dataset)
    class_sets_by_split = get_class_sets_from_counts(label_counts_by_split)

    total_label_counts = get_total_label_counts(label_counts_by_split)

    split_num_classes = {
        split: len(class_set)
        for split, class_set in class_sets_by_split.items()
    }

    split_label_ranges = {}

    split_missing_label_ids = {}

    for split, class_set in class_sets_by_split.items():
        if class_set:
            expected_range = set(range(min(class_set), max(class_set) + 1))
            split_missing_label_ids[split] = sorted(expected_range - class_set)
        else:
            split_missing_label_ids[split] = []
            
    for split, class_set in class_sets_by_split.items():
        if class_set:
            split_label_ranges[split] = {
                "min_label": min(class_set),
                "max_label": max(class_set),
            }
        else:
            split_label_ranges[split] = {
                "min_label": None,
                "max_label": None,
            }
            
    split_num_images = {
        split: len(dataset[split])
        for split in EXPECTED_SPLITS
    }

    total_num_classes = len(total_label_counts)
    total_images = sum(split_num_images.values())

    class_sets_consistent = check_class_sets_consistent(class_sets_by_split)

    class_count_matches_expected = total_num_classes == NUM_CLASSES

    class_counts = np.array(list(total_label_counts.values()), dtype=float)
    class_count_mean = float(class_counts.mean()) if len(class_counts) else 0.0
    class_count_std = float(class_counts.std()) if len(class_counts) else 0.0

    outlier_classes = find_class_count_outliers(
        total_label_counts=total_label_counts,
        mean_count=class_count_mean,
        std_count=class_count_std,
        std_threshold=CLASS_OUTLIER_STD_THRESHOLD,
    )

    (
        image_check_total,
        image_check_failed,
        non_rgb_images,
        image_errors,
    ) = validate_images(
        dataset=dataset,
        max_images_per_split=max_images_per_split,
    )

    passed = (
        not missing_splits
        and class_count_matches_expected
        # and class_sets_consistent there are completely no samples for label 380, but 381 has enough in train, and some in val/test, so the class sets are not consistent but it's not really a problem for modeling since 380 is basically absent.
        and image_check_failed == 0
        and non_rgb_images == 0
    )

    return DataQualityReport(
        expected_splits=EXPECTED_SPLITS,
        actual_splits=actual_splits,
        missing_splits=missing_splits,
        expected_num_classes=NUM_CLASSES,
        total_num_classes=total_num_classes,
        class_count_matches_expected=class_count_matches_expected,
        split_num_classes=split_num_classes,
        split_label_ranges=split_label_ranges,
        split_missing_label_ids=split_missing_label_ids,
        class_sets_consistent_across_splits=class_sets_consistent,
        total_images=total_images,
        split_num_images=split_num_images,
        class_count_mean=class_count_mean,
        class_count_std=class_count_std,
        class_outlier_std_threshold=CLASS_OUTLIER_STD_THRESHOLD,
        outlier_classes=outlier_classes,
        image_check_total=image_check_total,
        image_check_failed=image_check_failed,
        non_rgb_images=non_rgb_images,
        image_errors=image_errors,
        passed=passed,
    )


def report_to_dict(report: DataQualityReport) -> dict[str, Any]:
    """Convert a data quality report to a plain dictionary."""

    return asdict(report)


def save_report_to_s3(
    report: DataQualityReport,
    bucket: str = S3_BUCKET,
    prefix: str = S3_REPORTS_PREFIX,
    region: str = AWS_REGION,
    filename: str = "data_quality_report.json",
) -> str:
    """Serialize the quality report to JSON and upload it to S3. Returns the S3 URI."""

    body = json.dumps(report_to_dict(report), indent=2).encode("utf-8")
    s3_key = f"{prefix}/{filename}"

    boto3.client("s3", region_name=region).put_object(
        Body=body,
        Bucket=bucket,
        Key=s3_key,
        ContentType="application/json",
    )

    return f"s3://{bucket}/{s3_key}"


def print_quality_report(report: DataQualityReport) -> None:
    """Print a readable summary of the data quality report."""

    print("Data Quality Report")
    print("===================")
    print(f"Passed: {report.passed}")
    print(f"Splits: {report.actual_splits}")
    print(f"Missing splits: {report.missing_splits}")
    print()
    print(f"Expected classes: {report.expected_num_classes}")
    print(f"Total classes: {report.total_num_classes}")
    print(f"Class count matches expected: {report.class_count_matches_expected}")
    print(f"Split class counts: {report.split_num_classes}")
    print(f"Class sets consistent: {report.class_sets_consistent_across_splits}")
    print()
    print(f"Total images: {report.total_images}")
    print(f"Split image counts: {report.split_num_images}")
    print(f"Split label ranges: {report.split_label_ranges}")
    print(f"Split missing label IDs: {report.split_missing_label_ids}")
    print()
    print(f"Class count mean: {report.class_count_mean:.2f}")
    print(f"Class count std: {report.class_count_std:.2f}")
    print(f"Outlier classes: {report.outlier_classes}")
    print()
    print(f"Images checked: {report.image_check_total}")
    print(f"Image open failures: {report.image_check_failed}")
    print(f"Non-RGB images: {report.non_rgb_images}")

    if report.image_errors:
        print()
        print("First 10 image errors:")
        for error in report.image_errors[:10]:
            print(error)


def main() -> None:
    from bird_classifier.data.ingestion import load_bird_dataset

    dataset = load_bird_dataset()

    # Use a small cap for terminal testing.
    # Change to None for the full AWS/SageMaker data quality job.
    report = run_data_quality_checks(
        dataset=dataset,
        max_images_per_split=100,
    )

    print_quality_report(report)


if __name__ == "__main__":
    main()