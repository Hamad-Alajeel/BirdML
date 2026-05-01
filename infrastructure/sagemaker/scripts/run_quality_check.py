"""
Entry point for the SageMaker Data Quality Processing Job.

SageMaker executes this script directly inside the container. It runs top-to-bottom:
  1. Sync the raw dataset from S3 to the container.
  2. Load all three splits (train / validation / test).
  3. Run every data quality check.
  4. Write the JSON report to the SageMaker output path so SageMaker uploads it to S3.
  5. Upload the report to S3 via the quality module (separate from the SageMaker output).
  6. Exit non-zero if any check failed — this causes SageMaker to mark the job as Failed
     and stops the downstream pipeline steps from running.
"""

import argparse
import json
import sys
from pathlib import Path

from bird_classifier.data.ingestion import load_bird_dataset
from bird_classifier.data.quality import (
    print_quality_report,
    report_to_dict,
    run_data_quality_checks,
    save_report_to_s3,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bird classifier data quality job.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/opt/ml/processing/output"),
        help="Local path where SageMaker expects output artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1/4: Syncing dataset from S3 and loading splits ===")
    dataset = load_bird_dataset()

    print("=== Step 2/4: Running data quality checks ===")
    report = run_data_quality_checks(dataset=dataset, max_images_per_split=None)
    print_quality_report(report)

    print("=== Step 3/4: Writing report to SageMaker output path ===")
    report_path = args.output_dir / "data_quality_report.json"
    report_path.write_text(json.dumps(report_to_dict(report), indent=2))
    print(f"Report written to {report_path}")

    print("=== Step 4/4: Uploading report to S3 ===")
    s3_uri = save_report_to_s3(report)
    print(f"Report uploaded to {s3_uri}")

    if not report.passed:
        print("\nERROR: Data quality checks failed — aborting pipeline.")
        sys.exit(1)

    print("\nData quality checks passed.")


if __name__ == "__main__":
    main()
