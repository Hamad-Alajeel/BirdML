"""
Local SageMaker SDK test — runs the Data Quality Processing Job inside Docker.

Prerequisites:
  1. Docker is running locally.
  2. The training image is built:
       docker build -f infrastructure/docker/Dockerfile.training -t bird-ml-training:latest .
  3. AWS credentials are configured locally (used by the container to access S3).
  4. `pip install sagemaker` in your local environment.

Run from the project root:
    python infrastructure/sagemaker/local_test.py
"""

import os
import sagemaker
from sagemaker.local import LocalSession
from sagemaker.processing import ScriptProcessor

IMAGE_URI = "bird-ml-training:latest"
ROLE = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::000000000000:role/local-test")

session = LocalSession()
session.config = {"local": {"local_code": True}}

processor = ScriptProcessor(
    image_uri=IMAGE_URI,
    command=["python"],
    instance_type="ml.m5.large",
    instance_count=1,
    sagemaker_session=session,
    role=ROLE,
)

processor.run(
    code="infrastructure/sagemaker/scripts/run_quality_check.py",
    arguments=["--output-dir", "/opt/ml/processing/output"],
)

print("Local quality check job completed.")
