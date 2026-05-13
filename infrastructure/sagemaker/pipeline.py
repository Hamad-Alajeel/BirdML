"""SageMaker Pipeline definition for the Bird Species Classifier."""

import argparse
import logging

# Python 3.13 changed pathlib.Path.exists() error handling in a way that breaks
# the SageMaker SDK's studio config lookup. This patch skips the lookup entirely.
import sagemaker._studio as _studio
_studio._find_config = lambda working_dir: None

from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep

from bird_classifier.config import AWS_ACCOUNT_ID, AWS_REGION, S3_BUCKET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROLE_ARN = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/AmazonSageMakerUserIAMExecutionRole_1ab88b8d"
IMAGE_URI = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/bird-ml-training:latest"
PIPELINE_NAME = "bird-ml-training"


def build_pipeline(session: PipelineSession | None = None) -> Pipeline:
    if session is None:
        session = PipelineSession()

    # ── Pipeline parameters ────────────────────────────────────────────────────
    # All overridable at trigger time. Strings are used throughout because
    # SageMaker ProcessingStep job_arguments only accepts string values.
    retune = ParameterString("RetuneHyperparameters", default_value="false")
    skip_training = ParameterString("SkipTraining", default_value="false")
    tune_params = ParameterString(
        "TuneParams", default_value="lr_head,lr_backbone,n_warmup_epochs"
    )
    tune_trials = ParameterString("TuneTrials", default_value="30")

    # ── Shared processor definitions ───────────────────────────────────────────
    cpu_processor = ScriptProcessor(
        image_uri=IMAGE_URI,
        command=["python"],
        instance_type="ml.m5.large",
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=session,
    )

    gpu_processor = ScriptProcessor(
        image_uri=IMAGE_URI,
        command=["python"],
        instance_type="ml.g5.xlarge",
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=session,
    )

    # ── Step 1: Data Quality Check ─────────────────────────────────────────────
    # Validates the dataset before wasting GPU time on a bad dataset.
    quality_step = ProcessingStep(
        name="DataQualityCheck",
        processor=cpu_processor,
        code="infrastructure/sagemaker/scripts/run_quality_check.py",
    )

    # ── Step 2: Hyperparameter Tuning ─────────────────────────────────────────
    # Always runs but exits early inside run_tune.py when RetuneHyperparameters=false.
    # When enabled, searches the hyperparameter space and writes best_params.json to S3.
    tune_step = ProcessingStep(
        name="HyperparameterTuning",
        processor=gpu_processor,
        code="infrastructure/sagemaker/scripts/run_tune.py",
        job_arguments=[
            "--retune", retune,
            "--n-trials", tune_trials,
            "--tune-params", tune_params,
        ],
        depends_on=[quality_step],
    )

    # ── Step 3: Final Training ─────────────────────────────────────────────────
    # Always runs unless SkipTraining=true. Reads best_params.json from S3 if
    # it exists, otherwise falls back to BEST_PARAMS in config.
    train_step = ProcessingStep(
        name="FinalTraining",
        processor=gpu_processor,
        code="infrastructure/sagemaker/scripts/run_train.py",
        job_arguments=["--skip-training", skip_training],
        depends_on=[tune_step],
    )

    # ── Step 4: Evaluation ────────────────────────────────────────────────────
    # Exits non-zero if top-1 accuracy is below MIN_TEST_TOP1_FOR_REGISTRATION,
    # stopping the pipeline before registering a substandard model.
    eval_step = ProcessingStep(
        name="EvaluateModel",
        processor=gpu_processor,
        code="infrastructure/sagemaker/scripts/run_evaluate.py",
        depends_on=[train_step],
    )

    # ── Step 5: Register Model ────────────────────────────────────────────────
    # Registers the model as PendingManualApproval. A human approves or rejects
    # in SageMaker Studio before deployment proceeds.
    model = Model(
        image_uri=IMAGE_URI,
        model_data=f"s3://{S3_BUCKET}/models/EN_final.pth",
        role=ROLE_ARN,
        sagemaker_session=session,
    )

    register_step = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["image/jpeg", "image/png"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name="BirdMLModelGroup",
            approval_status="PendingManualApproval",
        ),
        depends_on=[eval_step],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[retune, skip_training, tune_params, tune_trials],
        steps=[quality_step, tune_step, train_step, eval_step, register_step],
        sagemaker_session=session,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upsert the SageMaker training pipeline and optionally start an execution."
    )
    parser.add_argument("--retune", default="false")
    parser.add_argument("--skip-training", default="false")
    parser.add_argument("--tune-trials", default="30")
    parser.add_argument("--tune-params", default="lr_head,lr_backbone,n_warmup_epochs")
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Only upsert the pipeline definition; do not trigger an execution.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pipeline = build_pipeline()
    pipeline.upsert(role_arn=ROLE_ARN)
    logger.info("Pipeline '%s' upserted.", PIPELINE_NAME)

    if args.no_start:
        logger.info("--no-start set; skipping pipeline execution.")
    else:
        execution = pipeline.start(parameters={
            "RetuneHyperparameters": args.retune,
            "SkipTraining":          args.skip_training,
            "TuneTrials":            args.tune_trials,
            "TuneParams":            args.tune_params,
        })
        logger.info("Pipeline execution started: %s", execution.arn)
