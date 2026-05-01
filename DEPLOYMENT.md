# Bird ML — End-to-End Deployment Runbook

This runbook takes you from the current state (local code, Dockerfile written, no AWS resources) to a deployed model serving predictions through a public ALB.

Phases are sequential. Do not skip ahead — each phase validates the previous one.

---

## Phase 1 — Validate the local container

**Goal:** Prove the Docker image builds and the quality check job runs end-to-end inside the container.

### 1.1 Build the image (from project root)
```bash
docker build -f infrastructure/docker/Dockerfile.training -t bird-ml-training:latest .
```

### 1.2 Smoke-test imports
```bash
docker run --rm bird-ml-training:latest -c "import bird_classifier; from infrastructure.sagemaker.scripts import _s3_utils; print('OK')"
```
If this fails, the issue is `__init__.py` files or `PYTHONPATH` — fix before continuing.

### 1.3 Run the quality check via SageMaker local mode
The container needs your AWS credentials to read the dataset from S3. SageMaker Local mounts them automatically if `~/.aws/credentials` is configured.
```bash
python infrastructure/sagemaker/local_test.py
```

### 1.4 What "passes" looks like
- Container pulls dataset from S3
- All 526 classes verified
- Image validation runs to completion (this can take a few minutes — it opens every image)
- `data_quality_report.json` is uploaded to `s3://bird-ml-halajeel/reports/`
- Job exits with code 0

If image validation is too slow for iteration, temporarily pass `--max-images-per-split 100` in `local_test.py`'s `arguments=[]` list. Set it back to no cap before running on AWS.

---

## Phase 2 — AWS account setup

**Goal:** Have all AWS resources the pipeline depends on before writing the pipeline definition.

### 2.1 IAM role for SageMaker
Create role `BirdMLSageMakerRole` with these AWS-managed policies attached:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess` (or scope to `bird-ml-halajeel` bucket)
- `AmazonEC2ContainerRegistryFullAccess` (for ECR pulls during jobs)

Trust policy: allow `sagemaker.amazonaws.com` to assume the role.

Save the role ARN — you'll need it everywhere. Recommend exporting:
```bash
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<your-account-id>:role/BirdMLSageMakerRole
```

### 2.2 ECR repository
```bash
aws ecr create-repository --repository-name bird-ml-training --region us-east-2
```

Note the repo URI: `<account-id>.dkr.ecr.us-east-2.amazonaws.com/bird-ml-training`

### 2.3 Upload pretrained EfficientNet-B3 weights to S3
The container does not have these weights baked in. They live in S3 and the entry point scripts download them at job start.

Locally (one time):
```python
import torch
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# Download weights
weights = EfficientNet_B3_Weights.IMAGENET1K_V1
model = efficientnet_b3(weights=weights)
torch.save(model.state_dict(), "efficientnet_b3_rwightman-b3899882.pth")
```

Upload:
```bash
aws s3 cp efficientnet_b3_rwightman-b3899882.pth s3://bird-ml-halajeel/models/pretrained/efficientnet_b3_rwightman-b3899882.pth
```

### 2.4 Verify S3 bucket layout
After Phase 2 you should see:
```
s3://bird-ml-halajeel/
├── data/raw/birds-525/         (already exists from earlier work)
├── models/pretrained/          (new — pretrained EfficientNet weights)
├── params/                     (will be populated by tune job)
├── reports/                    (already populated by quality job)
└── mlruns/                     (will be populated by training jobs)
```

---

## Phase 3 — Push image to ECR

```bash
# Authenticate Docker with ECR
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-2.amazonaws.com

# Tag and push
docker tag bird-ml-training:latest <account-id>.dkr.ecr.us-east-2.amazonaws.com/bird-ml-training:latest
docker push <account-id>.dkr.ecr.us-east-2.amazonaws.com/bird-ml-training:latest
```

Verify the image is in ECR via the AWS Console.

---

## Phase 4 — Code changes for production

The entry point scripts currently expect pretrained weights at `MODELS_DIR/efficientnet_b3_rwightman-b3899882.pth`. They need to download from S3 at job start.

### 4.1 Add a weights-download helper
In `src/bird_classifier/training/model.py`, add a function (above `build_model`):

```python
import boto3
from bird_classifier.config import AWS_REGION, S3_BUCKET

def ensure_pretrained_weights(weights_path: Path | None = None) -> Path:
    """Download EfficientNet-B3 pretrained weights from S3 if not present locally."""
    if weights_path is None:
        weights_path = MODELS_DIR / EFFICIENTNET_B3_WEIGHTS_FILENAME

    if weights_path.exists():
        return weights_path

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    s3_key = f"models/pretrained/{EFFICIENTNET_B3_WEIGHTS_FILENAME}"
    boto3.client("s3", region_name=AWS_REGION).download_file(
        S3_BUCKET, s3_key, str(weights_path)
    )
    return weights_path
```

Update `build_model` to call it:
```python
def build_model(num_classes=NUM_CLASSES, weights_path=None):
    weights_path = ensure_pretrained_weights(weights_path)
    # ... rest unchanged
```

### 4.2 Test locally
Re-run `python infrastructure/sagemaker/local_test.py` to confirm nothing broke. Then test with the train script (locally — caps batches/epochs for speed):

Create a one-off test in `local_test.py` after the quality check works:
```python
processor.run(
    code="infrastructure/sagemaker/scripts/run_train.py",
    arguments=[],
)
```
This is going to be slow on CPU. Just verify the first epoch starts correctly, then kill it.

### 4.3 Rebuild and re-push the image
```bash
docker build -f infrastructure/docker/Dockerfile.training -t bird-ml-training:latest .
docker tag bird-ml-training:latest <account-id>.dkr.ecr.us-east-2.amazonaws.com/bird-ml-training:latest
docker push <account-id>.dkr.ecr.us-east-2.amazonaws.com/bird-ml-training:latest
```

---

## Phase 5 — SageMaker Pipeline definition

**Goal:** Wire the four jobs (quality → tune?/skip → train → evaluate → register) into a SageMaker Pipeline.

### 5.1 Create `infrastructure/sagemaker/pipeline.py`
This is the file that defines the pipeline DAG. Skeleton:

```python
"""SageMaker Pipeline definition for the Bird Species Classifier."""

import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.workflow.parameters import ParameterBoolean, ParameterString, ParameterInteger
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model

REGION = "us-east-2"
ROLE_ARN = "<paste your role ARN here or load from env>"
IMAGE_URI = "<your ECR URI>:latest"
PIPELINE_NAME = "bird-ml-training"
INSTANCE_TYPE_GPU = "ml.g5.xlarge"
INSTANCE_TYPE_CPU = "ml.m5.large"


def build_pipeline() -> Pipeline:
    session = sagemaker.Session()

    # Pipeline parameters (overridable at trigger time)
    retune = ParameterBoolean("RetuneHyperparameters", default_value=False)
    tune_params = ParameterString(
        "TuneParams", default_value="lr_head,lr_backbone,n_warmup_epochs"
    )
    tune_trials = ParameterInteger("TuneTrials", default_value=30)

    # Step 1: Quality check
    quality_processor = ScriptProcessor(
        image_uri=IMAGE_URI,
        command=["python"],
        instance_type=INSTANCE_TYPE_CPU,
        instance_count=1,
        role=ROLE_ARN,
    )
    quality_step = ProcessingStep(
        name="DataQualityCheck",
        processor=quality_processor,
        code="infrastructure/sagemaker/scripts/run_quality_check.py",
        outputs=[ProcessingOutput(source="/opt/ml/processing/output", output_name="report")],
    )

    # Step 2 + 3a: Conditional tuning step
    tune_estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE_ARN,
        instance_count=1,
        instance_type=INSTANCE_TYPE_GPU,
        entry_point="infrastructure/sagemaker/scripts/run_tune.py",
    )
    tune_step = TrainingStep(
        name="HyperparameterTune",
        estimator=tune_estimator,
        # pass tune_params and tune_trials as hyperparameters or env vars
    )

    condition_step = ConditionStep(
        name="ShouldRetune",
        conditions=[ConditionEquals(left=retune, right=True)],
        if_steps=[tune_step],
        else_steps=[],
    )

    # Step 4: Final training (always runs, depends on condition_step)
    train_estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE_ARN,
        instance_count=1,
        instance_type=INSTANCE_TYPE_GPU,
        entry_point="infrastructure/sagemaker/scripts/run_train.py",
    )
    train_step = TrainingStep(
        name="FinalTraining",
        estimator=train_estimator,
        depends_on=[condition_step],
    )

    # Step 5: Evaluation
    eval_processor = ScriptProcessor(
        image_uri=IMAGE_URI,
        command=["python"],
        instance_type=INSTANCE_TYPE_GPU,  # eval uses GPU for inference speed
        instance_count=1,
        role=ROLE_ARN,
    )
    eval_step = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        code="infrastructure/sagemaker/scripts/run_evaluate.py",
        depends_on=[train_step],
    )

    # Step 6: Register in Model Registry (PendingManualApproval)
    model = Model(
        image_uri=IMAGE_URI,
        model_data=f"s3://bird-ml-halajeel/models/EN_final.pth",
        role=ROLE_ARN,
    )
    register_step = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["application/x-image"],
            response_types=["application/json"],
            inference_instances=[INSTANCE_TYPE_CPU],
            transform_instances=[INSTANCE_TYPE_CPU],
            model_package_group_name="BirdMLModelGroup",
            approval_status="PendingManualApproval",
        ),
        depends_on=[eval_step],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[retune, tune_params, tune_trials],
        steps=[quality_step, condition_step, train_step, eval_step, register_step],
        sagemaker_session=session,
    )


if __name__ == "__main__":
    pipeline = build_pipeline()
    pipeline.upsert(role_arn=ROLE_ARN)
    print(f"Pipeline '{PIPELINE_NAME}' upserted.")
```

This is a skeleton — refine the specifics (especially how `tune_params`/`tune_trials` are passed to entry points; SageMaker uses estimator `hyperparameters={}` for this) before running.

### 5.2 Upload the pipeline definition
```bash
python infrastructure/sagemaker/pipeline.py
```
This calls `pipeline.upsert()` which registers the pipeline in SageMaker.

### 5.3 Create the Model Package Group (one time)
In AWS Console or via boto3:
```python
import boto3
boto3.client("sagemaker").create_model_package_group(
    ModelPackageGroupName="BirdMLModelGroup",
    ModelPackageGroupDescription="Bird species classifier — EfficientNet-B3"
)
```

---

## Phase 6 — Trigger and run the pipeline

### 6.1 Manual trigger from CLI
```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name bird-ml-training \
  --pipeline-parameters Name=RetuneHyperparameters,Value=true \
                        Name=TuneTrials,Value=10
```
For the first run, set trials low (10) to verify the pipeline shape works before paying for 30 full GPU trials.

### 6.2 Watch the pipeline execute
SageMaker Studio → Pipelines → bird-ml-training → latest execution. Each step shows logs in CloudWatch.

### 6.3 What to verify
- Quality check writes report to S3
- (If retune=true) Tuning produces `best_params.json` and two checkpoints in S3
- Final training produces `EN_final.pth`
- Evaluation passes the threshold gate
- Model appears in Model Registry as `PendingManualApproval`

---

## Phase 7 — Manual approval

In SageMaker Studio → Model Registry → BirdMLModelGroup → latest version:
1. Inspect metrics (top-1, top-5, loss)
2. Inspect lineage (which pipeline run produced it)
3. Click Approve

This sets the model status to `Approved`. The deployment workflow (Phase 10) listens for this event.

---

## Phase 8 — Build inference containers

You need two new containers: API and Frontend.

### 8.1 FastAPI service (`api/`)
Create:
- `api/main.py` — FastAPI app with `/predict` endpoint that loads `EN_final.pth` from S3 (or mounts it from a model channel) and runs inference on uploaded images.
- `api/requirements.txt` — fastapi, uvicorn, pillow, torch, torchvision, boto3, bird_classifier
- `api/Dockerfile` — `FROM python:3.11-slim`, install deps, expose port 8000, `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]`

The API should reuse `bird_classifier.inference.predict` (you'll need to create `src/bird_classifier/inference/predict.py` and `preprocessor.py` — these are still empty per the CLAUDE.md repo structure).

### 8.2 Streamlit frontend (`frontend/`)
Create:
- `frontend/app.py` — Streamlit UI, file uploader, calls the API URL (read from env var `API_URL`)
- `frontend/requirements.txt` — streamlit, requests, pillow
- `frontend/Dockerfile` — `FROM python:3.11-slim`, install deps, expose 8501, `CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]`

### 8.3 Build and push both images
```bash
# Create ECR repos
aws ecr create-repository --repository-name bird-ml-api --region us-east-2
aws ecr create-repository --repository-name bird-ml-frontend --region us-east-2

# API
docker build -t bird-ml-api:latest api/
docker tag bird-ml-api:latest <account-id>.dkr.ecr.us-east-2.amazonaws.com/bird-ml-api:latest
docker push <account-id>.dkr.ecr.us-east-2.amazonaws.com/bird-ml-api:latest

# Frontend
docker build -t bird-ml-frontend:latest frontend/
docker tag bird-ml-frontend:latest <account-id>.dkr.ecr.us-east-2.amazonaws.com/bird-ml-frontend:latest
docker push <account-id>.dkr.ecr.us-east-2.amazonaws.com/bird-ml-frontend:latest
```

---

## Phase 9 — ECS Fargate deployment

This is the most complex AWS setup. Doing it via the console is fine for a first pass; switch to Terraform later if you want IaC.

### 9.1 Networking
- Use the default VPC (or create one — at least 2 public subnets across AZs)
- Security group `bird-ml-sg`: allow inbound 80/443 from 0.0.0.0/0, allow internal traffic between ECS tasks

### 9.2 ECS cluster
```bash
aws ecs create-cluster --cluster-name bird-ml-cluster --capacity-providers FARGATE
```

### 9.3 Task definitions (one per service)
For each (api, frontend) create a task definition:
- Launch type: Fargate
- Task role: IAM role with S3 read access (only API needs this — to pull the model)
- Execution role: `ecsTaskExecutionRole` (AWS-managed)
- Container image: ECR URI from Phase 8
- Port mappings: 8000 for API, 8501 for frontend
- Environment variables: API needs `MODEL_S3_KEY`; frontend needs `API_URL`
- CPU/memory: 1 vCPU / 2 GB for API, 0.5 vCPU / 1 GB for frontend (adjust as needed)

### 9.4 Application Load Balancer
- ALB `bird-ml-alb`, internet-facing, in the public subnets
- Two target groups: `bird-ml-api-tg` (port 8000), `bird-ml-frontend-tg` (port 8501), both type `ip` (Fargate requires this)
- Listener on port 80:
  - Default action → forward to frontend target group
  - Rule: path `/api/*` → forward to API target group

### 9.5 ECS services
```bash
# API service
aws ecs create-service \
  --cluster bird-ml-cluster \
  --service-name bird-ml-api \
  --task-definition bird-ml-api:1 \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:...:targetgroup/bird-ml-api-tg/...,containerName=api,containerPort=8000"

# Frontend service (same shape, different target group + container)
```

### 9.6 Verify
Get the ALB DNS name from the AWS Console, open it in a browser. The Streamlit app loads, you upload a bird photo, the frontend calls `/api/predict`, and a prediction appears.

---

## Phase 10 — GitHub Actions automation

Two workflows.

### 10.1 `.github/workflows/training.yml`
Triggers on push to `main`. Builds the training image, pushes to ECR, triggers the SageMaker pipeline:
```yaml
name: Trigger Training Pipeline
on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      retune:
        description: 'Re-run hyperparameter tuning'
        required: true
        default: 'false'
jobs:
  trigger:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::<account>:role/GitHubActionsBirdMLRole
          aws-region: us-east-2
      - uses: aws-actions/amazon-ecr-login@v2
      - run: |
          docker build -f infrastructure/docker/Dockerfile.training -t $ECR_URI:$GITHUB_SHA .
          docker push $ECR_URI:$GITHUB_SHA
          aws sagemaker start-pipeline-execution \
            --pipeline-name bird-ml-training \
            --pipeline-parameters Name=RetuneHyperparameters,Value=${{ inputs.retune }}
```
You'll need to set up an OIDC trust between GitHub and AWS for the assume-role to work (or use access keys as repo secrets — less ideal).

### 10.2 `.github/workflows/deployment.yml`
Triggered by Model Registry approval event. Approval events can be routed via EventBridge → SNS → GitHub repository_dispatch. Setup:
1. EventBridge rule: source `aws.sagemaker`, detail-type `SageMaker Model Package State Change`, status `Approved`
2. EventBridge target: a Lambda that calls GitHub's repository_dispatch API
3. Workflow listens for `repository_dispatch` event, builds api+frontend images, updates ECS services

This is the most fragile bit. As an alternative for the portfolio, you can manually trigger deployment with `workflow_dispatch` and skip the EventBridge plumbing.

---

## Files still to be created

Track these as TODOs — they don't exist yet:

- [ ] `src/bird_classifier/inference/preprocessor.py` (empty)
- [ ] `src/bird_classifier/inference/predict.py` (empty)
- [ ] `infrastructure/sagemaker/pipeline.py` (Phase 5)
- [ ] `api/main.py` + `api/Dockerfile` + `api/requirements.txt` (Phase 8)
- [ ] `frontend/app.py` + `frontend/Dockerfile` + `frontend/requirements.txt` (Phase 8)
- [ ] `.github/workflows/training.yml` (Phase 10)
- [ ] `.github/workflows/deployment.yml` (Phase 10)

## Code changes still needed

- [ ] `src/bird_classifier/training/model.py`: add `ensure_pretrained_weights()` and call it from `build_model` (Phase 4.1)

## Tests not yet written (per CLAUDE.md spec)

- [ ] `tests/` — pytest tests for each module under `src/`. Per CLAUDE.md step 3 of dev workflow, you should write these as you port each notebook. They were skipped during the rush to ship — backfill at least the data layer and engine.py before the pipeline runs on AWS, since debugging in a SageMaker job is much slower than locally.

---

## What to do if you get stuck

1. **Container build fails:** Check Docker has enough memory (8GB+). The PyTorch base image is large.
2. **SageMaker job fails to pull image:** ECR auth — the SageMaker role needs `AmazonEC2ContainerRegistryReadOnly`.
3. **Quality check fails on AWS but works locally:** S3 region mismatch. The bucket is in `us-east-2`; jobs must run there too.
4. **Pipeline upsert fails:** Usually missing IAM permissions on the role you're running it from (the local user, not the SageMaker role).
5. **Eval gate trips with low accuracy:** Lower `MIN_TEST_TOP1_FOR_REGISTRATION` in `config.py` temporarily to push through.
6. **Manual approval doesn't trigger deploy:** Skip EventBridge and use `workflow_dispatch` manually.

---

## What "done" looks like

You can:
1. Push code to `main`
2. The training pipeline runs on AWS
3. You approve the model in SageMaker Studio
4. The deployment workflow runs (or you trigger it manually)
5. You hit the ALB URL in a browser, upload a bird photo, get a prediction back
