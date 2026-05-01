# Bird Species Classifier — MLOps Project

## Goal
End-to-end ML pipeline: data ingestion → feature engineering → training/tuning/evaluation → inference → deployment. Fully automated on AWS via SageMaker Pipelines + GitHub Actions, with a manual approval gate before deployment. Designed to showcase production MLOps best practices for a portfolio project.

## Dataset
**HuggingFace — 525 Bird Species** (~90k images, 526 unique classes, pre-split train/val/test, all 224×224 RGB)
- `yashikota/birds-525-species-image-classification`
- Stored as parquet in `s3://bird-ml-halajeel/data/raw/birds-525/`

## Model
**EfficientNet-B3** (ImageNet pretrained), fine-tuned in two stages:
1. **Warmup** — backbone frozen, only the new 526-class head trains
2. **Fine-tune** — entire network unfrozen, separate learning rates for head and backbone

## Tech Stack

| Concern | Choice |
|---|---|
| Framework | PyTorch + torchvision |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow (logs to S3, viewed locally — no hosted server) |
| Pipeline orchestration | SageMaker Pipelines |
| Compute | SageMaker Training/Processing Jobs (GPU: ml.g5.xlarge) |
| Model registry | SageMaker Model Registry (with manual approval) |
| Container registry | ECR |
| Inference hosting | ECS Fargate behind ALB |
| Frontend | Streamlit on ECS Fargate |
| CI/CD | GitHub Actions |
| IaC | (TBD — Terraform if added) |

## End-to-End Pipeline

```
GitHub push (or manual trigger)
    │
    ▼
[GitHub Actions: trigger SageMaker Pipeline]
    parameters:
      RetuneHyperparameters: bool   (default: False)
      TuneParams: string             (default: "lr_head,lr_backbone,n_warmup_epochs")
      TuneTrials: int                (default: 30)
    │
    ▼
1. [Data Quality Processing Job]
   ├─ Sync dataset from S3
   ├─ Verify class count == 526
   ├─ Flag classes with sample count > 3σ from mean
   ├─ Validate every image is RGB and openable
   └─ Output report to S3
    │
    ▼
2. [ConditionStep: should we run tuning?]
   ├─ Yes if RetuneHyperparameters=True OR no params in S3 → 3a
   └─ No otherwise → skip to 4
    │
3a. [Tuning Job (SageMaker Training Job)]
    ├─ Read TuneParams pipeline parameter
    ├─ Sample only those hyperparameters via Optuna (others use defaults)
    ├─ Run TuneTrials Optuna trials
    ├─ Log every trial to MLflow (saved to S3 at end of job)
    └─ Save best params → s3://bird-ml-halajeel/params/best_params.json
    │
    ▼
4. [Final Training Job (SageMaker Training Job)]
   ├─ Read best_params.json from S3
   ├─ Train on full dataset for 15 epochs
   ├─ Log to MLflow (saved to S3 at end of job)
   └─ Save final model checkpoint to S3
    │
    ▼
5. [Evaluation Job (SageMaker Processing Job)]
   ├─ Load trained model
   ├─ Run on held-out test set
   ├─ Generate metrics JSON, confusion matrix, classification report
   └─ Conditional check: only proceed if test top-1 > threshold
    │
    ▼
6. [Register Model in SageMaker Model Registry]
   └─ Status: PendingManualApproval
    │
    ▼
👤 [Manual review in SageMaker Studio]
   ├─ Inspect metrics, lineage, artifacts
   ├─ Compare against previous versions
   └─ Approve OR Reject
    │
    ▼ (on approval)
7. [Deployment GitHub Actions workflow — triggered by Model Registry approval event]
   ├─ Build API Docker image → push to ECR
   ├─ Build Frontend Docker image → push to ECR
   ├─ Update ECS Fargate services (rolling deploy)
   └─ Both behind ALB → public URL
```

## Tunable Hyperparameter Search Space

The tuning script defines a fixed catalog. The `TuneParams` pipeline parameter selects which to actually search:

| Parameter | Default range | Default value (when not tuned) |
|---|---|---|
| `lr_head` | 1e-4 to 1e-2 (log) | 1e-3 |
| `lr_backbone` | 1e-5 to 1e-3 (log) | 1e-4 |
| `n_warmup_epochs` | 1 to 5 (int) | 3 |
| `batch_size` | choices [128] | 512 |
| `weight_decay` | 1e-6 to 1e-3 (log) | 1e-5 |
| `scheduler` | choices [StepLR, CosineAnnealingLR] | StepLR |

Fixed (not tuneable):
- Optimizer: RMSprop (momentum=0.9, alpha=0.9, eps=1e-8)
- Total epochs: 15

## Pipeline Trigger Decisions

| Decision Point | Who | Where |
|---|---|---|
| `RetuneHyperparameters` | You | At pipeline trigger (Studio UI / CLI / GH Actions input) |
| `TuneParams` | You | At pipeline trigger |
| `TuneTrials` | You | At pipeline trigger |
| Approve / Reject model | You | SageMaker Studio Model Registry, after eval |

## Development Workflow
1. Prototype each component in a numbered Jupyter notebook (`notebooks/d2_0X_<component>.ipynb`)
2. Once validated, port to production scripts under `src/bird_classifier/<component>/`
3. Write pytest unit tests in `tests/` for each script
4. Containerize and wire into the SageMaker Pipeline definition

## Repo Structure
```
Bird ML/
├── data/                          # raw + processed (gitignored)
├── notebooks/                     # exploration and prototyping
├── src/bird_classifier/
│   ├── config.py                  # constants, paths, defaults
│   ├── data/
│   │   ├── ingestion.py           # S3 sync, parquet load
│   │   ├── quality.py             # data quality checks
│   │   ├── transforms.py          # train/eval transforms
│   │   ├── dataset.py             # BirdDataset, label remap
│   │   └── dataloaders.py         # DataLoader factory
│   ├── training/
│   │   ├── model.py               # EfficientNet-B3 builder
│   │   ├── engine.py              # run_epoch, top_k_accuracy
│   │   ├── tune.py                # Optuna tuning entry point
│   │   └── train.py               # Final training entry point
│   ├── evaluation/
│   │   ├── metrics.py             # accuracy, F1, confusion matrix
│   │   └── evaluate.py            # Evaluation entry point
│   └── inference/
│       ├── preprocessor.py        # Single-image preprocessing
│       └── predict.py             # Inference entry point
├── api/                           # FastAPI app (wraps inference/predict.py)
├── frontend/                      # Streamlit app
├── tests/                         # pytest unit + integration tests
├── infrastructure/
│   ├── docker/                    # Dockerfiles (api, frontend, training)
│   └── sagemaker/                 # Pipeline definition + step scripts
├── .github/workflows/             # CI/CD: training pipeline trigger, deployment
├── models/                        # local checkpoints (gitignored)
├── mlruns/                        # local MLflow logs (gitignored, synced to S3)
└── pyproject.toml
```

## Implementation Order
1. **Refactor notebooks → `src/`** with unit tests
2. **Containerize training** (Dockerfile, run via SageMaker SDK locally first)
3. **Build SageMaker Pipeline definition** with all steps + ConditionStep + Model Registry
4. **GitHub Actions workflow** to trigger the training pipeline
5. **Manual approval flow** in Model Registry — verify it works end-to-end
6. **API + Frontend** (FastAPI + Streamlit)
7. **GitHub Actions workflow** for deployment, triggered by Model Registry approval event
8. **End-to-end smoke test** — push code, run pipeline, approve, verify live URL serves predictions
