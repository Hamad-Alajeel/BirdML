# Bird Species Classifier — MLOps Project

## Goal
End-to-end ML pipeline: data ingestion → feature engineering → training/tuning/evaluation → inference → deployment. Showcases MLOps best practices for a portfolio project.

## Dataset
**HuggingFace — 525 Bird Species** (~90k images, 525 classes, pre-split train/val/test, all 224×224 RGB)
- `yashikota/birds-525-species-image-classification`
- Downloaded via `huggingface_hub.snapshot_download`, stored as parquet in `data/raw/birds-525/`

## Pipeline Components

### 1. Data Ingestion
Download and load the dataset. Confirm class counts and split sizes are intact.

### 2. Feature Engineering
- Augmentations: random crop, horizontal flip, color jitter (train only)
- Normalization: scale pixel values, apply mean/std
- PyTorch `DataLoader` with configurable batch size and num_workers

### 3. Training, Tuning & Evaluation
- Transfer learning on a pretrained backbone (TBD)
- Track experiments with **MLflow**
- Hyperparameter search with **Optuna**
- Metrics: Top-1 accuracy, Top-5 accuracy, per-class F1, confusion matrix

### 4. Inference
- Load best model checkpoint
- Apply feature engineering pipeline to input image → return top-N predicted species + confidence scores

### 6. API
**FastAPI** — single `/predict` endpoint accepting an image upload, returning JSON with top predictions.

### 7. Frontend
**Streamlit** — upload an image, display the predicted species and confidence bar chart.

### 8. Deployment
- **Docker** — containerize API and frontend
- **AWS** — host on ECS (Fargate) or EC2 behind a load balancer; model artifacts on S3
- **GitHub Actions** — CI/CD: run tests → build Docker image → push to ECR → deploy

## Development Workflow
1. Prototype each component in a numbered Jupyter notebook (`notebooks/0X_<component>.ipynb`)
2. Once validated, convert to production scripts under `src/bird_classifier/<component>/`
3. Write pytest unit tests in `tests/` for each script
4. Integrate into the full pipeline via `main.py`

## Repo Structure
```
Bird ML/
├── data/               # raw + processed (gitignored)
├── notebooks/          # exploration and prototyping
├── src/bird_classifier/
│   ├── data/           # ingestion + preprocessing
│   ├── training/       # model + trainer
│   ├── evaluation/     # metrics + reporting
│   └── inference/      # predict pipeline
├── api/                # FastAPI app
├── frontend/           # Streamlit app
├── tests/              # pytest
├── models/             # saved checkpoints (gitignored)
├── mlruns/             # MLflow runs (gitignored)
├── .github/workflows/  # CI/CD
├── Dockerfile
└── pyproject.toml
```
