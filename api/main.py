"""FastAPI inference service for the Bird Species Classifier."""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import boto3
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from bird_classifier.config import AWS_REGION, MODELS_DIR, S3_BUCKET, S3_MODELS_PREFIX
from bird_classifier.data.dataset import load_idx_to_name
from bird_classifier.inference.predict import predict
from bird_classifier.training.model import get_device, load_checkpoint

MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "EN_final.pth")

# --- State shared across requests ---
state: dict[str, Any] = {}


def _download_from_s3(filename: str) -> Path:
    """Download a file from S3 to MODELS_DIR if not already present. Returns local path."""
    local_path = MODELS_DIR / filename
    if local_path.exists():
        return local_path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    s3_key = f"{S3_MODELS_PREFIX}/{filename}"
    boto3.client("s3", region_name=AWS_REGION).download_file(S3_BUCKET, s3_key, str(local_path))
    return local_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download model and class names from S3, load into memory at startup."""
    checkpoint_path = _download_from_s3(MODEL_FILENAME)
    class_names_path = _download_from_s3("class_names.json")

    device = get_device()
    model, device = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    state["model"] = model
    state["device"] = device
    state["idx_to_name"] = load_idx_to_name(class_names_path)

    yield

    state.clear()


app = FastAPI(title="Bird Species Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_species(file: UploadFile = File(...), k: int = 5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = predict(
        image=image,
        model=state["model"],
        device=state["device"],
        idx_to_name=state["idx_to_name"],
        k=k,
    )

    return {"predictions": results}
