"""Project-wide configuration: paths, dataset constants, hyperparameter defaults."""

from pathlib import Path

# ─── AWS / S3 ─────────────────────────────────────────────────────────────────

AWS_ACCOUNT_ID = "798092529023"
AWS_REGION     = "us-east-2"
S3_BUCKET      = "bird-ml-halajeel"

S3_DATA_PREFIX     = "data/raw/birds-525"
S3_PARAMS_PREFIX   = "params"
S3_MODELS_PREFIX   = "models"
S3_MLRUNS_PREFIX   = "mlruns"
S3_REPORTS_PREFIX  = "reports"

S3_BEST_PARAMS_KEY = f"{S3_PARAMS_PREFIX}/best_params.json"

# ─── Local paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
DATA_DIR      = PROJECT_ROOT / "data" / "raw" / "birds-525"
MODELS_DIR    = PROJECT_ROOT / "models"
MLRUNS_DIR    = PROJECT_ROOT / "mlruns"

# ─── Dataset constants ────────────────────────────────────────────────────────

NUM_CLASSES   = 526   # 525 unique species; labels 380 and 381 map to the same species (duplicate)
IMAGE_SIZE    = 224
RESIZE_BEFORE_CROP = 256

# ─── Normalization (ImageNet — required for pretrained EfficientNet) ─────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─── Augmentation ─────────────────────────────────────────────────────────────

CROP_SCALE = (0.8, 1.0)
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST   = 0.2

# ─── Training (fixed — never tuned) ───────────────────────────────────────────
#
# These values are inner mechanics of specific optimizers, not worth searching over.
# They only apply when the corresponding optimizer is chosen.

RMS_MOMENTUM  = 0.9    # only used when optimizer == "RMSprop"
RMS_ALPHA     = 0.9    # only used when optimizer == "RMSprop"
RMS_EPS       = 1e-8   # only used when optimizer == "RMSprop"
SGD_MOMENTUM  = 0.9    # only used when optimizer == "SGD"

# StepLR / CosineAnnealing scheduler internals (gamma/T_max derived from epoch counts)
STEPLR_GAMMA = 0.1

# ─── Pretrained weights ───────────────────────────────────────────────────────

EFFICIENTNET_B3_WEIGHTS_FILENAME = "efficientnet_b3_rwightman-b3899882.pth"

# ─── Hyperparameter search space catalog ──────────────────────────────────────
#
# Used by the tuning step. The TuneParams pipeline parameter selects which
# of these to actually search; everything else falls back to the defaults below.

SEARCH_SPACE = {
    "lr_head":           {"low": 1e-4, "high": 1e-2, "log": True},
    "lr_backbone":       {"low": 1e-5, "high": 1e-3, "log": True},
    "n_warmup_epochs":   {"low": 1,    "high": 5,    "type": "int"},
    "n_finetune_epochs": {"low": 5,    "high": 15,   "type": "int"},
    "batch_size":        {"choices": [128]},
    "weight_decay":      {"low": 1e-6, "high": 1e-3, "log": True},
    "optimizer":         {"choices": ["RMSprop", "Adam", "AdamW", "SGD"]},
    "scheduler":         {"choices": ["StepLR", "CosineAnnealingLR"]},
}

DEFAULT_HYPERPARAMS = {
    "lr_head":           1e-3,
    "lr_backbone":       1e-4,
    "n_warmup_epochs":   3,
    "n_finetune_epochs": 12,
    "batch_size":        128,
    "weight_decay":      1e-5,
    "optimizer":         "RMSprop",
    "scheduler":         "StepLR",
}

BEST_PARAMS = {
    "lr_head":           0.0007500045272821452,
    "lr_backbone":       3.987986937649681e-05,
    "n_warmup_epochs":   4,
    "n_finetune_epochs": 11,
    "batch_size":        128,
    "weight_decay":      1e-5,
    "optimizer":         "RMSprop",
    "scheduler":         "StepLR",
}
# ─── Data quality thresholds ──────────────────────────────────────────────────

CLASS_OUTLIER_STD_THRESHOLD = 3.0  # flag classes whose sample count is >Nσ from mean

# ─── Evaluation gate ──────────────────────────────────────────────────────────

MIN_TEST_TOP1_FOR_REGISTRATION = 0.01  # only register model if test top-1 ≥ this
