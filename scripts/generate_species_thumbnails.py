"""One-off utility: extract one representative image per bird species from the
HuggingFace 525-species parquet dataset and save them to
``reflex_app/assets/species/<SPECIES_NAME>.jpg``.

The thumbnails power the species cards in the Reflex frontend (both the
top-5 inference results and the catalogue browse page). Run this once after
the dataset is staged in S3; subsequent runs are idempotent (already-saved
species are skipped, so partial / interrupted runs resume cleanly).

Dependencies:
    pip install pyarrow s3fs pillow

Run:
    python scripts/generate_species_thumbnails.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

# --- Configuration --------------------------------------------------------

# Dataset location. The HuggingFace parquet files are mirrored to S3 for this
# project. To run offline, point this at a local directory containing the
# same .parquet files.
DATASET_PATH = "s3://bird-ml-halajeel/data/raw/birds-525/"

# Project paths resolved relative to this file so the script works regardless
# of the cwd it's invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"
OUTPUT_DIR = PROJECT_ROOT / "reflex_app" / "assets" / "species"

# Each thumbnail's longer side is capped at this many pixels. PIL.thumbnail()
# preserves aspect ratio so portraits stay portrait, landscapes stay landscape.
MAX_THUMBNAIL_SIDE = 400

# Pyarrow batch size for streaming — controls memory use vs. throughput.
BATCH_SIZE = 64


# --- Helpers --------------------------------------------------------------

def load_class_names() -> dict[int, str]:
    """Load the index -> species name map produced by training."""
    with CLASS_NAMES_PATH.open() as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def list_parquet_files(path: str) -> list[str]:
    """Return every .parquet file under an S3 prefix or local directory."""
    if path.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem(anon=False)
        return [f"s3://{p}" for p in fs.glob(path.rstrip("/") + "/**/*.parquet")]
    p = Path(path)
    return [str(f) for f in sorted(p.rglob("*.parquet"))]


def extract_image_bytes(image_field) -> bytes | None:
    """Pull raw image bytes from a HuggingFace dataset image field.

    HF image datasets serialize images as a struct ``{bytes, path}``.
    Depending on the pyarrow / dataset version this may surface as a dict,
    a struct, or just raw bytes, so we handle all three.
    """
    if isinstance(image_field, dict):
        return image_field.get("bytes")
    if isinstance(image_field, (bytes, bytearray)):
        return bytes(image_field)
    # Fallback for struct-like objects with a .get() method.
    if hasattr(image_field, "get"):
        return image_field.get("bytes")
    return None


def resolve_species(label, idx_to_name: dict[int, str]) -> str | None:
    """Convert a row's ``label`` value (int or str) into the canonical name."""
    if label is None:
        return None
    if isinstance(label, int):
        return idx_to_name.get(label)
    # The label is already a species name; normalize and validate.
    s = str(label).strip().upper()
    return s if s in set(idx_to_name.values()) else None


# --- Main -----------------------------------------------------------------

def main() -> int:
    idx_to_name = load_class_names()
    needed: set[str] = set(idx_to_name.values())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Skip species we've already saved (idempotent resume).
    already = {p.stem for p in OUTPUT_DIR.glob("*.jpg")}
    needed -= already
    if not needed:
        print(f"All {len(idx_to_name)} species images already present.")
        return 0
    print(f"Need {len(needed)} more species images -> {OUTPUT_DIR}")

    files = list_parquet_files(DATASET_PATH)
    if not files:
        print(f"No parquet files found under {DATASET_PATH}", file=sys.stderr)
        return 1
    print(f"Scanning {len(files)} parquet file(s).")

    saved = 0
    for file_path in files:
        if not needed:
            break
        pqf = pq.ParquetFile(file_path)
        for batch in pqf.iter_batches(batch_size=BATCH_SIZE):
            if not needed:
                break
            for row in batch.to_pylist():
                species = resolve_species(row.get("label"), idx_to_name)
                if not species or species not in needed:
                    continue
                img_bytes = extract_image_bytes(row.get("image"))
                if not img_bytes:
                    continue
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img.thumbnail(
                        (MAX_THUMBNAIL_SIDE, MAX_THUMBNAIL_SIDE),
                        Image.LANCZOS,
                    )
                    out_path = OUTPUT_DIR / f"{species}.jpg"
                    img.save(out_path, "JPEG", quality=85, optimize=True)
                except Exception as e:
                    print(f"Failed to save {species}: {e}", file=sys.stderr)
                    continue
                needed.discard(species)
                saved += 1
                if saved % 25 == 0:
                    print(f"  saved {saved} so far ({len(needed)} remaining)")

    if needed:
        print(
            f"\nWARNING: {len(needed)} species had no image in the dataset:",
            file=sys.stderr,
        )
        for s in sorted(needed):
            print(f"  - {s}", file=sys.stderr)
        return 2

    print(f"\nDone. Saved {saved} new species thumbnails to {OUTPUT_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
