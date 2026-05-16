"""Shared app state.

Lives in its own module so every page and component can import the same
State subclass without circular imports back through reflex_app.py.
"""

import base64
import json
import os
from pathlib import Path

import httpx
import reflex as rx

# Endpoint of the FastAPI inference service. Default points at a locally
# exposed bird-api container; in Docker the Reflex container reaches the
# host (and the sibling API container) via host.docker.internal.
BIRD_API_URL = os.environ.get("BIRD_API_URL", "http://localhost:8000")

# Load every species description into memory once at import time. The
# files live in reflex_app/assets/species_data/<NAME>.json and are tiny
# JSON blobs, so 525 reads at startup is cheap and avoids per-request IO.
_SPECIES_DATA_DIR = Path(__file__).parent.parent / "assets" / "species_data"


def _load_descriptions() -> dict[str, str]:
    out: dict[str, str] = {}
    if not _SPECIES_DATA_DIR.exists():
        return out
    for path in _SPECIES_DATA_DIR.glob("*.json"):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            out[blob["name"]] = blob.get("description", "")
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return out


_DESCRIPTIONS = _load_descriptions()


class State(rx.State):
    """Tracks app phase, the uploaded image, and carousel navigation."""

    phase: str = "landing"  # "landing" | "processing" | "results"
    img_data_uri: str = ""

    current_card: int = 0

    # Populated from /predict response on each inference run. Defaults show
    # placeholder content so the UI still renders if a developer opens the
    # results phase without an inference cycle (e.g. via dev tools).
    card_names: list[str] = [
        "ABBOTTS BABBLER",
        "ABBOTTS BOOBY",
        "ABYSSINIAN GROUND HORNBILL",
        "AFRICAN CROWNED CRANE",
        "AFRICAN EMERALD CUCKOO",
    ]
    card_confidences: list[float] = [0.95, 0.83, 0.67, 0.45, 0.31]
    card_descriptions: list[str] = [
        _DESCRIPTIONS.get("ABBOTTS BABBLER", ""),
        _DESCRIPTIONS.get("ABBOTTS BOOBY", ""),
        _DESCRIPTIONS.get("ABYSSINIAN GROUND HORNBILL", ""),
        _DESCRIPTIONS.get("AFRICAN CROWNED CRANE", ""),
        _DESCRIPTIONS.get("AFRICAN EMERALD CUCKOO", ""),
    ]

    inference_error: str = ""

    @rx.var
    def has_image(self) -> bool:
        return self.img_data_uri != ""

    @rx.var
    def is_landing(self) -> bool:
        return self.phase == "landing"

    @rx.var
    def is_processing(self) -> bool:
        return self.phase == "processing"

    @rx.var
    def is_results(self) -> bool:
        return self.phase == "results"

    @rx.var
    def has_error(self) -> bool:
        return self.inference_error != ""

    @rx.var
    def num_cards(self) -> int:
        return len(self.card_names)

    @rx.var
    def position_label(self) -> str:
        return f"{self.current_card + 1} / {self.num_cards}"

    @rx.var
    def current_card_name(self) -> str:
        return self.card_names[self.current_card]

    @rx.var
    def current_card_image(self) -> str:
        # Each species thumbnail is saved at assets/species/<NAME>.jpg by the
        # generate_species_thumbnails.py script, so the URL is derived from
        # the predicted species name directly.
        return f"/species/{self.card_names[self.current_card]}.jpg"

    @rx.var
    def current_card_description(self) -> str:
        return self.card_descriptions[self.current_card]

    @rx.var
    def current_card_confidence_pct(self) -> str:
        return f"{self.card_confidences[self.current_card] * 100:.1f}%"

    @rx.var
    def current_card_confidence_bar(self) -> int:
        return int(self.card_confidences[self.current_card] * 100)

    @rx.var
    def current_rank(self) -> str:
        return f"#{self.current_card + 1}"

    @rx.var
    def is_first_card(self) -> bool:
        return self.current_card == 0

    @rx.var
    def is_last_card(self) -> bool:
        return self.current_card == self.num_cards - 1

    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            return
        upload_data = await files[0].read()
        name = files[0].filename.lower()
        if name.endswith(".png"):
            mime = "image/png"
        elif name.endswith(".webp"):
            mime = "image/webp"
        else:
            mime = "image/jpeg"
        encoded = base64.b64encode(upload_data).decode("ascii")
        self.img_data_uri = f"data:{mime};base64,{encoded}"

    async def run_inference(self):
        if not self.has_image:
            return

        self.inference_error = ""
        self.phase = "processing"
        yield

        # Reverse the base64 data URI back to raw bytes so we can forward
        # them as multipart/form-data to the FastAPI /predict endpoint.
        header, _, b64 = self.img_data_uri.partition(",")
        mime = header.removeprefix("data:").partition(";")[0] or "image/jpeg"
        ext = {"image/png": "png", "image/webp": "webp"}.get(mime, "jpg")
        image_bytes = base64.b64decode(b64)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{BIRD_API_URL}/predict",
                    params={"k": 5},
                    files={"file": (f"upload.{ext}", image_bytes, mime)},
                )
                response.raise_for_status()
                predictions = response.json()["predictions"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            self.inference_error = f"Inference failed: {exc}"
            self.phase = "landing"
            return

        # Predictions arrive sorted by score descending. Map species names to
        # cached descriptions so the cards have rich text without an extra
        # round-trip.
        self.card_names = [p["name"] for p in predictions]
        self.card_confidences = [float(p["score"]) for p in predictions]
        self.card_descriptions = [
            _DESCRIPTIONS.get(p["name"], "") for p in predictions
        ]
        self.current_card = 0
        self.phase = "results"

    def prev_card(self):
        if self.current_card > 0:
            self.current_card -= 1

    def next_card(self):
        if self.current_card < self.num_cards - 1:
            self.current_card += 1

    def exit_to_landing(self):
        self.phase = "landing"
        self.current_card = 0
        self.img_data_uri = ""
        self.inference_error = ""
