"""Shared app state.

Lives in its own module so every page and component can import the same
State subclass without circular imports back through reflex_app.py.
"""

import asyncio
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

# Sorted master list of every species the catalogue can show. Built from
# the keys of _DESCRIPTIONS so the catalogue stays in sync with whatever
# species_data files exist on disk.
_ALL_SPECIES: list[str] = sorted(_DESCRIPTIONS.keys())

# Bucket species by first letter for the A-Z accordion. Keys are uppercase
# single characters; values are pre-sorted lists of species names.
_SPECIES_BY_LETTER: dict[str, list[str]] = {}
for _name in _ALL_SPECIES:
    _first = _name[0].upper() if _name else "?"
    _SPECIES_BY_LETTER.setdefault(_first, []).append(_name)

_ALPHABET_LETTERS: list[str] = sorted(_SPECIES_BY_LETTER.keys())


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

    # ---------------- Catalogue page state ----------------
    # Active sub-view of the catalogue page.
    #   "alphabet"        — default A-Z accordion
    #   "search_results"  — list of species matching the search query
    #   "no_results"      — confused-bird animation when search has no hits
    #   "species"         — single species card (drilled in from any list view)
    catalogue_view: str = "alphabet"

    # Tracks where the user came from when they drilled into a species card
    # so the Back button returns them to the right list.
    previous_catalogue_view: str = "alphabet"

    search_query: str = ""
    search_results: list[str] = []

    # Which letter of the A-Z accordion is currently expanded ("" = none).
    expanded_letter: str = ""

    # Species name whose card is currently shown.
    selected_species: str = ""

    @rx.var
    def alphabet_letters(self) -> list[str]:
        return _ALPHABET_LETTERS

    @rx.var
    def expanded_letter_species(self) -> list[str]:
        return _SPECIES_BY_LETTER.get(self.expanded_letter, [])

    @rx.var
    def catalogue_is_alphabet(self) -> bool:
        return self.catalogue_view == "alphabet"

    @rx.var
    def catalogue_is_search_results(self) -> bool:
        return self.catalogue_view == "search_results"

    @rx.var
    def catalogue_is_no_results(self) -> bool:
        return self.catalogue_view == "no_results"

    @rx.var
    def catalogue_is_species(self) -> bool:
        return self.catalogue_view == "species"

    @rx.var
    def selected_species_image(self) -> str:
        return f"/species/{self.selected_species}.jpg"

    @rx.var
    def selected_species_description(self) -> str:
        return _DESCRIPTIONS.get(self.selected_species, "")

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

    def clear_image(self):
        """Drop the currently-uploaded image so the user can start over."""
        self.img_data_uri = ""
        self.inference_error = ""

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
                # Run the API call and a minimum 3-second delay concurrently
                # so the SVG animation always completes one full rainbow cycle
                # (longest path = 3.5s color + 1.2s stagger) before the cards
                # appear, even if inference is fast.
                response, _ = await asyncio.gather(
                    client.post(
                        f"{BIRD_API_URL}/predict",
                        params={"k": 5},
                        files={"file": (f"upload.{ext}", image_bytes, mime)},
                    ),
                    asyncio.sleep(3.0),
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

    # ---------------- Catalogue handlers ----------------
    def submit_search(self, form_data: dict):
        """Match whole, case-insensitive tokens between the user query and
        every species name. A species is a hit if ANY word in its name equals
        ANY word in the query. Results are de-duplicated and sorted."""
        query = (form_data.get("query") or "").strip()
        self.search_query = query
        if not query:
            self.catalogue_view = "alphabet"
            self.search_results = []
            return

        user_tokens = {t.upper() for t in query.split() if t}
        matches: set[str] = set()
        for species in _ALL_SPECIES:
            species_tokens = {t.upper() for t in species.split() if t}
            if user_tokens & species_tokens:
                matches.add(species)

        self.search_results = sorted(matches)
        self.catalogue_view = "search_results" if matches else "no_results"
        self.expanded_letter = ""

    def back_to_alphabet(self):
        """Clear search state and return to the default A-Z accordion view."""
        self.search_query = ""
        self.search_results = []
        self.expanded_letter = ""
        self.catalogue_view = "alphabet"

    def toggle_letter(self, letter: str):
        """Open/close one letter of the A-Z accordion. Only one letter is
        expanded at a time so the list stays compact."""
        self.expanded_letter = "" if self.expanded_letter == letter else letter

    def select_species(self, name: str):
        """Drill into a single species card from whichever list view is
        active. previous_catalogue_view remembers where to return to."""
        self.previous_catalogue_view = self.catalogue_view
        self.selected_species = name
        self.catalogue_view = "species"

    def back_from_species(self):
        """Return to the list view the user drilled in from."""
        self.catalogue_view = self.previous_catalogue_view
        self.selected_species = ""
