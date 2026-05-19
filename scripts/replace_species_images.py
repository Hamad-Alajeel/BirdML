"""One-off script: replace weird/wrong species thumbnails with cleaner
images pulled from each species' Wikipedia article main image.

For each entry below, we hit Wikipedia's REST summary API to get the
article's lead image URL, download it, center-crop to a square, resize
to 224x224, and overwrite reflex_app/assets/species/<NAME>.jpg.

Wikipedia article titles were hand-picked so ambiguous entries map to
a sensible representative species:
  COCKATOO         -> Sulphur-crested cockatoo (the iconic one)
  BORNEAN PHEASANT -> Bornean peacock-pheasant
  LOONEY BIRDS     -> Common loon (loons get the "loony" nickname)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import requests
from PIL import Image, ImageFilter

ASSETS = Path(__file__).resolve().parent.parent / "reflex_app" / "assets" / "species"

# species filename (no extension) -> Wikipedia article title.
# Limited to the 11 that came out zoomed in / chopped on the first pass;
# the three already-good ones (Asian Green Bee Eater, Avadavat, Plush
# Crested Jay) are left alone.
SPECIES_TO_WIKI: dict[str, str] = {
    "BALD IBIS": "Northern_bald_ibis",
    "BAND TAILED GUAN": "Band-tailed_guan",
    "BLACK FACED SPOONBILL": "Black-faced_spoonbill",
    "BLACK HEADED CAIQUE": "Black-headed_parrot",
    "BORNEAN PHEASANT": "Bornean_peacock-pheasant",
    "CHATTERING LORY": "Chattering_lory",
    "COCKATOO": "Sulphur-crested_cockatoo",
    "DAURIAN REDSTART": "Daurian_redstart",
    "GREAT ARGUS": "Great_argus",
    "GREY HEADED FISH EAGLE": "Grey-headed_fish_eagle",
    "LOONEY BIRDS": "Common_loon",
}

USER_AGENT = "BirdMLAssetRefresh/1.0 (https://birdml.net; halajeel@ucsd.edu)"


def fetch_main_image_url(article_title: str) -> str:
    """Return the URL of the article's lead image (originalimage.source)."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{article_title}"
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    data = response.json()
    original = data.get("originalimage")
    if not original or "source" not in original:
        raise RuntimeError(f"No originalimage for {article_title}: {data.get('title')}")
    return original["source"]


def to_224_square(raw_bytes: bytes) -> Image.Image:
    """Fit the entire image into a 224x224 frame without cropping the bird.

    Strategy: scale the whole image so its longer side is 224, then paste
    it centered onto a blurred 224x224 version of itself. No subject is
    ever cropped out, and the leftover space looks intentional rather
    than a black bar.
    """
    original = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    # Blurred backdrop: aggressive center-crop to square, then blur heavy.
    w, h = original.size
    short = min(w, h)
    left = (w - short) // 2
    top = (h - short) // 2
    backdrop = original.crop((left, top, left + short, top + short))
    backdrop = backdrop.resize((224, 224), Image.LANCZOS)
    backdrop = backdrop.filter(ImageFilter.GaussianBlur(radius=18))

    # Foreground: keep aspect, fit inside 224x224.
    foreground = original.copy()
    foreground.thumbnail((224, 224), Image.LANCZOS)
    fw, fh = foreground.size
    x = (224 - fw) // 2
    y = (224 - fh) // 2
    backdrop.paste(foreground, (x, y))
    return backdrop


def main() -> int:
    if not ASSETS.is_dir():
        print(f"ERROR: assets dir not found: {ASSETS}", file=sys.stderr)
        return 1

    failures: list[str] = []
    for species, wiki_title in SPECIES_TO_WIKI.items():
        target = ASSETS / f"{species}.jpg"
        print(f"[{species}] -> {wiki_title}")
        try:
            image_url = fetch_main_image_url(wiki_title)
            print(f"  source: {image_url}")
            raw = requests.get(
                image_url,
                headers={"User-Agent": USER_AGENT},
                timeout=60,
            ).content
            img = to_224_square(raw)
            img.save(target, format="JPEG", quality=88, optimize=True)
            print(f"  wrote:  {target} ({target.stat().st_size} bytes)")
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures.append(species)

    print()
    if failures:
        print(f"Done with {len(failures)} failures: {failures}", file=sys.stderr)
        return 1
    print(f"Done. Replaced {len(SPECIES_TO_WIKI)} images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
