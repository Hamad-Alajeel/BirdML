"""Generate a 1200x630 Open Graph preview image for birdml.net link previews.

Takes the existing macaws background, center-crops/resizes to the standard
1.91:1 OG aspect ratio, darkens it for text legibility, and overlays the
BirdML title plus a subtitle. The result is saved to
reflex_app/assets/og-preview.jpg so the og:image meta tag can point at it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "reflex_app" / "assets" / "macaws_2.jpg"
OUT = REPO / "reflex_app" / "assets" / "og-preview.jpg"

# Standard OG image dimensions (Facebook/Slack/iMessage all show this best).
TARGET_W = 1200
TARGET_H = 630


def _load_font(candidates: list[tuple[str, int]]) -> ImageFont.ImageFont:
    """Try a list of (font_path, size) pairs, fall back to PIL default."""
    for path, size in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


def main() -> int:
    if not SRC.is_file():
        print(f"ERROR: source image not found: {SRC}", file=sys.stderr)
        return 1

    bg = Image.open(SRC).convert("RGB")

    # Center-crop to OG aspect ratio (1200/630 ~= 1.905).
    target_aspect = TARGET_W / TARGET_H
    w, h = bg.size
    if w / h > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        bg = bg.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        bg = bg.crop((0, top, w, top + new_h))
    bg = bg.resize((TARGET_W, TARGET_H), Image.LANCZOS).convert("RGBA")

    # Darken via translucent black overlay so the white text reads well.
    overlay = Image.new("RGBA", (TARGET_W, TARGET_H), (8, 6, 16, 130))
    bg = Image.alpha_composite(bg, overlay)

    draw = ImageDraw.Draw(bg)

    title_font = _load_font([
        ("C:/Windows/Fonts/arialbd.ttf", 180),
        ("C:/Windows/Fonts/arial.ttf", 180),
        ("/System/Library/Fonts/Helvetica.ttc", 180),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 180),
    ])
    subtitle_font = _load_font([
        ("C:/Windows/Fonts/arial.ttf", 44),
        ("/System/Library/Fonts/Helvetica.ttc", 44),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 44),
    ])

    # Title (centered horizontally + visually centered vertically with a
    # slight upward bias to leave room for the subtitle).
    title = "BirdML"
    tb = draw.textbbox((0, 0), title, font=title_font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = (TARGET_W - tw) // 2
    ty = (TARGET_H - th) // 2 - 40

    # Soft drop shadow then white text.
    for dx, dy in [(4, 4), (3, 3), (2, 2)]:
        draw.text((tx + dx, ty + dy), title, font=title_font, fill=(0, 0, 0, 180))
    draw.text((tx, ty), title, font=title_font, fill="white")

    # Subtitle below the title.
    subtitle = "Identify 525 bird species with AI"
    sb = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    sw, sh = sb[2] - sb[0], sb[3] - sb[1]
    sx = (TARGET_W - sw) // 2
    sy = ty + th + 36
    draw.text((sx + 2, sy + 2), subtitle, font=subtitle_font, fill=(0, 0, 0, 200))
    draw.text((sx, sy), subtitle, font=subtitle_font, fill=(225, 225, 245))

    # Domain footer in the bottom-right corner for a polished feel.
    domain = "birdml.net"
    dom_font = _load_font([
        ("C:/Windows/Fonts/arial.ttf", 32),
        ("/System/Library/Fonts/Helvetica.ttc", 32),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32),
    ])
    db = draw.textbbox((0, 0), domain, font=dom_font)
    dw = db[2] - db[0]
    draw.text(
        (TARGET_W - dw - 36, TARGET_H - 60),
        domain,
        font=dom_font,
        fill=(255, 255, 255, 200),
    )

    bg.convert("RGB").save(OUT, format="JPEG", quality=88, optimize=True)
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
