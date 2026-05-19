"""App entry point: build the rx.App and register the three routed pages.

All logic, styling, and UI live in sibling modules — this file is just
configuration so the framework can discover and mount the pages.
"""

import reflex as rx

from .pages.about import about
from .pages.catalogue import catalogue
from .pages.home import home

# Open Graph + Twitter Card metadata for link previews (iMessage, Slack,
# WhatsApp, Instagram DMs, Twitter, etc.). add_page accepts title,
# description, and image natively — these flow into the page's <head> as
# og:title / og:description / og:image meta tags. We also pass extra
# twitter:* tags via the meta=[] list. CRITICAL: must override the
# default image="favicon.ico" — otherwise scrapers see that first and
# render nothing because favicon.ico doesn't exist on this site.
_OG_DESCRIPTION = (
    "Upload a bird photo and get instant species identification with a "
    "deep learning model trained on 525 species and 89,885 images."
)
_OG_IMAGE = "https://birdml.net/og-preview.jpg"


def _twitter_meta(page_title: str) -> list[dict[str, str]]:
    """Twitter Card tags (og:* tags are auto-emitted by add_page)."""
    return [
        {"property": "og:site_name", "content": "BirdML"},
        {"property": "og:image:width", "content": "1200"},
        {"property": "og:image:height", "content": "630"},
        {"name": "twitter:card", "content": "summary_large_image"},
        {"name": "twitter:title", "content": page_title},
        {"name": "twitter:description", "content": _OG_DESCRIPTION},
        {"name": "twitter:image", "content": _OG_IMAGE},
    ]


app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=False,
        radius="large",
        accent_color="iris",
        scaling="100%",
    )
)

_HOME_TITLE = "BirdML — Identify 525 bird species with AI"
app.add_page(
    home,
    route="/",
    title=_HOME_TITLE,
    description=_OG_DESCRIPTION,
    image=_OG_IMAGE,
    meta=_twitter_meta(_HOME_TITLE),
)
app.add_page(
    about,
    route="/about",
    title="About — BirdML",
    description=_OG_DESCRIPTION,
    image=_OG_IMAGE,
    meta=_twitter_meta("About — BirdML"),
)
app.add_page(
    catalogue,
    route="/species",
    title="Species — BirdML",
    description=_OG_DESCRIPTION,
    image=_OG_IMAGE,
    meta=_twitter_meta("Species — BirdML"),
)
