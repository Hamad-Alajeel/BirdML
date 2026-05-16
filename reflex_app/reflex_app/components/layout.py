"""Shared page chrome: background image, dark overlay, navbar, content slot.

Wrapping each page in page_layout() guarantees all three routes share the
exact same background, overlay, and navigation — no per-page duplication.
"""

import reflex as rx

from ..styles import BACKGROUND_IMAGE
from .navbar import navbar


def page_layout(content: rx.Component) -> rx.Component:
    """Wrap a page's content with the shared background + overlay + navbar."""
    return rx.box(
        # Dark gradient overlay sits above the background image so text and
        # glass cards stay legible regardless of which photo loads.
        rx.vstack(
            navbar(),
            # flex=1 lets the content area expand to fill the rest of the
            # viewport, preserving the home page's vertical centering.
            rx.center(content, width="100%", flex="1"),
            spacing="0",
            min_height="100vh",
            width="100%",
            background=(
                "linear-gradient(180deg, "
                "rgba(8, 6, 16, 0.55) 0%, "
                "rgba(8, 6, 16, 0.78) 100%)"
            ),
        ),
        background_image=BACKGROUND_IMAGE,
        background_size="cover, cover, auto",
        background_repeat="no-repeat",
        background_position="center 45%",
        background_attachment="fixed",
        background_color="#0a1a14",
        min_height="100vh",
        width="100%",
    )
