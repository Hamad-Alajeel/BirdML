"""Floating glass-pill navigation, rendered at the top of every page.

Active-tab highlighting uses State.router.page.path so the same component
works for all three routes without per-page wiring.
"""

import reflex as rx

from ..state import State
from ..styles import ACCENT, ACCENT_HOVER, GLASS_BG, GLASS_BLUR, GLASS_BORDER


def _nav_link(label: str, path: str) -> rx.Component:
    is_active = State.router.page.path == path
    return rx.link(
        rx.text(
            label,
            size="3",
            weight="medium",
            color="white",
            style={"text-shadow": "0 1px 4px rgba(0, 0, 0, 0.6)"},
        ),
        href=path,
        padding_x="4",
        padding_y="2",
        border_radius="9999px",
        background=rx.cond(is_active, ACCENT, "transparent"),
        _hover={"background": ACCENT_HOVER},
        style={
            "text-decoration": "none",
            "transition": "background 0.2s ease",
        },
    )


def navbar() -> rx.Component:
    return rx.center(
        # Pill styling on rx.box so border-radius doesn't clip link text.
        # rx.hstack applies overflow:hidden when border-radius is set on it
        # directly, which cuts off the words at the curved ends.
        rx.box(
            rx.hstack(

                _nav_link("Home", "/"),
                _nav_link("About", "/about"),
                _nav_link("Catalogue", "/species"),
                spacing="6",
                align_items="center",
            ),
            background=GLASS_BG,
            border=GLASS_BORDER,
            border_radius="9999px",
            box_shadow="0 14px 30px -12px rgba(99, 102, 241, 0.45)",
            style={"backdrop-filter": GLASS_BLUR},
        ),
        width="100%",
        padding_top="5",
        padding_bottom="2",
    )
