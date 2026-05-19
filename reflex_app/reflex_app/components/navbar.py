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
            size=["2", "2", "3"],
            weight="medium",
            color="white",
            style={"text-shadow": "0 1px 4px rgba(0, 0, 0, 0.6)"},
        ),
        href=path,
        padding_x=["2", "3", "4"],
        padding_y=["1", "1", "2"],
        border_radius="9999px",
        background=rx.cond(is_active, ACCENT, "transparent"),
        _hover={"background": ACCENT_HOVER},
        style={
            "text-decoration": "none",
            "transition": "background 0.2s ease",
        },
    )


_GITHUB_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" '
    'viewBox="0 0 24 24" fill="rgba(255,255,255,0.75)">'
    '<path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.205 '
    '11.385.6.113.82-.263.82-.577 0-.285-.01-1.04-.015-2.04-3.338 '
    '.727-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.09'
    '-.745.083-.73.083-.73 1.205.085 1.84 1.238 1.84 1.238 1.07 '
    '1.835 2.807 1.305 3.492.998.108-.775.418-1.305.762-1.605-2.665'
    '-.305-5.467-1.333-5.467-5.93 0-1.31.468-2.382 1.235-3.222-.123'
    '-.303-.535-1.523.117-3.176 0 0 1.008-.322 3.3 1.23A11.52 '
    '11.52 0 0 1 12 5.8c1.02.005 2.047.138 3.006.405 2.29-1.552 '
    '3.295-1.23 3.295-1.23.653 1.653.242 2.873.12 3.176.77.84 '
    '1.232 1.912 1.232 3.222 0 4.61-2.807 5.622-5.48 5.92.43.372'
    '.815 1.103.815 2.222 0 1.606-.015 2.9-.015 3.293 0 .317.216'
    '.695.825.577C20.565 21.795 24 17.297 24 12c0-6.63-5.37-12-12-12z"/>'
    '</svg>'
)

_LINKEDIN_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="rgba(255,255,255,0.75)">'
    '<path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046'
    'c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065z'
    'm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>'
    '</svg>'
)


def _github_icon() -> rx.Component:
    return rx.html(_GITHUB_SVG)


def _linkedin_icon() -> rx.Component:
    return rx.html(_LINKEDIN_SVG)


def _icon_link(icon_component: rx.Component, href: str, label: str) -> rx.Component:
    return rx.link(
        icon_component,
        href=href,
        is_external=True,
        aria_label=label,
        padding="2",
        border_radius="9999px",
        _hover={"background": ACCENT_HOVER},
        style={
            "text-decoration": "none",
            "display": "flex",
            "align-items": "center",
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
                rx.divider(orientation="vertical", size="2", color_scheme="gray"),
                _icon_link(_github_icon(), "https://github.com/Hamad-Alajeel", "GitHub"),
                _icon_link(_linkedin_icon(), "https://www.linkedin.com/in/hamad-alajeel-a6bb41210/", "LinkedIn"),
                spacing=["1", "2", "3"],
                align_items="center",
                padding_x=["3", "4", "6"],
                padding_y=["2", "2", "3"],
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
