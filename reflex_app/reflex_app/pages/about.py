"""About page — stub.

Real content (training writeup, hyperparameters, transfer learning notes)
will replace the placeholder card later.
"""

import reflex as rx

from ..components.layout import page_layout
from ..styles import GLASS_BG, GLASS_BLUR, GLASS_BORDER


def _coming_soon_card() -> rx.Component:
    return rx.card(
        rx.center(
            rx.vstack(
                rx.icon("book-open", size=40, color="#f0abfc"),
                rx.heading(
                    "About",
                    size="8",
                    color="white",
                    weight="bold",
                    text_align="center",
                ),
                rx.text(
                    "A writeup of how BirdML was trained — dataset, transfer "
                    "learning, hyperparameters, and evaluation — is coming soon.",
                    size="3",
                    color="rgba(255, 255, 255, 0.75)",
                    text_align="center",
                    max_width="520px",
                    style={"text-shadow": "0 1px 4px rgba(0, 0, 0, 0.6)"},
                ),
                align_items="center",
                spacing="4",
            ),
            width="100%",
            height="100%",
        ),
        width="100%",
        max_width="640px",
        min_height="320px",
        padding="6",
        style={
            "background": GLASS_BG,
            "border": GLASS_BORDER,
            "backdrop_filter": GLASS_BLUR,
            "box_shadow": "0 25px 60px -20px rgba(99, 102, 241, 0.5)",
        },
    )


def about() -> rx.Component:
    return page_layout(
        rx.vstack(
            _coming_soon_card(),
            align_items="center",
            width="100%",
            max_width="780px",
            padding_x="4",
            padding_y="10",
        )
    )
