"""Home page — upload + inference + top-5 carousel.

Wraps the existing landing/processing/results flow in page_layout so it
inherits the shared background and navbar. No behavioral changes from the
pre-refactor version.
"""

import reflex as rx

from ..components.home_widgets import (
    carousel_card,
    gradient_heading,
    landing_view,
    nn_animation,
)
from ..components.layout import page_layout
from ..state import State


def home() -> rx.Component:
    return page_layout(
        rx.vstack(
            gradient_heading(),
            rx.cond(
                State.is_landing,
                landing_view(),
                rx.cond(
                    State.is_processing,
                    nn_animation(),
                    carousel_card(),
                ),
            ),
            spacing="7",
            align_items="center",
            width="100%",
            max_width="780px",
            padding_x=rx.breakpoints(initial="7", md="4"),
            padding_y="10",
        )
    )
