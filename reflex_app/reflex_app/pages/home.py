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
    low_confidence_view,
    nn_animation,
)
from ..components.layout import page_layout
from ..state import State

_DEFAULT_PARROT = "https://cultofthepartyparrot.com/parrots/hd/parrot.gif"
_CONFUSED_PARROT = "https://cultofthepartyparrot.com/parrots/hd/confusedparrot.gif"


def home() -> rx.Component:
    return page_layout(
        rx.vstack(
            # Swap the party parrot for a confused parrot when the model
            # flagged the predictions as low-confidence, to visually echo
            # the "Hmm, the model isn't sure" message in the body.
            gradient_heading(
                parrot_gif=rx.cond(
                    State.is_low_confidence,
                    _CONFUSED_PARROT,
                    _DEFAULT_PARROT,
                ),
            ),
            rx.cond(
                State.is_landing,
                landing_view(),
                rx.cond(
                    State.is_processing,
                    nn_animation(),
                    rx.cond(
                        State.is_low_confidence,
                        low_confidence_view(),
                        carousel_card(),
                    ),
                ),
            ),
            spacing=rx.breakpoints(initial="5", md="7"),
            align_items="center",
            width="100%",
            max_width=rx.breakpoints(initial="86vw", md="780px"),
            padding_x=rx.breakpoints(initial="5", md="4"),
            padding_top=rx.breakpoints(initial="6", md="10"),
            padding_bottom=rx.breakpoints(initial="64px", md="40px"),
        )
    )
