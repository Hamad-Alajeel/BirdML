"""Species catalogue: search + A-Z accordion + species cards.

Top-level page mounts the shared chrome and switches between four
sub-views (alphabet / search results / no results / single species)
based on State.catalogue_view.
"""

import reflex as rx

from ..components.catalogue_widgets import (
    alphabet_accordion,
    no_results_view,
    search_bar,
    search_results_list,
    species_card_view,
)
from ..components.home_widgets import gradient_heading
from ..components.layout import page_layout
from ..state import State

# Matrix-style parrot variant, as requested for the catalogue page.
_MATRIX_PARROT = "https://cultofthepartyparrot.com/parrots/matrixparrot.gif"


def catalogue() -> rx.Component:
    return page_layout(
        rx.vstack(
            gradient_heading(parrot_gif=_MATRIX_PARROT, subtitle=None),
            # The species card view doesn't need the search bar — drilling
            # into a species is a focused detail view.
            rx.cond(
                State.catalogue_is_species,
                species_card_view(),
                rx.vstack(
                    search_bar(),
                    rx.cond(
                        State.catalogue_is_alphabet,
                        alphabet_accordion(),
                        rx.cond(
                            State.catalogue_is_search_results,
                            search_results_list(),
                            # Fallback = "no_results" branch.
                            no_results_view(),
                        ),
                    ),
                    spacing="5",
                    align_items="start",
                    width="100%",
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
