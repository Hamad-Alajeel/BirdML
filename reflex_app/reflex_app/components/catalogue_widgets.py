"""Catalogue page UI building blocks.

Each function returns one self-contained sub-view (search bar, accordion,
results list, no-results screen, species card). The page-level catalogue()
composition decides which to render based on State.catalogue_view.
"""

import reflex as rx

from ..state import State
from ..styles import GLASS_BG, GLASS_BORDER

# Confused-bird GIF shown when a search returns zero results. The user
# requested this exact Giphy URL — pulling the raw .gif from the Giphy
# media subdomain (media.giphy.com) is required for direct embedding;
# the giphy.com page URL only renders a player.
_NO_RESULTS_GIF = "https://media.giphy.com/media/XGU4CyI27f5xBWGJlY/giphy.gif"

_LIST_ROW_STYLE = {
    "cursor": "pointer",
    "transition": "background 0.15s ease, color 0.15s ease",
}


def search_bar() -> rx.Component:
    """Form with an input + Search button. on_submit dispatches to
    State.submit_search which decides between results / no-results views."""
    return rx.form(
        rx.hstack(
            rx.input(
                name="query",
                placeholder="Search species (e.g. EAGLE, BALD EAGLE)…",
                default_value=State.search_query,
                size="3",
                flex="1",
                style={
                    "background": "rgba(0, 0, 0, 0.30)",
                    "border": "1px solid rgba(167, 139, 250, 0.35)",
                    "color": "white",
                },
            ),
            rx.button(
                rx.hstack(rx.icon("search", size=18), rx.text("Search")),
                type="submit",
                size="3",
                color_scheme="iris",
                style={"cursor": "pointer"},
            ),
            spacing="3",
            width="100%",
        ),
        on_submit=State.submit_search,
        reset_on_submit=False,
        width="100%",
    )


def _species_row(name: rx.Var | str) -> rx.Component:
    """Single clickable species name. Uses lambda-captured name from foreach
    so each row dispatches select_species(<that name>)."""
    return rx.box(
        rx.hstack(
            rx.text("•", color="#a78bfa", size="3"),
            rx.text(
                name,
                color="rgba(255, 255, 255, 0.92)",
                size="3",
            ),
            spacing="3",
            align_items="center",
        ),
        on_click=lambda: State.select_species(name),
        padding_x="3",
        padding_y="2",
        border_radius="6px",
        _hover={"background": "rgba(167, 139, 250, 0.18)"},
        style=_LIST_ROW_STYLE,
        width="100%",
    )


def _letter_row(letter: rx.Var | str) -> rx.Component:
    """One letter of the A-Z accordion. Shows a chevron that flips when
    the letter is expanded, and renders the species list directly beneath."""
    is_open = State.expanded_letter == letter
    return rx.vstack(
        rx.box(
            rx.hstack(
                rx.cond(
                    is_open,
                    rx.icon("chevron-down", size=18, color="#a78bfa"),
                    rx.icon("chevron-right", size=18, color="rgba(255,255,255,0.6)"),
                ),
                rx.text(
                    letter,
                    weight="bold",
                    size="4",
                    color="white",
                ),
                spacing="3",
                align_items="center",
            ),
            on_click=lambda: State.toggle_letter(letter),
            padding_x="3",
            padding_y="2",
            border_radius="6px",
            _hover={"background": "rgba(167, 139, 250, 0.18)"},
            style=_LIST_ROW_STYLE,
            width="100%",
        ),
        rx.cond(
            is_open,
            rx.vstack(
                rx.foreach(State.expanded_letter_species, _species_row),
                spacing="1",
                width="100%",
                padding_left="6",
            ),
            rx.fragment(),
        ),
        spacing="1",
        width="100%",
    )


def alphabet_accordion() -> rx.Component:
    return rx.vstack(
        rx.foreach(State.alphabet_letters, _letter_row),
        spacing="1",
        width="100%",
    )


def search_results_list() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(
                f'Results for "',
                color="rgba(255, 255, 255, 0.7)",
                size="3",
            ),
            rx.text(
                State.search_query,
                color="white",
                weight="bold",
                size="3",
            ),
            rx.text(
                '"',
                color="rgba(255, 255, 255, 0.7)",
                size="3",
            ),
            spacing="1",
            align_items="center",
        ),
        rx.vstack(
            rx.foreach(State.search_results, _species_row),
            spacing="1",
            width="100%",
        ),
        rx.button(
            rx.hstack(rx.icon("arrow-left", size=18), rx.text("Back to A-Z")),
            on_click=State.back_to_alphabet,
            variant="outline",
            size="3",
            color_scheme="gray",
            style={"cursor": "pointer", "color": "white"},
        ),
        spacing="4",
        align_items="start",
        width="100%",
    )


def no_results_view() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.image(
                src=_NO_RESULTS_GIF,
                width="240px",
                height="auto",
                border_radius="14px",
                box_shadow="0 20px 50px -15px rgba(0, 0, 0, 0.7)",
            ),
            rx.heading(
                "No matching species",
                size="5",
                color="white",
                weight="bold",
                text_align="center",
            ),
            rx.hstack(
                rx.text(
                    "Nothing matched",
                    color="rgba(255, 255, 255, 0.75)",
                    size="3",
                ),
                rx.text(
                    State.search_query,
                    color="white",
                    weight="bold",
                    size="3",
                ),
                spacing="2",
                align_items="center",
            ),
            rx.button(
                rx.hstack(rx.icon("arrow-left", size=18), rx.text("Back to A-Z")),
                on_click=State.back_to_alphabet,
                variant="outline",
                size="3",
                color_scheme="gray",
                margin_top="3",
                style={"cursor": "pointer", "color": "white"},
            ),
            spacing="4",
            align_items="center",
            width="100%",
        ),
        width="100%",
        padding_y="8",
    )


def species_card_view() -> rx.Component:
    """Standalone bird card shown when the user drills into a species from
    the catalogue. Mirrors the home page carousel card layout but without
    the rank badge / confidence bar / navigation chevrons, since this card
    isn't a prediction — it's a static encyclopedia entry."""
    card = rx.box(
        rx.hstack(
            rx.box(
                rx.image(
                    src=State.selected_species_image,
                    width="224px",
                    height="224px",
                    object_fit="cover",
                    border="4px solid rgba(167, 139, 250, 0.75)",
                    border_radius="14px",
                    flex_shrink="0",
                ),
                width="260px",
                flex_shrink="0",
                style={
                    "align-self": "stretch",
                    "display": "flex",
                    "align-items": "center",
                    "justify-content": "center",
                },
            ),
            rx.vstack(
                rx.heading(
                    State.selected_species,
                    size="6",
                    weight="bold",
                    color="white",
                ),
                rx.text(
                    State.selected_species_description,
                    size="2",
                    color="rgba(255, 255, 255, 0.80)",
                    line_height="1.7",
                ),
                spacing="4",
                flex="1",
                align_items="start",
                style={
                    "padding-top": "36px",
                    "padding-bottom": "36px",
                    "padding-left": "48px",
                    "padding-right": "40px",
                },
            ),
            spacing="0",
            align_items="stretch",
            width="100%",
            height="100%",
        ),
        overflow="hidden",
        border_radius="16px",
        width="680px",
        min_height="340px",
        border=GLASS_BORDER,
        background=GLASS_BG,
        style={
            "backdrop-filter": "blur(14px)",
            "box-shadow": "0 25px 60px -20px rgba(99, 102, 241, 0.5)",
        },
    )

    return rx.vstack(
        card,
        rx.button(
            rx.hstack(rx.icon("arrow-left", size=18), rx.text("Back")),
            on_click=State.back_from_species,
            variant="outline",
            size="3",
            color_scheme="gray",
            margin_top="4",
            style={"cursor": "pointer", "color": "white"},
        ),
        spacing="4",
        align_items="center",
        width="100%",
    )
