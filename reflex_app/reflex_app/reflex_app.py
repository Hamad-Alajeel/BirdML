"""App entry point: build the rx.App and register the three routed pages.

All logic, styling, and UI live in sibling modules — this file is just
configuration so the framework can discover and mount the pages.
"""

import reflex as rx

from .pages.about import about
from .pages.catalogue import catalogue
from .pages.home import home

app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=False,
        radius="large",
        accent_color="iris",
        scaling="100%",
    )
)

app.add_page(home, route="/", title="BirdML")
app.add_page(about, route="/about", title="About — BirdML")
app.add_page(catalogue, route="/species", title="Species — BirdML")
