import asyncio
import base64

import reflex as rx


# --- STATE MANAGEMENT ---
class State(rx.State):
    """Tracks app phase, the uploaded image, and carousel navigation."""

    # Phase: "landing" | "processing" | "results"
    phase: str = "landing"

    # Uploaded image as a base64 data URI (no filesystem dependency)
    img_data_uri: str = ""

    # Carousel state
    current_card: int = 0
    card_names: list[str] = [
        "Placeholder Bird 1",
        "Placeholder Bird 2",
        "Placeholder Bird 3",
        "Placeholder Bird 4",
        "Placeholder Bird 5",
    ]

    @rx.var
    def has_image(self) -> bool:
        return self.img_data_uri != ""

    @rx.var
    def is_landing(self) -> bool:
        return self.phase == "landing"

    @rx.var
    def is_processing(self) -> bool:
        return self.phase == "processing"

    @rx.var
    def is_results(self) -> bool:
        return self.phase == "results"

    @rx.var
    def num_cards(self) -> int:
        return len(self.card_names)

    @rx.var
    def position_label(self) -> str:
        return f"{self.current_card + 1} / {self.num_cards}"

    @rx.var
    def current_card_name(self) -> str:
        return self.card_names[self.current_card]

    @rx.var
    def current_rank(self) -> str:
        return f"#{self.current_card + 1}"

    @rx.var
    def is_first_card(self) -> bool:
        return self.current_card == 0

    @rx.var
    def is_last_card(self) -> bool:
        return self.current_card == self.num_cards - 1

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Read the uploaded image and store it as a data URI for instant preview."""
        if not files:
            return

        upload_data = await files[0].read()
        name = files[0].filename.lower()
        if name.endswith(".png"):
            mime = "image/png"
        elif name.endswith(".webp"):
            mime = "image/webp"
        else:
            mime = "image/jpeg"

        encoded = base64.b64encode(upload_data).decode("ascii")
        self.img_data_uri = f"data:{mime};base64,{encoded}"

    async def run_inference(self):
        """Show the neural-network animation briefly, then reveal the carousel."""
        if not self.has_image:
            return

        self.phase = "processing"
        yield
        # Let the animation play for ~1 second
        await asyncio.sleep(1.2)
        self.phase = "results"
        self.current_card = 0

    def prev_card(self):
        if self.current_card > 0:
            self.current_card -= 1

    def next_card(self):
        if self.current_card < self.num_cards - 1:
            self.current_card += 1

    def exit_to_landing(self):
        """Reset everything to the original landing state."""
        self.phase = "landing"
        self.current_card = 0
        self.img_data_uri = ""


# --- STYLE TOKENS ---
GRADIENT_TITLE = "linear-gradient(135deg, var(--iris-11) 0%, var(--violet-10) 50%, var(--cyan-10) 100%)"


# --- UI COMPONENTS ---
def gradient_heading() -> rx.Component:
    return rx.vstack(
        rx.heading(
            "OrniTrack AI",
            size="9",
            background_image=GRADIENT_TITLE,
            background_clip="text",
            style={
                "-webkit-background-clip": "text",
                "-webkit-text-fill-color": "transparent",
            },
            weight="bold",
            letter_spacing="-0.02em",
        ),
        rx.text(
            "Drop a bird photo, run inference, browse the top 5 predictions.",
            color_scheme="gray",
            size="3",
        ),
        align_items="center",
        spacing="2",
        margin_y="8",
    )


def image_preview() -> rx.Component:
    """Top section — shows the uploaded image or a hint placeholder."""
    return rx.cond(
        State.has_image,
        rx.box(
            rx.image(
                src=State.img_data_uri,
                width="100%",
                height="auto",
                max_height="320px",
                object_fit="contain",
                border_radius="14px",
                box_shadow="0 14px 40px -12px rgba(99, 102, 241, 0.45)",
            ),
            width="100%",
            background="var(--gray-2)",
            padding="3",
            border_radius="16px",
            border="1px solid var(--gray-4)",
        ),
        rx.center(
            rx.vstack(
                rx.icon("image", size=36, color="var(--gray-8)"),
                rx.text("Your image will appear here", color_scheme="gray", size="2"),
                align_items="center",
                spacing="2",
            ),
            width="100%",
            height="200px",
            border="1px dashed var(--gray-5)",
            border_radius="16px",
            background="var(--gray-1)",
        ),
    )


def upload_zone() -> rx.Component:
    """Middle section — drag and drop / browse for an image."""
    return rx.upload(
        rx.vstack(
            rx.icon("upload", size=28, color="var(--iris-10)"),
            rx.text("Drop a bird photo here", weight="medium", size="3"),
            rx.text("or click to browse", size="2", color_scheme="gray"),
            align_items="center",
            spacing="2",
            padding_y="4",
        ),
        id="bird_uploader",
        accept={"image/*": [".png", ".jpg", ".jpeg", ".webp"]},
        max_files=1,
        on_drop=State.handle_upload(rx.upload_files(upload_id="bird_uploader")),
        border="2px dashed var(--gray-6)",
        border_radius="14px",
        padding="5",
        width="100%",
        background="var(--gray-1)",
        _hover={"border_color": "var(--iris-9)", "background": "var(--gray-2)"},
        transition="all 0.25s ease",
    )


def run_inference_button() -> rx.Component:
    """Bottom section — main action."""
    return rx.button(
        rx.hstack(
            rx.icon("zap", size=18),
            rx.text("Run Inference", weight="bold"),
            spacing="2",
            align_items="center",
        ),
        on_click=State.run_inference,
        disabled=~State.has_image,
        size="4",
        color_scheme="iris",
        width="100%",
        style={"cursor": "pointer"},
    )


def nn_animation() -> rx.Component:
    """Pulsing SVG neural network — pure SVG SMIL, no CSS required."""
    layers = [3, 5, 5, 3]
    layer_xs = [80, 220, 360, 500]

    def y_positions(n: int) -> list[float]:
        spacing = 240 / (n + 1)
        return [spacing * (i + 1) for i in range(n)]

    positions = [
        [(x, y) for y in y_positions(n)]
        for x, n in zip(layer_xs, layers)
    ]

    # Lines between every pair of adjacent-layer nodes
    line_parts: list[str] = []
    for i in range(len(positions) - 1):
        for (x1, y1) in positions[i]:
            for (x2, y2) in positions[i + 1]:
                line_parts.append(
                    f'<line x1="{x1}" y1="{y1:.1f}" x2="{x2}" y2="{y2:.1f}"/>'
                )

    # Pulsing circles with staggered start
    circle_parts: list[str] = []
    delay = 0.0
    for layer in positions:
        for (x, y) in layer:
            circle_parts.append(
                f'<circle cx="{x}" cy="{y:.1f}" r="10">'
                f'<animate attributeName="r" values="8;14;8" dur="1.2s" '
                f'begin="{delay:.2f}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="0.45;1;0.45" dur="1.2s" '
                f'begin="{delay:.2f}s" repeatCount="indefinite"/>'
                f"</circle>"
            )
            delay += 0.05

    svg = (
        '<svg viewBox="0 0 580 240" xmlns="http://www.w3.org/2000/svg" '
        'style="width: 100%; max-width: 580px; height: auto;">'
        '<g stroke="#a78bfa" stroke-width="0.7" opacity="0.35">'
        + "".join(line_parts)
        + "</g>"
        '<g fill="#a78bfa" filter="drop-shadow(0 0 6px rgba(167, 139, 250, 0.6))">'
        + "".join(circle_parts)
        + "</g>"
        "</svg>"
    )

    return rx.vstack(
        rx.html(svg),
        rx.hstack(
            rx.spinner(size="3"),
            rx.text("Running inference...", size="3", weight="medium", color_scheme="iris"),
            spacing="3",
            align_items="center",
        ),
        rx.text(
            "Forward pass through the network",
            size="2",
            color_scheme="gray",
            italic=True,
        ),
        spacing="5",
        align_items="center",
        width="100%",
        padding_y="6",
    )


def carousel_card() -> rx.Component:
    """Single placeholder card with left/right arrows and a position indicator."""
    return rx.vstack(
        rx.hstack(
            rx.button(
                rx.icon("chevron-left", size=24),
                on_click=State.prev_card,
                disabled=State.is_first_card,
                variant="soft",
                size="3",
                color_scheme="gray",
                style={"cursor": "pointer"},
            ),
            rx.card(
                rx.vstack(
                    rx.badge(
                        State.current_rank,
                        color_scheme="iris",
                        variant="soft",
                        size="2",
                        radius="full",
                    ),
                    rx.heading(
                        State.current_card_name,
                        size="6",
                        text_align="center",
                        weight="bold",
                    ),
                    rx.text("placeholder", size="2", color_scheme="gray", italic=True),
                    align_items="center",
                    spacing="3",
                    padding_y="8",
                ),
                width="360px",
                min_height="220px",
                style={
                    "background": "var(--gray-2)",
                    "border": "1px solid var(--gray-5)",
                    "transition": "transform 0.25s ease, border-color 0.25s ease",
                },
                _hover={"transform": "translateY(-2px)", "border_color": "var(--iris-7)"},
            ),
            rx.button(
                rx.icon("chevron-right", size=24),
                on_click=State.next_card,
                disabled=State.is_last_card,
                variant="soft",
                size="3",
                color_scheme="gray",
                style={"cursor": "pointer"},
            ),
            align_items="center",
            spacing="4",
            justify="center",
            width="100%",
        ),
        rx.hstack(
            rx.foreach(
                [0, 1, 2, 3, 4],
                lambda i: rx.box(
                    width="28px",
                    height="4px",
                    border_radius="full",
                    background=rx.cond(
                        State.current_card == i,
                        "var(--iris-9)",
                        "var(--gray-5)",
                    ),
                    transition="background 0.2s ease",
                ),
            ),
            spacing="2",
            justify="center",
        ),
        rx.text(State.position_label, size="1", color_scheme="gray"),
        rx.button(
            rx.hstack(rx.icon("x", size=16), rx.text("Exit"), spacing="2"),
            on_click=State.exit_to_landing,
            variant="outline",
            size="2",
            color_scheme="gray",
            margin_top="4",
            style={"cursor": "pointer"},
        ),
        spacing="5",
        align_items="center",
        width="100%",
    )


def landing_view() -> rx.Component:
    """Image preview (top) + upload zone (middle) + run-inference button (bottom)."""
    return rx.vstack(
        image_preview(),
        upload_zone(),
        run_inference_button(),
        spacing="5",
        align_items="stretch",
        width="100%",
    )


def index() -> rx.Component:
    return rx.box(
        rx.container(
            gradient_heading(),
            rx.box(
                rx.cond(
                    State.is_landing,
                    landing_view(),
                    rx.cond(
                        State.is_processing,
                        nn_animation(),
                        carousel_card(),
                    ),
                ),
                background="var(--gray-2)",
                border="1px solid var(--gray-4)",
                border_radius="20px",
                padding="6",
                width="100%",
            ),
            rx.text(
                "Demo UI — predictions are placeholder data.",
                size="1",
                color_scheme="gray",
                text_align="center",
                margin_y="6",
            ),
            max_width="620px",
            padding_x="4",
        ),
        min_height="100vh",
        background_image="radial-gradient(circle at top, var(--iris-3), var(--gray-1) 60%)",
    )


# --- APP CONFIGURATION ---
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        radius="large",
        accent_color="iris",
        scaling="100%",
    )
)
app.add_page(index, title="OrniTrack AI")
