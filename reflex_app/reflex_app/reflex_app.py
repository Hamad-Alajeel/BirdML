import reflex as rx


# --- STATE MANAGEMENT ---
class State(rx.State):
    """Handles the application logic, image upload, and carousel navigation."""

    # Upload state
    is_processing: bool = False
    image_uploaded: bool = False
    img_path: str = ""

    # Carousel state
    cards_visible: bool = False
    current_card: int = 0

    # Placeholder card names — replace with model output once wired
    card_names: list[str] = [
        "Placeholder Bird 1",
        "Placeholder Bird 2",
        "Placeholder Bird 3",
        "Placeholder Bird 4",
        "Placeholder Bird 5",
    ]

    @rx.var
    def num_cards(self) -> int:
        return len(self.card_names)

    @rx.var
    def current_card_label(self) -> str:
        return f"#{self.current_card + 1} of {self.num_cards}"

    @rx.var
    def current_card_name(self) -> str:
        return self.card_names[self.current_card]

    @rx.var
    def is_first_card(self) -> bool:
        return self.current_card == 0

    @rx.var
    def is_last_card(self) -> bool:
        return self.current_card == self.num_cards - 1

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Processes the uploaded bird image and triggers (mock) model inference."""
        if not files:
            return

        self.is_processing = True
        self.cards_visible = False
        yield  # Update UI to show loading spinner

        # Save uploaded file to the local assets directory
        upload_data = await files[0].read()
        outline_path = rx.get_asset_path(files[0].filename)
        with open(outline_path, "wb") as f:
            f.write(upload_data)

        self.img_path = files[0].filename
        self.image_uploaded = True

        # Reveal the carousel with placeholder cards
        self.current_card = 0
        self.cards_visible = True
        self.is_processing = False

    def next_card(self):
        if self.current_card < self.num_cards - 1:
            self.current_card += 1

    def prev_card(self):
        if self.current_card > 0:
            self.current_card -= 1


# --- UI COMPONENTS ---
def carousel_card() -> rx.Component:
    """A single placeholder card centered between left/right arrow buttons."""
    return rx.hstack(
        rx.button(
            rx.icon("chevron-left", size=22),
            on_click=State.prev_card,
            disabled=State.is_first_card,
            variant="soft",
            size="3",
        ),
        rx.card(
            rx.vstack(
                rx.text(State.current_card_label, size="2", color_scheme="gray"),
                rx.heading(State.current_card_name, size="5"),
                rx.text("(placeholder card)", size="2", color_scheme="gray", italic=True),
                align_items="center",
                spacing="3",
                padding_y="6",
            ),
            variant="surface",
            width="320px",
            height="200px",
        ),
        rx.button(
            rx.icon("chevron-right", size=22),
            on_click=State.next_card,
            disabled=State.is_last_card,
            variant="soft",
            size="3",
        ),
        align_items="center",
        spacing="4",
        justify="center",
        width="100%",
    )


def index() -> rx.Component:
    """The main dashboard layout structure."""
    return rx.container(
        # Top Header
        rx.vstack(
            rx.heading("🦅 OrniTrack AI", size="8", margin_bottom="2"),
            rx.text("Enterprise Computer Vision Pipeline for Avian Species Classification", color_scheme="gray"),
            align_items="center",
            margin_y="6",
        ),

        rx.grid(
            # Left Column: Upload and Input Preview
            rx.vstack(
                rx.heading("Image Ingestion", size="4"),
                rx.upload(
                    rx.vstack(
                        rx.button("Select Bird Image", color_scheme="blue", variant="outline"),
                        rx.text("Drag and drop your image here", size="2", color_scheme="gray"),
                        align_items="center",
                        spacing="2"
                    ),
                    id="bird_uploader",
                    border="1px dashed var(--gray-6)",
                    padding="6",
                    radius="lg",
                    width="100%",
                ),
                rx.button(
                    "Run Model Inference",
                    on_click=State.handle_upload(rx.upload_files(upload_id="bird_uploader")),
                    loading=State.is_processing,
                    width="100%",
                    color_scheme="blue"
                ),
                rx.cond(
                    State.image_uploaded,
                    rx.box(
                        rx.heading("Uploaded Subject", size="2", margin_bottom="2"),
                        rx.image(src=State.img_path, width="100%", height="auto", radius="md"),
                        width="100%"
                    )
                ),
                spacing="4",
                align_items="start",
            ),

            # Right Column: Carousel of placeholder cards
            rx.vstack(
                rx.heading("Top 5 Predictions", size="4"),
                rx.cond(
                    State.cards_visible,
                    carousel_card(),
                    rx.center(
                        rx.text(
                            "Upload an image and run inference to view the top 5 classifications.",
                            color_scheme="gray",
                            italic=True,
                        ),
                        border="1px dashed var(--gray-4)",
                        radius="md",
                        width="100%",
                        height="200px",
                    ),
                ),
                spacing="4",
                align_items="start",
                width="100%",
            ),
            columns="2",
            spacing="6",
            width="100%",
        ),
        max_width="1050px",
        padding_x="4",
    )


# --- APP CONFIGURATION ---
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        radius="medium",
        accent_color="blue",
    )
)
app.add_page(index)
