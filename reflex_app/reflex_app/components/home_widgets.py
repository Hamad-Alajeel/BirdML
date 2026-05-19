"""All home-page UI building blocks.

Pulled out of reflex_app.py verbatim so the home page logic stays
unchanged — only the import path moved.
"""

import reflex as rx

from ..state import State
from ..styles import GRADIENT_TITLE


_DEFAULT_PARROT = "https://cultofthepartyparrot.com/parrots/hd/parrot.gif"
_HOME_SUBTITLE = (
    "Upload a bird photo, run inference, browse the top 5 predictions."
)


def gradient_heading(
    parrot_gif: str = _DEFAULT_PARROT,
    subtitle: str | None = _HOME_SUBTITLE,
) -> rx.Component:
    """The shared BirdML title block. The parrot gif (and optional subtitle
    underneath) varies per page — home uses the classic party parrot, About
    uses the science parrot, Catalogue uses the matrix parrot."""
    children: list[rx.Component] = [
        # Inject the rainbow_shift keyframes once at render time. Reflex has
        # no first-class keyframe helper, so we drop raw <style> here.
        rx.html(
            "<style>@keyframes rainbow_shift { "
            "0% { background-position: 0% 50%; } "
            "100% { background-position: 200% 50%; } "
            "}</style>"
        ),
        rx.hstack(
            rx.image(
                src=parrot_gif,
                height=rx.breakpoints(initial="48px", sm="56px", md="72px"),
                width=rx.breakpoints(initial="48px", sm="56px", md="72px"),
                style={"filter": "drop-shadow(0 4px 12px rgba(0, 0, 0, 0.6))"},
            ),
            rx.heading(
                "BirdML",
                size=rx.breakpoints(initial="7", sm="8", md="9"),
                style={
                    "background-image": GRADIENT_TITLE,
                    "background-size": "200% 100%",
                    "background-clip": "text",
                    "-webkit-background-clip": "text",
                    "-webkit-text-fill-color": "transparent",
                    "animation": "rainbow_shift 6s linear infinite",
                    "filter": "drop-shadow(0 2px 12px rgba(236, 72, 153, 0.45))",
                },
                weight="bold",
                letter_spacing="-0.02em",
            ),
            align="center",
            spacing=rx.breakpoints(initial="3", md="4"),
            justify="center",
            # Hstack has natural content width; margin auto centers it
            # within the parent vstack (works regardless of parent's align
            # prop or SSR/CSR hydration mismatch).
            margin_x="auto",
        ),
    ]
    if subtitle:
        children.append(
            rx.text(
                subtitle,
                color="rgba(255, 255, 255, 0.85)",
                size="3",
                text_align="center",
                style={"text-shadow": "0 1px 6px rgba(0, 0, 0, 0.6)"},
                margin_x="auto",
            )
        )
    return rx.vstack(*children, align="center", spacing="2", width="100%")


def image_preview() -> rx.Component:
    # Match the search-bar pattern: outer width=100% block, no rx.center
    # wrapper. Centering of max-width-capped children is via margin_x="auto"
    # so it doesn't depend on parent align-items behavior.
    return rx.cond(
        State.has_image,
        rx.image(
            src=State.img_data_uri,
            max_height="320px",
            max_width="100%",
            width="auto",
            height="auto",
            object_fit="contain",
            border_radius="16px",
            box_shadow="0 20px 50px -15px rgba(0, 0, 0, 0.7)",
            border="2px solid rgba(255, 255, 255, 0.15)",
            margin_x="auto",
            display="block",
        ),
        rx.vstack(
            rx.icon("image", size=40, color="rgba(255, 255, 255, 0.55)"),
            rx.text(
                "Your image will appear here",
                color="rgba(255, 255, 255, 0.75)",
                size="2",
                style={"text-shadow": "0 1px 4px rgba(0, 0, 0, 0.7)"},
            ),
            align="center",
            justify="center",
            spacing="2",
            width="100%",
            max_width="420px",
            margin_x="auto",
            height="220px",
            border="1px dashed rgba(255, 255, 255, 0.4)",
            border_radius="16px",
            background="rgba(0, 0, 0, 0.18)",
            backdrop_filter="blur(6px)",
        ),
    )


def upload_zone() -> rx.Component:
    # Match search-bar pattern: no rx.center wrapper. The inner element
    # (button or upload zone) is width=100% capped at 420px, centered via
    # margin_x="auto" so it doesn't depend on parent align behavior.
    return rx.cond(
        State.has_image,
        # When an image is loaded, replace the dropzone with a red
        # "Delete Upload" button so the user can clear and start over.
        rx.button(
            rx.hstack(
                rx.icon("trash-2", size=20),
                rx.text("Delete Upload", weight="bold"),
                spacing="2",
                align="center",
            ),
            on_click=State.clear_image,
            size="4",
            color_scheme="red",
            width="100%",
            max_width="420px",
            margin_x="auto",
            style={
                "cursor": "pointer",
                "box-shadow": "0 14px 30px -10px rgba(239, 68, 68, 0.6)",
            },
        ),
        rx.upload(
            rx.vstack(
                rx.icon("upload", size=28, color="#f0abfc"),
                rx.text(
                    "Drop a bird photo here",
                    weight="medium",
                    size="3",
                    color="white",
                    text_align="center",
                    style={"text-shadow": "0 1px 4px rgba(0, 0, 0, 0.6)"},
                ),
                rx.text(
                    "or click to browse",
                    size="2",
                    color="rgba(255, 255, 255, 0.7)",
                    text_align="center",
                ),
                align="center",
                justify="center",
                spacing="2",
                width="100%",
            ),
            id="bird_uploader",
            accept={"image/*": [".png", ".jpg", ".jpeg", ".webp"]},
            max_files=1,
            on_drop=State.handle_upload(rx.upload_files(upload_id="bird_uploader")),
            border="2px dashed rgba(240, 171, 252, 0.5)",
            border_radius="16px",
            padding="5",
            width="100%",
            max_width="420px",
            margin_x="auto",
            background="rgba(0, 0, 0, 0.18)",
            backdrop_filter="blur(6px)",
            _hover={
                "border_color": "#f0abfc",
                "background": "rgba(240, 171, 252, 0.1)",
            },
            transition="all 0.25s ease",
            # Force the underlying dropzone div to flex-center its child vstack.
            style={
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
                "text-align": "center",
            },
        ),
    )


def run_inference_button() -> rx.Component:
    return rx.button(
        rx.hstack(
            rx.icon("zap", size=20),
            rx.text("Run Inference", weight="bold"),
            spacing="2",
            align="center",
        ),
        on_click=State.run_inference,
        disabled=~State.has_image,
        size="4",
        color_scheme="iris",
        width="100%",
        max_width="420px",
        margin_x="auto",
        display="flex",
        style={
            "cursor": "pointer",
            "box-shadow": "0 14px 30px -10px rgba(99, 102, 241, 0.7)",
        },
    )


def nn_animation() -> rx.Component:
    """Pastel neural network with traveling-data effects on connections."""
    # 5 layers, 3-4-4-4-3 — extra hidden layer in the middle.
    layers = [3, 4, 4, 4, 3]
    layer_xs = [180, 290, 380, 470, 580]
    layer_colors = ["#f0abfc", "#e879f9", "#a78bfa", "#67e8f9", "#7dd3fc"]
    layer_glow = ["#f5d0fe", "#f0abfc", "#c4b5fd", "#a5f3fc", "#bae6fd"]

    canvas_height = 360

    def y_positions(n: int, is_middle_layer: bool = False) -> list[float]:
        spacing = canvas_height / (n + 1)
        base = [spacing * (i + 1) for i in range(n)]
        center = canvas_height / 2
        # 3-node layers (input/output): compress toward center.
        if n < 4:
            return [center + (y - center) * 0.78 for y in base]
        # Only the absolute middle layer gets the extra vertical stretch.
        if is_middle_layer:
            return [center + (y - center) * 1.18 for y in base]
        return base

    middle_idx = len(layers) // 2
    positions = [
        [(x, y) for y in y_positions(n, is_middle_layer=(i == middle_idx))]
        for i, (x, n) in enumerate(zip(layer_xs, layers))
    ]

    # Each layer-pair starts its color cycle slightly later so the rainbow
    # visibly propagates from input to output.
    rainbow_values = (
        "#ef4444;#f97316;#eab308;#22c55e;#06b6d4;#3b82f6;#a855f7;#ec4899;#ef4444"
    )
    line_parts: list[str] = []
    for i in range(len(positions) - 1):
        layer_delay = i * 0.4
        for (x1, y1) in positions[i]:
            for (x2, y2) in positions[i + 1]:
                line_parts.append(
                    f'<line x1="{x1}" y1="{y1:.1f}" x2="{x2}" y2="{y2:.1f}" '
                    f'stroke-width="1.6" stroke-linecap="round" opacity="0.85">'
                    f'<animate attributeName="stroke" '
                    f'values="{rainbow_values}" dur="3.5s" '
                    f'begin="{layer_delay:.2f}s" repeatCount="indefinite"/>'
                    f"</line>"
                )

    gradient_defs = []
    for idx, (color, glow) in enumerate(zip(layer_colors, layer_glow)):
        gradient_defs.append(
            f'<radialGradient id="g{idx}" cx="35%" cy="35%" r="65%">'
            f'<stop offset="0%" stop-color="{glow}"/>'
            f'<stop offset="100%" stop-color="{color}"/>'
            f"</radialGradient>"
        )

    circle_parts: list[str] = []
    for layer_idx, layer in enumerate(positions):
        for node_idx, (x, y) in enumerate(layer):
            delay = layer_idx * 0.15 + node_idx * 0.06
            circle_parts.append(
                f'<circle cx="{x}" cy="{y:.1f}" r="13" '
                f'fill="url(#g{layer_idx})" filter="url(#nodeGlow)">'
                f'<animate attributeName="r" values="11;16;11" dur="1.5s" '
                f'begin="{delay:.2f}s" repeatCount="indefinite" '
                f'calcMode="spline" keyTimes="0;0.5;1" '
                f'keySplines="0.4 0 0.2 1;0.4 0 0.2 1"/>'
                f'<animate attributeName="opacity" values="0.75;1;0.75" dur="1.5s" '
                f'begin="{delay:.2f}s" repeatCount="indefinite"/>'
                f"</circle>"
            )

    svg = (
        f'<svg viewBox="0 0 720 {canvas_height}" xmlns="http://www.w3.org/2000/svg" '
        'style="width: 100%; max-width: none; height: auto; display: block;">'
        '<defs>'
        + "".join(gradient_defs)
        + '<filter id="nodeGlow" x="-100%" y="-100%" width="300%" height="300%">'
        '<feGaussianBlur stdDeviation="5" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        '</filter>'
        '</defs>'
        '<g>'
        + "".join(line_parts)
        + "</g>"
        + "".join(circle_parts)
        + "</svg>"
    )

    return rx.vstack(
        rx.box(
            rx.html(svg),
            width="55vw",
            max_width="750px",
            margin_x="auto",
        ),
        rx.hstack(
            rx.spinner(size="3"),
            rx.text(
                "Running inference...",
                size="5",
                weight="bold",
                color="white",
                style={"text-shadow": "0 2px 8px rgba(0, 0, 0, 0.7)"},
            ),
            spacing="3",
            align="center",
            # hstack has natural content width — margin auto centers it
            # within the parent vstack regardless of parent's align prop.
            margin_x="auto",
        ),
        rx.text(
            "Forward pass through the network",
            size="2",
            color="rgba(255, 255, 255, 0.7)",
            italic=True,
            style={"text-shadow": "0 1px 4px rgba(0, 0, 0, 0.7)"},
            margin_x="auto",
        ),
        spacing="5",
        align="center",
        width="100%",
        padding_y="6",
    )


def carousel_card() -> rx.Component:
    card = rx.box(
        rx.flex(
            # LEFT/TOP: bird photo. Side-by-side with text on desktop,
            # stacked above the text on mobile.
            rx.flex(
                rx.image(
                    src=State.current_card_image,
                    width=rx.breakpoints(initial="140px", sm="180px", md="224px"),
                    height=rx.breakpoints(initial="140px", sm="180px", md="224px"),
                    object_fit="cover",
                    border="4px solid rgba(167, 139, 250, 0.75)",
                    border_radius="14px",
                    flex_shrink="0",
                ),
                # rx.flex with direction=column + align=center + justify=center
                # gives us both vertical and horizontal centering via Radix
                # props (reliable across SSR/CSR) instead of a CSS style dict.
                direction="column",
                align="center",
                justify="center",
                width=rx.breakpoints(initial="100%", md="260px"),
                flex_shrink="0",
                padding=rx.breakpoints(initial="14px 0 0 0", md="0"),
                align_self="stretch",
            ),
            # RIGHT/BOTTOM: prediction details
            rx.vstack(
                rx.badge(
                    State.current_rank,
                    color_scheme="iris",
                    variant="soft",
                    size="3",
                    radius="full",
                ),
                rx.heading(
                    State.current_card_name,
                    size=rx.breakpoints(initial="5", md="6"),
                    weight="bold",
                    color="white",
                ),
                rx.vstack(
                    rx.hstack(
                        rx.text(
                            "Confidence",
                            size="2",
                            color="rgba(255, 255, 255, 0.65)",
                        ),
                        rx.text(
                            State.current_card_confidence_pct,
                            size="2",
                            weight="bold",
                            color="#a78bfa",
                        ),
                        width="100%",
                        justify="between",
                    ),
                    rx.progress(
                        value=State.current_card_confidence_bar,
                        color_scheme="iris",
                        size="2",
                        width="100%",
                    ),
                    spacing="1",
                    width="100%",
                ),
                rx.text(
                    State.current_card_description,
                    size="2",
                    color="rgba(255, 255, 255, 0.80)",
                    line_height="1.7",
                ),
                spacing=rx.breakpoints(initial="3", md="4"),
                flex="1",
                align_items="start",
                # Tighter padding on mobile, original spacing on desktop.
                padding=rx.breakpoints(initial="14px", md="36px 40px 36px 48px"),
                # Scroll inside the column if a description is too long for
                # the fixed-height card, so the image stays in a consistent
                # vertical position from card to card.
                style={"overflow-y": "auto"},
            ),
            direction=rx.breakpoints(initial="column", md="row"),
            spacing="0",
            # Radix prop name rather than CSS pass-through so the
            # class-based default doesn't override.
            align=rx.breakpoints(initial="center", md="stretch"),
            width="100%",
            height="100%",
        ),
        overflow="hidden",
        border_radius="16px",
        width=rx.breakpoints(initial="100%", md="680px"),
        max_width="680px",
        height=rx.breakpoints(initial="auto", md="380px"),
        border="1px solid rgba(167, 139, 250, 0.35)",
        background="rgba(15, 13, 24, 0.42)",
        style={
            "backdrop-filter": "blur(14px)",
            "transition": "transform 0.25s ease, border-color 0.25s ease",
            "box-shadow": "0 25px 60px -20px rgba(99, 102, 241, 0.5)",
        },
        _hover={
            "transform": "translateY(-3px)",
            "border-color": "rgba(167, 139, 250, 0.7)",
        },
    )

    return rx.vstack(
        rx.flex(
            rx.button(
                rx.icon("chevron-left", size=28),
                on_click=State.prev_card,
                disabled=State.is_first_card,
                variant="soft",
                size="4",
                color_scheme="gray",
                style={"cursor": "pointer", "border-radius": "9999px"},
            ),
            card,
            rx.button(
                rx.icon("chevron-right", size=28),
                on_click=State.next_card,
                disabled=State.is_last_card,
                variant="soft",
                size="4",
                color_scheme="gray",
                style={"cursor": "pointer", "border-radius": "9999px"},
            ),
            direction=rx.breakpoints(initial="column", md="row"),
            align="center",
            spacing=rx.breakpoints(initial="3", md="5"),
            justify="center",
            width="100%",
        ),
        rx.hstack(
            rx.foreach(
                [0, 1, 2, 3, 4],
                lambda i: rx.box(
                    width="32px",
                    height="5px",
                    border_radius="full",
                    background=rx.cond(
                        State.current_card == i,
                        "#a78bfa",
                        "rgba(255, 255, 255, 0.25)",
                    ),
                    transition="background 0.2s ease",
                ),
            ),
            spacing="2",
            justify="center",
            # hstack has natural content width — margin auto centers it
            margin_x="auto",
        ),
        rx.text(
            State.position_label,
            size="2",
            color="rgba(255, 255, 255, 0.7)",
            style={"text-shadow": "0 1px 4px rgba(0, 0, 0, 0.7)"},
            margin_x="auto",
        ),
        rx.button(
            rx.hstack(
                rx.icon("x", size=18),
                rx.text("Exit", weight="medium"),
                spacing="2",
            ),
            on_click=State.exit_to_landing,
            variant="outline",
            size="3",
            color_scheme="gray",
            margin_top="4",
            margin_x="auto",
            style={"cursor": "pointer", "color": "white"},
        ),
        spacing="5",
        align="center",
        width="100%",
    )


def low_confidence_view() -> rx.Component:
    """Gate shown when every top-5 prediction is below 50% confidence.
    Lets the user either bail (re-upload a better photo) or proceed anyway
    to see the model's best guesses."""
    return rx.vstack(
        rx.image(
            src="/budgie_thinking.gif",
            width=rx.breakpoints(initial="240px", md="320px"),
            height="auto",
            border_radius="14px",
            box_shadow="0 20px 50px -15px rgba(0, 0, 0, 0.7)",
        ),
        rx.heading(
            "Hmm, the model isn't sure",
            size=rx.breakpoints(initial="5", md="6"),
            color="white",
            weight="bold",
            text_align="center",
            style={"text-shadow": "0 2px 8px rgba(0, 0, 0, 0.7)"},
        ),
        rx.text(
            "The confidence levels are too low for the image uploaded. "
            "Make sure to upload an image of a bird centered in the middle. "
            "Do you want to proceed with the model's predictions?",
            size="3",
            color="rgba(255, 255, 255, 0.85)",
            text_align="center",
            line_height="1.6",
            max_width="520px",
            style={"text-shadow": "0 1px 4px rgba(0, 0, 0, 0.6)"},
        ),
        rx.hstack(
            rx.button(
                rx.hstack(
                    rx.icon("arrow-left", size=18),
                    rx.text("Back", weight="medium"),
                    spacing="2",
                ),
                on_click=State.exit_to_landing,
                variant="outline",
                size="3",
                color_scheme="gray",
                style={"cursor": "pointer", "color": "white"},
            ),
            rx.button(
                rx.hstack(
                    rx.text("Proceed", weight="bold"),
                    rx.icon("arrow-right", size=18),
                    spacing="2",
                ),
                on_click=State.proceed_to_results,
                size="3",
                color_scheme="iris",
                style={
                    "cursor": "pointer",
                    "box-shadow": "0 14px 30px -10px rgba(99, 102, 241, 0.7)",
                },
            ),
            spacing="4",
            justify="center",
            wrap="wrap",
        ),
        spacing="5",
        align="center",
        width="100%",
        padding_y="4",
    )


def landing_view() -> rx.Component:
    return rx.vstack(
        image_preview(),
        upload_zone(),
        run_inference_button(),
        rx.cond(
            State.has_error,
            rx.callout(
                State.inference_error,
                icon="triangle_alert",
                color_scheme="red",
                size="2",
                width="100%",
                max_width="420px",
                margin_x="auto",
            ),
            rx.fragment(),
        ),
        spacing="5",
        align="center",
        width="100%",
    )
