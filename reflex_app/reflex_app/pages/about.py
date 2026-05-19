"""About page — OrniTrack AI project overview, motivation, and author."""

import reflex as rx

from ..components.home_widgets import gradient_heading
from ..components.layout import page_layout
from ..styles import GLASS_BG, GLASS_BLUR, GLASS_BORDER, ACCENT

_SCIENCE_PARROT = "https://cultofthepartyparrot.com/parrots/hd/scienceparrot.gif"
_AUTHOR_IMAGE = "/hamad_bird.JPG"


def _section_divider() -> rx.Component:
    return rx.box(height="1px", background="rgba(255, 255, 255, 0.1)", width="100%")


def _section_text(content: str) -> rx.Component:
    return rx.text(
        content,
        size="3",
        color="rgba(255, 255, 255, 0.85)",
        line_height="1.6",
    )


def _section_heading(title: str) -> rx.Component:
    return rx.heading(
        title,
        size="5",
        color="white",
        weight="bold",
    )


def _motivation_section() -> rx.Component:
    return rx.vstack(
        _section_heading("Motivation"),
        _section_text(
            "BirdML began as a passion project built around my love for birds. "
            "Birds are one of the most diverse and ecologically important groups of animals on Earth. "
            "Beyond their beauty, they play important roles in seed dispersal, pollination, pest control, "
            "disease prevention, and maintaining healthy ecosystems."
        ),
        _section_text(
            "At the same time, many bird species are facing serious threats from habitat loss, "
            "climate change, and other environmental pressures. By helping users identify bird species from images, "
            "this project aims to make birds feel more visible, familiar, and memorable. My hope is that a simple prediction "
            "can spark curiosity, encourage people to learn more about the species around them, and raise awareness about the "
            "importance of protecting biodiversity."
        ),
        spacing="3",
        width="100%",
    )


def _mlops_section() -> rx.Component:
    return rx.vstack(
        _section_heading("MLOps and Technical Purpose"),
        _section_text(
            "This project was also built to showcase my ability to develop and productionize an end-to-end machine learning system. "
            "The application uses an EfficientNetB3 convolutional neural network trained with transfer learning to classify bird images "
            "across 525 species. The model was trained using the yashikota/birds-525-species-image-classification dataset, "
            "which contains 89,885 bird images."
        ),
        _section_text(
            "The system includes a backend prediction API that loads the trained model and serves inference requests, "
            "as well as a frontend that allows users to upload bird images and receive predictions. The training pipeline was "
            "developed using AWS SageMaker, with the goal of creating infrastructure that can support future model improvements, "
            "additional bird species, and eventually a feedback system where users can rate prediction quality."
        ),
        rx.text(
            "Images uploaded by users are not collected on this site.",
            size="2",
            color="rgba(255, 255, 255, 0.55)",
            line_height="1.6",
        ),
        spacing="3",
        width="100%",
    )


def _about_me_section() -> rx.Component:
    return rx.vstack(
        _section_heading("About Me"),
        rx.flex(
            rx.vstack(
                _section_text(
                    "Hi, I'm Hamad Alajeel. I have a bachelor's and master's degree in Electrical Engineering "
                    "from UC San Diego, with a focus on Machine Learning and Data Science. I am passionate about ML engineering "
                    "and building systems that turn machine learning models into real, usable applications."
                ),
                _section_text(
                    "During my academic work, I built projects involving recommendation systems, computer vision, generative models, "
                    "and diffusion models. However, most of that work was completed in Jupyter and Google Colab notebooks, which did not "
                    "fully reflect the production reality of machine learning systems."
                ),
                _section_text(
                    "This project helped me bridge that gap. It gave me the opportunity to work beyond model training and focus on the full "
                    "ML lifecycle, including data handling, training pipelines, backend inference, frontend integration, cloud deployment, "
                    "and future system monitoring. My goal is to continue building practical AI and ML systems that are reliable, useful, and impactful."
                ),
                spacing="3",
                flex="1",
                width="100%",
            ),
            rx.image(
                src=_AUTHOR_IMAGE,
                alt="Hamad with birds",
                width=["160px", "180px", "200px"],
                height="auto",
                border_radius="12px",
                box_shadow="0 8px 24px rgba(99, 102, 241, 0.3)",
            ),
            direction=["column", "column", "row"],
            spacing="6",
            align_items=["center", "center", "flex-start"],
            width="100%",
        ),
        spacing="3",
        width="100%",
    )


def _contact_section() -> rx.Component:
    return rx.vstack(
        _section_heading("Contact"),
        _section_text(
            "For questions, feedback, or professional opportunities, feel free to reach out through "
        ),
        rx.hstack(
            rx.link(
                        "LinkedIn",
                        href="https://www.linkedin.com/in/hamad-alajeel-a6bb41210/",
                        is_external=True,
                        weight="bold",
                        text_decoration="none",
                        # Rainbow effect styling
                        background_image="linear-gradient(to right, #ff0055, #00ffcc, #9900ff)",
                        background_clip="text",
                        color="transparent",
                        # Hover effect styling
                        _hover={
                            "background_image": "none",
                            "background_clip": "unset",
                            "color": ACCENT, 
                            "text_decoration": "underline"
                        },
                    ),
            rx.text(" • ", color="rgba(255, 255, 255, 0.5)"),
            rx.link(
                "GitHub",
                href="https://github.com/Hamad-Alajeel",
                is_external=True,
                weight="bold",
                text_decoration="none",
                background_image="linear-gradient(to right, #ff0055, #00ffcc, #9900ff)",
                background_clip="text",
                color="transparent",
                _hover={
                    "background_image": "none",
                    "background_clip": "unset",
                    "color": ACCENT,
                    "text_decoration": "underline",
                },
            ),
            rx.text(" • ", color="rgba(255, 255, 255, 0.5)"),
            rx.link(
                "hamad.alajeel2019@gmail.com",
                href="mailto:hamad.alajeel2019@gmail.com",
                is_external=True,
                weight="bold",
                text_decoration="none",
                background_image="linear-gradient(to right, #ff0055, #00ffcc, #9900ff)",
                background_clip="text",
                color="transparent",
                _hover={
                    "background_image": "none",
                    "background_clip": "unset",
                    "color": ACCENT,
                    "text_decoration": "underline",
                },
            ),
            spacing="2",
            font_size="3",
        ),
        spacing="3",
        width="100%",
    )


def about() -> rx.Component:
    return page_layout(
        rx.vstack(
            gradient_heading(parrot_gif=_SCIENCE_PARROT, subtitle=None),
            rx.vstack(
                _motivation_section(),
                _section_divider(),
                _mlops_section(),
                _section_divider(),
                _about_me_section(),
                _section_divider(),
                _contact_section(),
                spacing="6",
                width="100%",
            ),
            spacing="7",
            align_items="center",
            width="100%",
            max_width="800px",
            padding_x="4",
            padding_top="10",
            padding_bottom="24",
        )
    )
