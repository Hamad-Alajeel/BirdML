import reflex as rx
import random  # Used to simulate model outputs

# --- STATE MANAGEMENT ---
class State(rx.State):
    """Handles the application logic, image upload, and model execution."""
    # Track upload status and results
    is_processing: bool = False
    image_uploaded: bool = False
    img_path: str = ""
    
    # Store top 5 predictions as a list of dictionaries
    top_predictions: list[dict] = []
    
    # Mock model latency metric for MLOps tracking
    inference_time: str = "0.00ms"

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Processes the uploaded bird image and triggers model inference."""
        if not files:
            return
            
        self.is_processing = True
        self.image_uploaded = False
        yield # Update UI to show loading spinner
        
        # 1. Save uploaded file to the local assets directory
        upload_data = await files[0].read()
        outline_path = rx.get_asset_path(files[0].filename)
        with open(outline_path, "wb") as f:
            f.write(upload_data)
            
        self.img_path = files[0].filename
        self.image_uploaded = True

        # 2. Simulate Machine Learning Model Inference
        # In production, replace this block with your PyTorch/TensorFlow/YOLO code:
        # result = my_bird_model(outline_path)
        import time
        start_time = time.time()
        
        # Mocking model response logic
        birds = ["Bald Eagle", "Peregrine Falcon", "Red-tailed Hawk", "Osprey", "Golden Eagle", "Barn Owl", "Blue Jay"]
        selected_birds = random.sample(birds, 5)
        
        # Generate 5 random descending probabilities that sum close to 100%
        probs = sorted([random.randint(50, 95), random.randint(20, 45), random.randint(10, 25), random.randint(5, 15), random.randint(1, 5)], reverse=True)
        
        self.top_predictions = [
            {"rank": i+1, "species": selected_birds[i], "confidence": f"{probs[i]}%"}
            for i in range(5)
        ]
        
        # Calculate mock telemetry metrics
        self.inference_time = f"{(time.time() - start_time + 0.124) * 1000:.2f}ms"
        self.is_processing = False


# --- UI COMPONENTS ---
def prediction_card(bird: dict) -> rx.Component:
    """Component template for a single styled prediction card."""
    return rx.card(
        rx.hstack(
            rx.badge(
                bird["rank"], 
                variant="solid", 
                color_scheme="blue", 
                radius="full",
                size="2"
            ),
            rx.vstack(
                rx.text(bird["species"], weight="bold", font_size="1rem"),
                rx.text(f"Confidence: {bird['confidence']}", size="2", color_scheme="gray"),
                align_items="start",
                spacing="1"
            ),
            justify="start",
            align_items="center",
            spacing="3",
        ),
        variant="surface",
        width="100%",
        padding="4",
    )

def index() -> rx.Component:
    """The main dashboard layout structure."""
    return rx.container(
        # Top Header Banner
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
                # Show uploaded image preview if available
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
            
            # Right Column: Model Output Cards & MLOps Telemetry
            rx.vstack(
                rx.hstack(
                    rx.heading("Model Telemetry", size="4"),
                    rx.badge(f"Latency: {State.inference_time}", color_scheme="green", variant="surface"),
                    justify="between",
                    width="100%"
                ),
                
                # Dynamic rendering conditional block
                rx.cond(
                    State.top_predictions,
                    # Iterate and render prediction_card for each item in the top_predictions array
                    rx.vstack(
                        rx.foreach(State.top_predictions, prediction_card),
                        width="100%",
                        spacing="3"
                    ),
                    # Fallback message before user uploads anything
                    rx.center(
                        rx.text("Upload a bird image to view top 5 classifications.", color_scheme="gray", italic=True),
                        border="1px dashed var(--gray-4)",
                        radius="md",
                        width="100%",
                        height="200px"
                    )
                ),
                spacing="4",
                align_items="start",
            ),
            columns="2",
            spacing="6",
            width="100%",
        ),
        max_width="1050px",
        padding_x="4"
    )

# --- APP CONFIGURATION ---
app = rx.App(
    theme=rx.theme(
        appearance="dark", 
        has_background=True, 
        radius="medium", 
        accent_color="blue"
    )
)
app.add_page(index)
