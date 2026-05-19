import os
import reflex as rx

config = rx.Config(
    app_name="reflex_app",
    api_url=os.environ.get("BACKEND_URL", "http://localhost:8001"),
)