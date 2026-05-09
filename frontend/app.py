"""Streamlit frontend for the Bird Species Classifier."""

import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Bird Species Classifier", page_icon="🐦")
st.title("Bird Species Classifier")
st.write("Upload a photo of a bird to identify its species.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Identifying species..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                params={"k": 5},
                timeout=30,
            )
            response.raise_for_status()
            predictions = response.json()["predictions"]

            st.subheader("Top 5 Predictions")
            for i, pred in enumerate(predictions, 1):
                name = pred.get("name", f"Class {pred['label']}")
                score = pred["score"]
                st.write(f"**{i}. {name}**")
                st.progress(score, text=f"{score * 100:.1f}%")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Is it running?")
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e}")
