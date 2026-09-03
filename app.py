"""Streamlit entry point for ImageEnhance AI."""

from __future__ import annotations

import streamlit as st

from utils.filters import apply_filter
from utils.image_processing import (
    apply_adjustments,
    apply_transformation,
    image_metadata,
    image_to_png_bytes,
    load_image,
)

st.set_page_config(
    page_title="ImageEnhance AI | Image Editing Studio",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ImageEnhance AI")
st.caption("AI-Powered Image Enhancement & Editing Studio")

with st.sidebar:
    st.header("Edit controls")
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, and JPEG.",
    )

    st.subheader("Adjustments")
    brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.05)
    contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.05)
    saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05)
    sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.05)
    blur = st.slider("Blur", 0.0, 10.0, 0.0, 0.5)

    st.subheader("Filter")
    filter_name = st.selectbox(
        "Choose a filter",
        [
            "Original",
            "Grayscale",
            "Sepia",
            "Gaussian Blur",
            "Edge Detection",
            "Cartoon",
        ],
    )

    st.subheader("Transformation")
    transformation = st.selectbox(
        "Choose a transformation",
        ["None", "Rotate left", "Rotate right", "Flip horizontal", "Flip vertical"],
    )

if uploaded_file is None:
    st.info("Upload an image from the sidebar to begin editing.")
else:
    try:
        original_image = load_image(uploaded_file)
        processed_image = apply_adjustments(
            original_image,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            sharpness=sharpness,
            blur=blur,
        )
        processed_image = apply_filter(processed_image, filter_name)
        processed_image = apply_transformation(processed_image, transformation)
    except (ValueError, NotImplementedError) as exc:
        st.error(str(exc))
    else:
        image_columns = st.columns(2)
        with image_columns[0]:
            st.subheader("Original")
            st.image(original_image, use_container_width=True)
        with image_columns[1]:
            st.subheader("Processed")
            st.image(processed_image, use_container_width=True)

        st.subheader("Image information")
        metadata_columns = st.columns(5)
        for column, (label, value) in zip(metadata_columns, image_metadata(original_image).items()):
            column.metric(label, value)

        st.download_button(
            "Download processed image",
            data=image_to_png_bytes(processed_image),
            file_name="imageenhance_processed.png",
            mime="image/png",
            type="primary",
        )
