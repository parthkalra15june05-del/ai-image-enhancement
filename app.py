"""Streamlit entry point for ImageEnhance AI."""

from __future__ import annotations

import streamlit as st
import matplotlib.pyplot as plt

from utils.filters import apply_filter
from utils.image_processing import (
    apply_adjustments,
    apply_processing_option,
    apply_transformation,
    histogram_data,
    image_analysis,
    image_metadata,
    image_to_png_bytes,
    load_image,
    visualize_color_space,
)

st.set_page_config(
    page_title="ImageEnhance AI | Image Editing Studio",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ImageEnhance AI")
st.caption("AI-Powered Image Enhancement & Editing Studio")

editor_tab, analysis_tab = st.tabs(["🎨 Editor", "📊 Analysis"])

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, and JPEG.",
    )

    with st.expander("Adjustments", expanded=True):
        brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.05)
        contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.05)
        saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05)
        sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.05)
        blur = st.slider("Blur", 0.0, 10.0, 0.0, 0.5)

    with st.expander("Filters and transformations", expanded=True):
        filter_name = st.selectbox(
            "Choose a filter",
            ["Original", "Grayscale", "Sepia", "Gaussian Blur", "Edge Detection", "Cartoon"],
        )
        transformation = st.selectbox(
            "Choose a transformation",
            ["None", "Rotate left", "Rotate right", "Flip horizontal", "Flip vertical"],
        )

    with st.expander("Advanced Enhancement", expanded=True):
        processing_option = st.selectbox(
            "Processing option",
            [
                "None",
                "Histogram Equalization",
                "CLAHE Enhancement",
                "Gaussian Denoising",
                "Median Denoising",
                "Convolution Sharpen",
            ],
        )

if uploaded_file is None:
    with editor_tab:
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
        processed_image = apply_processing_option(processed_image, processing_option)
        processed_image = apply_transformation(processed_image, transformation)
    except (ValueError, NotImplementedError) as exc:
        st.error(str(exc))
    else:
        with editor_tab:
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

        with analysis_tab:
            st.header("Image Analysis")
            analysis = image_analysis(original_image)
            analysis_labels = {
                "Aspect ratio": "Aspect Ratio",
                "Average brightness": "Average Brightness",
                "Contrast value": "Contrast",
                "Image mode": "Image Mode",
                "Mean Red channel": "Mean Red",
                "Mean Green channel": "Mean Green",
                "Mean Blue channel": "Mean Blue",
            }
            metric_items = list(analysis.items())
            for start in range(0, len(metric_items), 5):
                metric_columns = st.columns(min(5, len(metric_items) - start))
                for column, (label, value) in zip(metric_columns, metric_items[start : start + 5]):
                    display_label = analysis_labels.get(label, label)
                    display_value = f"{value:.2f}" if isinstance(value, float) else value
                    column.metric(display_label, display_value)

            red_histogram, green_histogram, blue_histogram, gray_histogram = histogram_data(original_image)
            intensity = range(256)
            st.header("RGB Histogram")
            rgb_figure, rgb_axis = plt.subplots(figsize=(10, 3.5))
            # A histogram shows the frequency of every image intensity from 0 to 255.
            rgb_axis.plot(intensity, red_histogram, color="red", label="Red", linewidth=1)
            rgb_axis.plot(intensity, green_histogram, color="green", label="Green", linewidth=1)
            rgb_axis.plot(intensity, blue_histogram, color="blue", label="Blue", linewidth=1)
            rgb_axis.set(xlim=(0, 255), xlabel="Intensity", ylabel="Pixel count")
            rgb_axis.legend()
            rgb_axis.grid(alpha=0.2)
            st.pyplot(rgb_figure, use_container_width=True)
            plt.close(rgb_figure)

            st.header("Grayscale Histogram")
            gray_figure, gray_axis = plt.subplots(figsize=(10, 3.5))
            gray_axis.plot(intensity, gray_histogram, color="dimgray", linewidth=1)
            gray_axis.fill_between(intensity, gray_histogram, color="gray", alpha=0.2)
            gray_axis.set(xlim=(0, 255), xlabel="Intensity", ylabel="Pixel count")
            gray_axis.grid(alpha=0.2)
            st.pyplot(gray_figure, use_container_width=True)
            plt.close(gray_figure)

            st.header("Color Space Visualization")
            color_space = st.radio("Color space", ["RGB", "HSV", "Grayscale"], horizontal=True)
            st.image(visualize_color_space(original_image, color_space), use_container_width=True)
