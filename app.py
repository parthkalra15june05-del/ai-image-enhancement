"""Streamlit entry point for ImageEnhance AI."""

from __future__ import annotations

import streamlit as st
import matplotlib.pyplot as plt

from utils.ai_tools import (
    classify_scene,
    generate_caption,
    portrait_blur,
    quality_assessment,
    remove_background,
    replace_background,
    scene_recommendations,
    smart_auto_enhance,
    subject_enhance,
    subject_pop,
    super_resolve,
)
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

editor_tab, ai_tab, analysis_tab = st.tabs(["🎨 Editor", "✨ AI Studio", "📊 Analysis"])

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
        if st.session_state.get("ai_source_id") != uploaded_file.file_id:
            st.session_state.ai_source_id = uploaded_file.file_id
            st.session_state.pop("smart_result", None)
            st.session_state.pop("background_result", None)
            st.session_state.pop("background_error", None)
            st.session_state.pop("portrait_result", None)
            st.session_state.pop("super_resolution_result", None)
            st.session_state.pop("caption_result", None)

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

        with ai_tab:
            st.header("🚀 AI Super Resolution")
            st.write("Use the pretrained Swin2SR neural network to enhance the image at 2x resolution.")
            sr_columns = st.columns(2)
            with sr_columns[0]:
                st.caption(f"Before: {original_image.width} x {original_image.height}")
                st.image(processed_image, use_container_width=True)
            with sr_columns[1]:
                if st.button("2x Enhance", type="primary", key="super_resolution_button"):
                    with st.spinner("Loading Swin2SR and enhancing image..."):
                        try:
                            st.session_state.super_resolution_result = super_resolve(processed_image)
                            st.session_state.super_resolution_error = None
                        except RuntimeError as exc:
                            st.session_state.super_resolution_result = None
                            st.session_state.super_resolution_error = str(exc)
                if st.session_state.get("super_resolution_error"):
                    st.error(st.session_state.super_resolution_error)
                if st.session_state.get("super_resolution_result") is not None:
                    result = st.session_state.super_resolution_result
                    st.caption(f"After: {result.width} x {result.height}")
                    st.image(result, use_container_width=True)
                    st.download_button("Download 2x enhanced PNG", image_to_png_bytes(result), "imageenhance_2x.png", "image/png", key="download_sr")

            st.header("👤 AI Portrait Studio")
            st.write("Use the rembg neural segmentation mask to process the foreground and background separately.")
            portrait_mode = st.selectbox("Portrait effect", ["Portrait Blur", "Subject Enhance", "Subject Pop"], key="portrait_mode")
            if st.button("Apply Portrait Effect", key="portrait_button"):
                with st.spinner("Segmenting subject and applying portrait effect..."):
                    try:
                        portrait_functions = {"Portrait Blur": portrait_blur, "Subject Enhance": subject_enhance, "Subject Pop": subject_pop}
                        result, mask = portrait_functions[portrait_mode](processed_image)
                        st.session_state.portrait_result = (result, mask, portrait_mode)
                        st.session_state.portrait_error = None
                    except RuntimeError as exc:
                        st.session_state.portrait_result = None
                        st.session_state.portrait_error = str(exc)
            if st.session_state.get("portrait_error"):
                st.error(st.session_state.portrait_error)
            if st.session_state.get("portrait_result") is not None:
                result, mask, effect_name = st.session_state.portrait_result
                before_after = st.columns(2)
                with before_after[0]:
                    st.caption("Before")
                    st.image(processed_image, use_container_width=True)
                with before_after[1]:
                    st.caption(f"After: {effect_name}")
                    st.image(result, use_container_width=True)
                st.success("AI Subject Segmentation: Completed. Foreground preserved and background processed separately.")
                st.download_button("Download portrait result", image_to_png_bytes(result), "imageenhance_portrait.png", "image/png", key="download_portrait")

            replacement_file = st.file_uploader("Replacement background", type=["png", "jpg", "jpeg"], key="replacement_background")
            if st.button("Replace Background", key="replace_background_button"):
                if replacement_file is None:
                    st.warning("Upload a replacement background first.")
                else:
                    with st.spinner("Segmenting subject and replacing background..."):
                        try:
                            replacement = load_image(replacement_file)
                            result, mask = replace_background(processed_image, replacement)
                            st.session_state.portrait_result = (result, mask, "Replace Background")
                            st.session_state.portrait_error = None
                        except RuntimeError as exc:
                            st.session_state.portrait_error = str(exc)
            
            st.header("🧠 AI Image Understanding")
            st.write("Generate a scene description with pretrained BLIP; editing advice below is deterministic and metric-based.")
            if st.button("Analyze with AI", type="primary", key="caption_button"):
                with st.spinner("Loading BLIP and analyzing scene..."):
                    try:
                        caption = generate_caption(processed_image)
                        scene = classify_scene(caption)
                        st.session_state.caption_result = (caption, scene, scene_recommendations(scene, processed_image))
                        st.session_state.caption_error = None
                    except RuntimeError as exc:
                        st.session_state.caption_result = None
                        st.session_state.caption_error = str(exc)
            if st.session_state.get("caption_error"):
                st.error(st.session_state.caption_error)
            if st.session_state.get("caption_result") is not None:
                caption, scene, recommendations = st.session_state.caption_result
                st.subheader("AI Scene Description")
                st.write(caption)
                st.write(f"AI detected: **{scene}**")
                st.caption("Suggested edits are rule-based recommendations derived from the AI scene category and image metrics.")
                st.write("Suggested edits:")
                for recommendation in recommendations:
                    st.write(f"- {recommendation}")

            st.header("✨ Smart Auto Enhance")
            st.write("Adaptive, rule-based corrections based on measurable image properties.")
            smart_columns = st.columns(2)
            with smart_columns[0]:
                st.caption("Current image")
                st.image(processed_image, use_container_width=True)
            with smart_columns[1]:
                if st.button("Run Smart Auto Enhance", type="primary", key="smart_enhance_button"):
                    enhanced, improvements = smart_auto_enhance(processed_image)
                    st.session_state.smart_result = (enhanced, improvements)
                if "smart_result" in st.session_state:
                    enhanced_image, improvements = st.session_state.smart_result
                    st.caption("Enhanced image")
                    st.image(enhanced_image, use_container_width=True)
                    st.write("Applied improvements:")
                    for improvement in improvements:
                        st.write(f"- {improvement}")
                    st.download_button(
                        "Download auto-enhanced PNG",
                        data=image_to_png_bytes(enhanced_image),
                        file_name="imageenhance_auto_enhanced.png",
                        mime="image/png",
                        key="download_smart_result",
                    )

            st.header("🧍 Remove Background")
            st.write("Use the pretrained rembg model to isolate the foreground on a transparent canvas.")
            if st.button("Remove Background", key="remove_background_button"):
                with st.spinner("Removing background with the AI model..."):
                    try:
                        st.session_state.background_result = remove_background(processed_image)
                        st.session_state.background_error = None
                    except RuntimeError as exc:
                        st.session_state.background_result = None
                        st.session_state.background_error = str(exc)
            if st.session_state.get("background_error"):
                st.error(st.session_state.background_error)
            if st.session_state.get("background_result") is not None:
                st.image(st.session_state.background_result, caption="Transparent result", use_container_width=True)
                st.download_button(
                    "Download background-removed PNG",
                    data=image_to_png_bytes(st.session_state.background_result),
                    file_name="imageenhance_no_background.png",
                    mime="image/png",
                    key="download_background_result",
                )

            st.header("🔍 Image Quality Assistant")
            st.write("Review measurable image properties and receive local enhancement recommendations.")
            quality = quality_assessment(processed_image)
            st.metric("Image Quality", quality["summary"])
            quality_columns = st.columns(4)
            for column, (label, value) in zip(quality_columns, quality["labels"].items()):
                column.metric(label, value)
            quality_metrics = quality["metrics"]
            numeric_columns = st.columns(5)
            numeric_values = [
                ("Brightness value", quality_metrics["brightness"]),
                ("Contrast value", quality_metrics["contrast"]),
                ("Sharpness value", quality_metrics["sharpness"]),
                ("Dynamic range", quality_metrics["dynamic_range"]),
                ("Resolution", f"{int(quality_metrics['resolution']):,} px"),
            ]
            for column, (label, value) in zip(numeric_columns, numeric_values):
                column.metric(label, f"{value:.2f}" if isinstance(value, float) else value)
            st.write("Recommendations:")
            for recommendation in quality["recommendations"]:
                st.write(f"- {recommendation}")

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
