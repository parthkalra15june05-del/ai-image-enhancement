"""Streamlit entry point for ImageEnhance AI."""

from __future__ import annotations

import streamlit as st
import matplotlib.pyplot as plt

from utils.ai_tools import (
    ai_restore,
    get_subject_mask,
    image_quality_score,
    portrait_blur,
    quality_assessment,
    remove_background,
    replace_background,
    smart_auto_enhance,
    subject_enhance,
    subject_pop,
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
    resize_for_preview,
    visualize_color_space,
)

st.set_page_config(
    page_title="ImageEnhance AI | Image Editing Studio",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ImageEnhance AI")
st.subheader("AI-Powered Image Enhancement & Intelligent Editing Studio")
st.caption("Computer Graphics • Deep Learning • Image Processing")
st.markdown(
    """
    <style>
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        padding: .75rem;
        border-radius: .45rem;
    }
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] {
        color: #c9d1d9 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.35rem;
        font-weight: 700;
    }
    div[data-testid="stMetricDelta"] {
        color: #c9d1d9 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

editor_tab, ai_tab, analysis_tab = st.tabs(["🎨 Editor", "✨ AI Studio", "📊 Analysis"])

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, and JPEG.",
    )

    editor_settings = st.session_state.setdefault(
        "editor_settings",
        {
            "brightness": 1.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "sharpness": 1.0,
            "blur": 0.0,
            "filter_name": "Original",
            "transformation": "None",
            "processing_option": "None",
        },
    )
    with st.form("editor_controls"):
        with st.expander("Adjustments", expanded=True):
            brightness_input = st.slider("Brightness", 0.0, 2.0, editor_settings["brightness"], 0.05)
            contrast_input = st.slider("Contrast", 0.0, 2.0, editor_settings["contrast"], 0.05)
            saturation_input = st.slider("Saturation", 0.0, 2.0, editor_settings["saturation"], 0.05)
            sharpness_input = st.slider("Sharpness", 0.0, 3.0, editor_settings["sharpness"], 0.05)
            blur_input = st.slider("Blur", 0.0, 10.0, editor_settings["blur"], 0.5)

        with st.expander("Filters and transformations", expanded=True):
            filter_input = st.selectbox(
            "Choose a filter",
            ["Original", "Grayscale", "Sepia", "Gaussian Blur", "Edge Detection", "Cartoon"],
                index=["Original", "Grayscale", "Sepia", "Gaussian Blur", "Edge Detection", "Cartoon"].index(editor_settings["filter_name"]),
        )
            transformation_input = st.selectbox(
            "Choose a transformation",
            ["None", "Rotate left", "Rotate right", "Flip horizontal", "Flip vertical"],
                index=["None", "Rotate left", "Rotate right", "Flip horizontal", "Flip vertical"].index(editor_settings["transformation"]),
        )
        with st.expander("Advanced Enhancement", expanded=True):
            processing_input = st.selectbox(
            "Processing option",
            [
                "None",
                "Histogram Equalization",
                "CLAHE Enhancement",
                "Gaussian Denoising",
                "Median Denoising",
                "Convolution Sharpen",
            ],
                index=["None", "Histogram Equalization", "CLAHE Enhancement", "Gaussian Denoising", "Median Denoising", "Convolution Sharpen"].index(editor_settings["processing_option"]),
        )
        apply_editor = st.form_submit_button("Apply Adjustments", type="primary")

    if apply_editor:
        editor_settings.update(
            brightness=brightness_input,
            contrast=contrast_input,
            saturation=saturation_input,
            sharpness=sharpness_input,
            blur=blur_input,
            filter_name=filter_input,
            transformation=transformation_input,
            processing_option=processing_input,
        )
    brightness = editor_settings["brightness"]
    contrast = editor_settings["contrast"]
    saturation = editor_settings["saturation"]
    sharpness = editor_settings["sharpness"]
    blur = editor_settings["blur"]
    filter_name = editor_settings["filter_name"]
    transformation = editor_settings["transformation"]
    processing_option = editor_settings["processing_option"]

if uploaded_file is None:
    with editor_tab:
        st.info("Upload an image from the sidebar to begin editing.")
else:
    try:
        original_image = load_image(uploaded_file)
        preview_image = resize_for_preview(original_image)
        processing_key = (uploaded_file.file_id, brightness, contrast, saturation, sharpness, blur, filter_name, processing_option, transformation)
        if st.session_state.get("processing_key") != processing_key:
            processed_image = apply_adjustments(
                preview_image,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                sharpness=sharpness,
                blur=blur,
            )
            processed_image = apply_filter(processed_image, filter_name)
            processed_image = apply_processing_option(processed_image, processing_option)
            processed_image = apply_transformation(processed_image, transformation)
            st.session_state.processed_image = processed_image
            st.session_state.processing_key = processing_key
        else:
            processed_image = st.session_state.processed_image
    except (ValueError, NotImplementedError) as exc:
        st.error(str(exc))
    else:
        if st.session_state.get("ai_source_id") != uploaded_file.file_id:
            st.session_state.ai_source_id = uploaded_file.file_id
            st.session_state.pop("smart_result", None)
            st.session_state.pop("background_result", None)
            st.session_state.pop("background_error", None)
            st.session_state.pop("portrait_result", None)
            st.session_state.pop("restore_result", None)
            st.session_state.pop("restore_error", None)
            st.session_state.pop("quality_score_before", None)
            st.session_state.pop("quality_score_after", None)
            st.session_state.pop("subject_mask", None)
            st.session_state.pop("processing_key", None)
            st.session_state.pop("processed_image", None)

        with editor_tab:
            image_columns = st.columns(2)
            with image_columns[0]:
                st.subheader("Original")
                st.image(preview_image, use_container_width=True)
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
            st.header("✨ AI Restore")
            st.write("Automatically analyzes image quality and applies only the enhancements that are needed.")
            if st.button("Restore Image with AI", type="primary", key="restore_button"):
                with st.spinner("Analyzing image quality and restoring image..."):
                    try:
                        restored, original_metrics, final_metrics, applied, skipped = ai_restore(processed_image)
                        st.session_state.restore_result = (restored, original_metrics, final_metrics, applied, skipped)
                        st.session_state.quality_score_before = image_quality_score(processed_image)
                        st.session_state.quality_score_after = image_quality_score(restored)
                        st.session_state.restore_error = None
                    except RuntimeError as exc:
                        st.session_state.restore_result = None
                        st.session_state.restore_error = str(exc)
            if st.session_state.get("restore_error"):
                st.error(st.session_state.restore_error)
            if st.session_state.get("restore_result") is not None:
                restored, original_metrics, final_metrics, applied, skipped = st.session_state.restore_result
                restore_columns = st.columns(2)
                with restore_columns[0]:
                    st.subheader("Before")
                    st.image(processed_image, use_container_width=True)
                    st.caption(f"Original Resolution: {processed_image.width} x {processed_image.height}")
                with restore_columns[1]:
                    st.subheader("After")
                    st.image(restored, use_container_width=True)
                    st.caption(f"Restored Resolution: {restored.width} x {restored.height}")
                report_columns = st.columns(2)
                with report_columns[0]:
                    st.write("**Original metrics**")
                    st.write(f"Brightness: {original_metrics['brightness']:.2f}")
                    st.write(f"Contrast: {original_metrics['contrast']:.2f}")
                    st.write(f"Sharpness: {original_metrics['sharpness']:.2f}")
                with report_columns[1]:
                    st.write("**Final metrics**")
                    st.write(f"Brightness: {final_metrics['brightness']:.2f}")
                    st.write(f"Contrast: {final_metrics['contrast']:.2f}")
                    st.write(f"Sharpness: {final_metrics['sharpness']:.2f}")
                st.write("**AI Restore Report**")
                st.write("Image Assessment: " + ", ".join(f"{key} {value}" for key, value in {"Brightness": "Low" if original_metrics['brightness'] < 70 else "High" if original_metrics['brightness'] > 210 else "Good", "Contrast": "Low" if original_metrics['contrast'] < 38 else "High" if original_metrics['contrast'] > 85 else "Good", "Sharpness": "Low" if original_metrics['sharpness'] < 90 else "Moderate" if original_metrics['sharpness'] < 300 else "Good", "Resolution": "Low" if original_metrics['resolution'] < 1_000_000 else "Good"}.items()))
                st.write("Applied:")
                for operation in applied:
                    st.write(f"✓ {operation}")
                st.write("Skipped:")
                for operation in skipped:
                    st.write(f"• {operation}")
                score_columns = st.columns(2)
                score_columns[0].metric("ImageEnhance Quality Score Before", f"{st.session_state.quality_score_before:.1f}/100")
                score_columns[1].metric("ImageEnhance Quality Score After", f"{st.session_state.quality_score_after:.1f}/100")
                st.caption("Transparent heuristic based on brightness, contrast, dynamic range, and sharpness; not scientific or neural.")
                st.success("AI Restore completed with adaptive image-quality analysis.")
                st.download_button("Download restored PNG", image_to_png_bytes(restored), "imageenhance_restored.png", "image/png", key="download_restore")

            st.header("👤 AI Portrait Studio")
            st.write("Use the rembg neural segmentation mask to process the foreground and background separately.")
            portrait_mode = st.selectbox("Portrait effect", ["Portrait Blur", "Subject Enhance", "Subject Pop"], key="portrait_mode")
            if st.button("Apply Portrait Effect", key="portrait_button"):
                with st.spinner("Segmenting subject and applying portrait effect..."):
                    try:
                        portrait_functions = {"Portrait Blur": portrait_blur, "Subject Enhance": subject_enhance, "Subject Pop": subject_pop}
                        mask = st.session_state.get("subject_mask")
                        if mask is None:
                            mask = get_subject_mask(processed_image)
                            st.session_state.subject_mask = mask
                        result, mask = portrait_functions[portrait_mode](processed_image, mask)
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
                            mask = st.session_state.get("subject_mask")
                            if mask is None:
                                mask = get_subject_mask(processed_image)
                                st.session_state.subject_mask = mask
                            result, mask = replace_background(processed_image, replacement, mask)
                            st.session_state.portrait_result = (result, mask, "Replace Background")
                            st.session_state.portrait_error = None
                        except RuntimeError as exc:
                            st.session_state.portrait_error = str(exc)
            
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
                        mask = st.session_state.get("subject_mask")
                        if mask is None:
                            mask = get_subject_mask(processed_image)
                            st.session_state.subject_mask = mask
                        st.session_state.background_result = remove_background(processed_image, mask)
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

            with st.expander("How the AI works"):
                st.write("AI segmentation uses the optional pretrained rembg model, loaded only after an AI segmentation button is clicked.")
                st.write("Smart Auto Enhance and AI Restore use adaptive image-quality analysis with traditional CLAHE, gamma, denoising, and sharpening algorithms.")
                st.write("ImageEnhance Quality Score is a transparent heuristic, not a scientific measurement or neural-network score.")

        with analysis_tab:
            st.header("Image Analysis")
            analysis = image_analysis(preview_image)
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

            red_histogram, green_histogram, blue_histogram, gray_histogram = histogram_data(preview_image)
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
            st.image(visualize_color_space(preview_image, color_space), use_container_width=True)
