"""General image loading, adjustment, transformation, and export helpers."""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError


def resize_for_preview(image: Image.Image, max_dimension: int = 900) -> Image.Image:
    """Return a copy bounded for responsive previews and analysis."""
    preview = image.copy()
    preview.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    return preview


def load_image(uploaded_file) -> Image.Image:
    """Load an uploaded file as an independent RGB/RGBA Pillow image."""
    try:
        image = Image.open(uploaded_file)
        image.load()
        return image.convert("RGBA" if "A" in image.getbands() else "RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


def apply_adjustments(
    image: Image.Image,
    brightness: float,
    contrast: float,
    saturation: float,
    sharpness: float,
    blur: float,
) -> Image.Image:
    """Apply enhancement controls to a copy of the image."""
    adjusted = image.copy()
    adjusted = ImageEnhance.Brightness(adjusted).enhance(brightness)
    adjusted = ImageEnhance.Contrast(adjusted).enhance(contrast)
    adjusted = ImageEnhance.Color(adjusted).enhance(saturation)
    adjusted = ImageEnhance.Sharpness(adjusted).enhance(sharpness)
    if blur > 0:
        adjusted = adjusted.filter(ImageFilter.GaussianBlur(radius=blur))
    return adjusted


def apply_transformation(image: Image.Image, transformation: str) -> Image.Image:
    """Apply one selected geometric transformation."""
    transformations = {
        "Rotate left": Image.Transpose.ROTATE_90,
        "Rotate right": Image.Transpose.ROTATE_270,
        "Flip horizontal": Image.Transpose.FLIP_LEFT_RIGHT,
        "Flip vertical": Image.Transpose.FLIP_TOP_BOTTOM,
    }
    if transformation == "None":
        return image.copy()
    if transformation not in transformations:
        raise ValueError(f"Unsupported transformation: {transformation}")
    return image.transpose(transformations[transformation])


def apply_processing_option(image: Image.Image, option: str) -> Image.Image:
    """Apply an OpenCV enhancement without modifying the input image."""
    if option == "None":
        return image.copy()
    rgb = np.asarray(image.convert("RGB"))

    if option == "Histogram Equalization":
        # Equalize luminance only, preserving the original color relationships.
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    elif option == "CLAHE Enhancement":
        # CLAHE enhances local lightness while limiting noise amplification.
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lightness, green_red, blue_yellow = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.merge((clahe.apply(lightness), green_red, blue_yellow))
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    elif option == "Gaussian Denoising":
        result = cv2.GaussianBlur(rgb, (5, 5), 0)
    elif option == "Median Denoising":
        result = cv2.medianBlur(rgb, 5)
    elif option == "Convolution Sharpen":
        # Convolution applies this 3x3 kernel to emphasize local edges.
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        result = cv2.filter2D(rgb, ddepth=-1, kernel=kernel)
    else:
        raise ValueError(f"Unsupported processing option: {option}")
    return Image.fromarray(result, mode="RGB")


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a Pillow image as PNG bytes for download."""
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def image_metadata(image: Image.Image) -> dict[str, str]:
    """Return display-ready metadata for an image."""
    width, height = image.size
    aspect_ratio = width / height if height else 0
    return {
        "Width": f"{width:,} px",
        "Height": f"{height:,} px",
        "Resolution": f"{width:,} x {height:,} px",
        "Aspect ratio": f"{aspect_ratio:.2f}:1",
        "Image mode": image.mode,
    }


def image_analysis(image: Image.Image) -> dict[str, float | str]:
    """Calculate image statistics for the Analysis view."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    grayscale = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return {
        **image_metadata(image),
        "Average brightness": float(np.mean(grayscale)),
        "Contrast value": float(np.std(grayscale)),
        "Mean Red channel": float(np.mean(rgb[:, :, 0])),
        "Mean Green channel": float(np.mean(rgb[:, :, 1])),
        "Mean Blue channel": float(np.mean(rgb[:, :, 2])),
    }


def histogram_data(image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return RGB and grayscale histograms over the full 0-255 intensity range."""
    rgb = np.asarray(image.convert("RGB"))
    channels = tuple(
        np.histogram(rgb[:, :, index], bins=256, range=(0, 256))[0]
        for index in range(3)
    )
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_histogram = np.histogram(grayscale, bins=256, range=(0, 256))[0]
    return channels[0], channels[1], channels[2], gray_histogram


def visualize_color_space(image: Image.Image, color_space: str) -> Image.Image:
    """Create a display image for RGB, HSV, or grayscale visualization."""
    rgb = np.asarray(image.convert("RGB"))
    if color_space == "RGB":
        return image.convert("RGB").copy()
    if color_space == "HSV":
        # HSV separates hue, saturation, and value for color-space visualization.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    if color_space == "Grayscale":
        return Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)).convert("RGB")
    raise ValueError(f"Unsupported color space: {color_space}")


def ensure_displayable(image: Image.Image) -> np.ndarray:
    """Convert a Pillow image to an array suitable for Streamlit display."""
    return np.asarray(image)
