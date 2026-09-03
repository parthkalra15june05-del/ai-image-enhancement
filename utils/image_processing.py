"""General image loading, adjustment, transformation, and export helpers."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError


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


def ensure_displayable(image: Image.Image) -> np.ndarray:
    """Convert a Pillow image to an array suitable for Streamlit display."""
    return np.asarray(image)
