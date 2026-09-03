"""Classic image filters used by the first application version."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps


def apply_filter(image: Image.Image, filter_name: str) -> Image.Image:
    """Apply a named filter and return a Pillow image."""
    if filter_name == "Original":
        return image.copy()
    if filter_name == "Grayscale":
        return ImageOps.grayscale(image).convert("RGB")
    if filter_name == "Sepia":
        grayscale = np.asarray(ImageOps.grayscale(image), dtype=np.float32)
        sepia = np.stack(
            [grayscale * 1.08, grayscale * 0.95, grayscale * 0.78], axis=-1
        ).clip(0, 255).astype(np.uint8)
        return Image.fromarray(sepia, mode="RGB")
    if filter_name == "Gaussian Blur":
        return image.filter(ImageFilter.GaussianBlur(radius=4))
    if filter_name == "Edge Detection":
        rgb = np.asarray(image.convert("RGB"))
        edges = cv2.Canny(rgb, threshold1=80, threshold2=160)
        return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
    if filter_name == "Cartoon":
        return _cartoon_filter(image)
    raise ValueError(f"Unsupported filter: {filter_name}")


def _cartoon_filter(image: Image.Image) -> Image.Image:
    """Create a lightweight cartoon effect using smoothing and edge masking."""
    rgb = np.asarray(image.convert("RGB"))
    smoothed = cv2.bilateralFilter(rgb, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 7),
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=9,
        C=2,
    )
    cartoon = cv2.bitwise_and(smoothed, smoothed, mask=edges)
    return Image.fromarray(cartoon)
