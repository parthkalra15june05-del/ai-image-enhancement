"""Adaptive enhancement, quality recommendations, and AI image tools."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from utils.image_processing import apply_processing_option, resize_for_preview


def _image_metrics(image: Image.Image) -> dict[str, float]:
    """Measure brightness, contrast, dynamic range, sharpness, and resolution."""
    rgb = np.asarray(image.convert("RGB"))
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "brightness": float(np.mean(grayscale)),
        "contrast": float(np.std(grayscale)),
        "dynamic_range": float(np.percentile(grayscale, 99) - np.percentile(grayscale, 1)),
        "sharpness": float(cv2.Laplacian(grayscale, cv2.CV_64F).var()),
        "resolution": float(image.width * image.height),
    }


def smart_auto_enhance(image: Image.Image) -> tuple[Image.Image, list[str]]:
    """Apply conservative, rule-based corrections based on measurable properties."""
    metrics = _image_metrics(image)
    enhanced = image.convert("RGB").copy()
    applied: list[str] = []

    if metrics["brightness"] < 70:
        # Mild gamma correction lifts dark midtones without aggressively clipping highlights.
        source = np.asarray(enhanced)
        gamma = 0.85
        lookup = np.array([((value / 255.0) ** gamma) * 255 for value in range(256)])
        enhanced = Image.fromarray(cv2.LUT(source, lookup.astype(np.uint8)))
        applied.append("Brightness correction")

    if metrics["contrast"] < 38 or metrics["dynamic_range"] < 150:
        enhanced = apply_processing_option(enhanced, "CLAHE Enhancement")
        applied.append("CLAHE contrast enhancement")

    if metrics["sharpness"] < 90:
        sharpened = apply_processing_option(enhanced, "Convolution Sharpen")
        enhanced = Image.blend(enhanced, sharpened, 0.35)
        applied.append("Mild sharpening")

    if metrics["sharpness"] < 25 and metrics["contrast"] > 25:
        enhanced = apply_processing_option(enhanced, "Gaussian Denoising")
        applied.append("Mild denoising")

    if not applied:
        applied.append("No correction needed")
    return enhanced, applied


def _quality_label(metrics: dict[str, float]) -> dict[str, str]:
    """Convert numeric restore measurements into readable assessment labels."""
    return {
        "Brightness": "Low" if metrics["brightness"] < 70 else "High" if metrics["brightness"] > 210 else "Good",
        "Contrast": "Low" if metrics["contrast"] < 38 else "High" if metrics["contrast"] > 85 else "Good",
        "Sharpness": "Low" if metrics["sharpness"] < 90 else "Moderate" if metrics["sharpness"] < 300 else "Good",
        "Resolution": "Low" if metrics["resolution"] < 1_000_000 else "Good",
    }


def image_quality_score(image: Image.Image) -> float:
    """Return a transparent 0-100 heuristic, not a scientific or neural score."""
    metrics = _image_metrics(image)
    brightness_score = max(0.0, 1.0 - abs(metrics["brightness"] - 128) / 128)
    contrast_score = min(metrics["contrast"] / 64, 1.0)
    range_score = min(metrics["dynamic_range"] / 220, 1.0)
    sharpness_score = min(metrics["sharpness"] / 300, 1.0)
    return round(100 * (0.25 * brightness_score + 0.25 * contrast_score + 0.20 * range_score + 0.30 * sharpness_score), 1)


def ai_restore(image: Image.Image, allow_super_resolution: bool = False) -> tuple[Image.Image, dict[str, float], dict[str, float], list[str], list[str]]:
    """Adaptively restore an image and return before/after metrics and an audit report."""
    original_metrics = _image_metrics(image)
    restored = image.convert("RGB").copy()
    applied: list[str] = []
    skipped: list[str] = []

    if original_metrics["brightness"] < 70:
        source = np.asarray(restored)
        lookup = np.array([((value / 255.0) ** 0.88) * 255 for value in range(256)])
        restored = Image.fromarray(cv2.LUT(source, lookup.astype(np.uint8)))
        applied.append("Gamma correction for low brightness")
    else:
        skipped.append("Gamma correction - brightness is balanced")

    if original_metrics["contrast"] < 38 or original_metrics["dynamic_range"] < 150:
        restored = apply_processing_option(restored, "CLAHE Enhancement")
        applied.append("CLAHE contrast enhancement")
    else:
        skipped.append("CLAHE - contrast and dynamic range are sufficient")

    if original_metrics["sharpness"] < 90:
        sharpened = apply_processing_option(restored, "Convolution Sharpen")
        restored = Image.blend(restored, sharpened, 0.30)
        applied.append("Mild convolution sharpening")
    else:
        skipped.append("Sharpening - edge detail is already sufficient")

    if original_metrics["sharpness"] < 25 and original_metrics["contrast"] > 25:
        restored = apply_processing_option(restored, "Gaussian Denoising")
        applied.append("Mild Gaussian denoising")
    else:
        skipped.append("Denoising - no strong noise indicator")

    if allow_super_resolution:
        skipped.append("Super Resolution - disabled in lightweight demo")
    else:
        skipped.append("Super Resolution - disabled in lightweight demo")

    return restored, original_metrics, _image_metrics(restored), applied, skipped


def quality_assessment(image: Image.Image) -> dict[str, object]:
    """Return understandable quality labels and recommendations without external APIs."""
    metrics = _image_metrics(image)
    brightness = metrics["brightness"]
    contrast = metrics["contrast"]
    sharpness = metrics["sharpness"]
    recommendations: list[str] = []

    if brightness < 70:
        brightness_label = "Low"
        recommendations.append("Image appears underexposed. Brightness enhancement is recommended.")
    elif brightness > 210:
        brightness_label = "High"
        recommendations.append("Image may be overexposed. Avoid increasing brightness.")
    else:
        brightness_label = "Good"

    if contrast < 38:
        contrast_label = "Low"
        recommendations.append("Low contrast detected. CLAHE enhancement is recommended.")
    elif contrast > 85:
        contrast_label = "High"
    else:
        contrast_label = "Good"

    if sharpness < 90:
        sharpness_label = "Low"
        recommendations.append("Image appears soft. Mild sharpening may improve edge detail.")
    elif sharpness < 300:
        sharpness_label = "Moderate"
    else:
        sharpness_label = "Good"

    resolution_label = "Good" if metrics["resolution"] >= 1_000_000 else "Low"
    if resolution_label == "Low":
        recommendations.append("Resolution is limited. Use a larger source image when possible.")

    needs_enhancement = bool(recommendations)
    if not needs_enhancement:
        recommendations.append("Image quality appears balanced. Only minor enhancement may be needed.")

    return {
        "metrics": metrics,
        "labels": {
            "Brightness": brightness_label,
            "Contrast": contrast_label,
            "Sharpness": sharpness_label,
            "Resolution": resolution_label,
        },
        "recommendations": recommendations,
        "summary": "Needs Enhancement" if needs_enhancement else "Good",
    }

def _image_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


@lru_cache(maxsize=1)
def _cached_rembg_session():
    """Load the lightweight rembg model only when segmentation is requested."""
    try:
        from rembg import new_session
        return new_session("u2netp")
    except (ImportError, SystemExit) as exc:
        raise RuntimeError("Portrait tools require rembg with its CPU backend.") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not load the rembg u2netp model: {exc}") from exc


@lru_cache(maxsize=8)
def _cached_subject_mask(image_bytes: bytes) -> Image.Image:
    """Run rembg once per image and cache its soft alpha mask."""
    try:
        from rembg import remove
        source = Image.open(BytesIO(image_bytes)).convert("RGB")
        preview = resize_for_preview(source, max_dimension=768)
        result = remove(_image_bytes(preview), session=_cached_rembg_session())
        mask = Image.open(BytesIO(result)).convert("RGBA").getchannel("A")
        return mask.resize(source.size, Image.Resampling.BILINEAR).copy()
    except (ImportError, SystemExit) as exc:
        raise RuntimeError("Portrait tools require rembg with its CPU backend.") from exc
    except (Exception, SystemExit) as exc:
        raise RuntimeError(f"Subject segmentation failed: {exc}") from exc


def get_subject_mask(image: Image.Image) -> Image.Image:
    """Return a cached, softened rembg foreground mask."""
    return _cached_subject_mask(_image_bytes(image)).filter(ImageFilter.GaussianBlur(radius=1.2))


def _blend_with_mask(foreground: Image.Image, background: Image.Image, mask: Image.Image) -> Image.Image:
    return Image.composite(foreground.convert("RGBA"), background.convert("RGBA"), mask).convert("RGBA")


def portrait_blur(image: Image.Image, mask: Image.Image | None = None) -> tuple[Image.Image, Image.Image]:
    """Keep the rembg subject sharp while strongly blurring the background."""
    mask = mask if mask is not None else get_subject_mask(image)
    return _blend_with_mask(image, image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=14)), mask), mask


def subject_enhance(image: Image.Image, mask: Image.Image | None = None) -> tuple[Image.Image, Image.Image]:
    """Apply conservative enhancement to the subject only."""
    mask = mask if mask is not None else get_subject_mask(image)
    enhanced = ImageEnhance.Sharpness(ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(1.08)).enhance(1.10)).enhance(1.25)
    return _blend_with_mask(enhanced, image, mask), mask


def subject_pop(image: Image.Image, mask: Image.Image | None = None) -> tuple[Image.Image, Image.Image]:
    """Brighten, sharpen, and saturate the subject against a softened background."""
    mask = mask if mask is not None else get_subject_mask(image)
    foreground = ImageEnhance.Sharpness(ImageEnhance.Color(ImageEnhance.Brightness(image).enhance(1.08)).enhance(1.10)).enhance(1.25)
    background = ImageEnhance.Color(ImageEnhance.Brightness(image).enhance(0.92)).enhance(0.88).filter(ImageFilter.GaussianBlur(radius=2.5))
    return _blend_with_mask(foreground, background, mask), mask


def replace_background(image: Image.Image, replacement: Image.Image, mask: Image.Image | None = None) -> tuple[Image.Image, Image.Image]:
    """Composite the subject over a cover-cropped replacement background."""
    mask = mask if mask is not None else get_subject_mask(image)
    background = ImageOps.fit(replacement.convert("RGB"), image.size, method=Image.Resampling.LANCZOS)
    return _blend_with_mask(image, background, mask), mask


def remove_background(image: Image.Image, mask: Image.Image | None = None) -> Image.Image:
    """Remove the background with rembg and return a transparent RGBA image."""
    result = image.convert("RGBA").copy()
    result.putalpha(mask if mask is not None else get_subject_mask(image))
    return result


