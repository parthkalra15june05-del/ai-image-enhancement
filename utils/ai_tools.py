"""Adaptive enhancement, quality recommendations, and AI image tools."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from utils.image_processing import apply_processing_option


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


@lru_cache(maxsize=8)
def _cached_subject_mask(image_bytes: bytes) -> Image.Image:
    """Run rembg once per image and cache its soft alpha mask."""
    try:
        from rembg import remove
        result = remove(image_bytes)
        return Image.open(BytesIO(result)).convert("RGBA").getchannel("A").copy()
    except (ImportError, SystemExit) as exc:
        raise RuntimeError("Portrait tools require rembg with its CPU or GPU backend.") from exc
    except (Exception, SystemExit) as exc:
        raise RuntimeError(f"Subject segmentation failed: {exc}") from exc


def get_subject_mask(image: Image.Image) -> Image.Image:
    """Return a cached, softened rembg foreground mask."""
    return _cached_subject_mask(_image_bytes(image)).filter(ImageFilter.GaussianBlur(radius=1.2))


def _blend_with_mask(foreground: Image.Image, background: Image.Image, mask: Image.Image) -> Image.Image:
    return Image.composite(foreground.convert("RGBA"), background.convert("RGBA"), mask).convert("RGBA")


def portrait_blur(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Keep the rembg subject sharp while strongly blurring the background."""
    mask = get_subject_mask(image)
    return _blend_with_mask(image, image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=14)), mask), mask


def subject_enhance(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Apply conservative enhancement to the subject only."""
    mask = get_subject_mask(image)
    enhanced = ImageEnhance.Sharpness(ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(1.08)).enhance(1.10)).enhance(1.25)
    return _blend_with_mask(enhanced, image, mask), mask


def subject_pop(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Brighten, sharpen, and saturate the subject against a softened background."""
    mask = get_subject_mask(image)
    foreground = ImageEnhance.Sharpness(ImageEnhance.Color(ImageEnhance.Brightness(image).enhance(1.08)).enhance(1.10)).enhance(1.25)
    background = ImageEnhance.Color(ImageEnhance.Brightness(image).enhance(0.92)).enhance(0.88).filter(ImageFilter.GaussianBlur(radius=2.5))
    return _blend_with_mask(foreground, background, mask), mask


def replace_background(image: Image.Image, replacement: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Composite the subject over a cover-cropped replacement background."""
    mask = get_subject_mask(image)
    background = ImageOps.fit(replacement.convert("RGB"), image.size, method=Image.Resampling.LANCZOS)
    return _blend_with_mask(image, background, mask), mask


def remove_background(image: Image.Image) -> Image.Image:
    """Remove the background with rembg and return a transparent RGBA image."""
    result = image.convert("RGBA").copy()
    result.putalpha(get_subject_mask(image))
    return result

@lru_cache(maxsize=1)
def _load_super_resolution_model():
    """Lazily load the actual pretrained Swin2SR neural network."""
    try:
        from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
    except ImportError as exc:
        raise RuntimeError("AI Super Resolution requires torch and transformers.") from exc
    model_id = "caidas/swin2SR-classical-sr-x2-64"
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = Swin2SRForImageSuperResolution.from_pretrained(model_id)
        model.eval()
        return processor, model
    except Exception as exc:
        raise RuntimeError(f"Could not load Swin2SR model: {exc}") from exc

def super_resolve(image: Image.Image) -> Image.Image:
    """Run pretrained Swin2SR x2 super resolution."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("AI Super Resolution requires torch.") from exc
    processor, model = _load_super_resolution_model()
    try:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).reconstruction
        output = output.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((output * 255).round().astype(np.uint8), mode="RGB")
    except Exception as exc:
        raise RuntimeError(f"Super-resolution inference failed: {exc}") from exc

@lru_cache(maxsize=1)
def _load_captioning_model():
    """Lazily load the pretrained BLIP image-captioning model."""
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except ImportError as exc:
        raise RuntimeError("AI Image Understanding requires torch and transformers.") from exc
    model_id = "Salesforce/blip-image-captioning-base"
    try:
        return BlipProcessor.from_pretrained(model_id), BlipForConditionalGeneration.from_pretrained(model_id)
    except Exception as exc:
        raise RuntimeError(f"Could not load BLIP model: {exc}") from exc

def generate_caption(image: Image.Image) -> str:
    """Generate a short scene description with pretrained BLIP."""
    try:
        import torch
        processor, model = _load_captioning_model()
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=32)
        return processor.decode(output[0], skip_special_tokens=True).strip()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"AI image understanding failed: {exc}") from exc

def classify_scene(caption: str) -> str:
    """Classify a BLIP caption into a broad editing category with transparent rules."""
    text = caption.lower()
    keywords = {
        "Portrait": ["person", "man", "woman", "child", "people", "face"],
        "Landscape": ["mountain", "landscape", "beach", "lake", "sky", "forest", "field"],
        "Indoor": ["room", "indoor", "kitchen", "office", "bedroom"],
        "Outdoor": ["outdoor", "street", "park", "building", "road"],
        "Document": ["document", "paper", "text", "book", "screen"],
        "Food": ["food", "dish", "meal", "pizza", "cake", "fruit"],
    }
    return max(keywords, key=lambda category: sum(word in text for word in keywords[category]), default="General") if any(word in text for words in keywords.values() for word in words) else "General"

def scene_recommendations(scene: str, image: Image.Image) -> list[str]:
    """Generate deterministic editing advice from scene category and image metrics."""
    metrics = _image_metrics(image)
    recommendations = {
        "Portrait": ["Enhance foreground subject", "Apply mild background blur", "Improve subject brightness"],
        "Landscape": ["Improve local contrast", "Enhance color moderately", "Preserve overall sharpness"],
        "Indoor": ["Lift shadows carefully", "Improve local contrast", "Preserve natural colors"],
        "Outdoor": ["Balance highlights and shadows", "Improve local contrast", "Enhance color moderately"],
        "Document": ["Improve contrast for readability", "Reduce noise", "Preserve edge sharpness"],
        "Food": ["Enhance color moderately", "Improve local contrast", "Preserve fine detail"],
        "General": ["Use mild contrast enhancement", "Check brightness before editing", "Preserve natural detail"],
    }[scene].copy()
    if metrics["brightness"] < 70:
        recommendations.append("Brightness is low; apply a mild brightness correction")
    if metrics["contrast"] < 38:
        recommendations.append("Low contrast detected; CLAHE may help")
    if metrics["sharpness"] < 90:
        recommendations.append("Image appears soft; use mild sharpening")
    return recommendations

