# ImageEnhance AI

**AI-Powered Image Enhancement & Editing Studio**

A university Computer Graphics and Multimedia project built with Python, Streamlit, Pillow, OpenCV, NumPy, Matplotlib, and rembg. It combines interactive image editing, adaptive enhancement, image analysis, and neural foreground segmentation.

## Features

- Upload PNG, JPG, and JPEG images
- Compare original and processed images
- View image dimensions, resolution, aspect ratio, and mode
- Adjust brightness, contrast, saturation, sharpness, and blur
- Apply grayscale, sepia, Gaussian blur, edge detection, and cartoon filters
- Rotate and flip images
- Apply adaptive restoration and quality recommendations
- Segment portraits, replace backgrounds, and remove backgrounds with rembg `u2netp`
- Download the processed image as PNG

## Setup

Use Python 3.10 or newer, then install the dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit in your browser.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. In Streamlit Community Cloud, create an app from the repository.
3. Set the entry-point file to `app.py` and select Python 3.10 or 3.11.
4. Deploy. Community Cloud installs the pinned packages from `requirements.txt` automatically.

No secrets are required. The lightweight `u2netp` weights are downloaded by rembg on the first portrait or background request, so that first request takes longer than later cached requests. Uploaded images are limited to 20 MB to protect the deployment's memory.

## Project Structure

```text
app.py
requirements.txt
README.md
utils/
  __init__.py
  image_processing.py
  filters.py
  ai_tools.py
assets/
```
