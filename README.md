# ImageEnhance AI

**AI-Powered Image Enhancement & Editing Studio**

A university Computer Graphics and Multimedia project built with Python, Streamlit, Pillow, OpenCV, NumPy, and Matplotlib. The first version provides a lightweight image enhancement and editing workflow. AI model features such as background removal are reserved for a later stage.

## Features

- Upload PNG, JPG, and JPEG images
- Compare original and processed images
- View image dimensions, resolution, aspect ratio, and mode
- Adjust brightness, contrast, saturation, sharpness, and blur
- Apply grayscale, sepia, Gaussian blur, edge detection, and cartoon filters
- Rotate and flip images
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
