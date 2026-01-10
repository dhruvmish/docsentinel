# docsentinel2/image_ocr.py

import pytesseract
from PIL import Image

def ocr_image(path: str) -> str:
    """
    Extract text from an image using Tesseract OCR.
    Returns cleaned text string.
    """
    try:
        img = Image.open(path).convert("RGB")
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception:
        return ""
