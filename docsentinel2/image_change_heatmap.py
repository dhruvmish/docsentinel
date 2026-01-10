import os
import cv2
import fitz
import numpy as np
import torch
from PIL import Image

from siamese_unet import SiameseUNet

# ------------------------------------------------------------------
# CONFIG (same as your spotchange.py)
# ------------------------------------------------------------------
MODEL_PATH = "siamese_unet_best.pth"
IMG_SIZE = 512
THRESH = 0.5
MIN_AREA = 50
ZOOM = 2.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "outputs/visual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load model ONCE (important for performance)
# ------------------------------------------------------------------
_model = None


def _load_model():
    global _model
    if _model is None:
        model = SiameseUNet(in_ch=3, base_ch=32, out_ch=1).to(DEVICE)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        _model = model
    return _model


# ------------------------------------------------------------------
# Helpers copied (safely) from spotchange.py
# ------------------------------------------------------------------
def pdf_page_to_image(doc, page_no, zoom=ZOOM):
    page = doc.load_page(page_no)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def extract_content_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        h, w = img.shape[:2]
        return img, (0, 0, w, h)

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return img[y:y + h, x:x + w], (x, y, w, h)


def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = img.transpose(2, 0, 1)
    return torch.tensor(img).unsqueeze(0)


@torch.no_grad()
def predict_change_mask(model, imgA, imgB):
    tA = preprocess(imgA).to(DEVICE)
    tB = preprocess(imgB).to(DEVICE)
    logits = model(tA, tB)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (probs > THRESH).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# ------------------------------------------------------------------
# PUBLIC API — this is what visual_diff will call
# ------------------------------------------------------------------
def generate_heatmap_for_page(old_pdf, new_pdf, page_number):
    """
    Returns path to heatmap PNG for IMAGE_CHANGE_REGION
    page_number is 1-based
    """
    model = _load_model()

    docA = fitz.open(old_pdf)
    docB = fitz.open(new_pdf)

    page_idx = page_number - 1

    imgA = pdf_page_to_image(docA, page_idx)
    imgB = pdf_page_to_image(docB, page_idx)

    imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))

    cropA, boxA = extract_content_region(imgA)
    cropB, boxB = extract_content_region(imgB)

    x0 = min(boxA[0], boxB[0])
    y0 = min(boxA[1], boxB[1])
    x1 = max(boxA[0] + boxA[2], boxB[0] + boxB[2])
    y1 = max(boxA[1] + boxA[3], boxB[1] + boxB[3])

    cropA = imgA[y0:y1, x0:x1]
    cropB = imgB[y0:y1, x0:x1]

    mask = predict_change_mask(model, cropA, cropB)

    overlay = imgB.copy()
    # resize mask back to crop size
    mask_resized = cv2.resize(
        mask,
        (cropB.shape[1], cropB.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    overlay_crop = overlay[y0:y1, x0:x1]
    overlay_crop[mask_resized > 0] = [0, 0, 255]

    out_path = os.path.join(
        OUTPUT_DIR,
        f"heatmap_page_{page_number}.png"
    )

    cv2.imwrite(out_path, overlay)

    docA.close()
    docB.close()

    return out_path
