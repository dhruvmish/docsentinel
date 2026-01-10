import sys
import os
import io

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import torch

from siamese_unet import SiameseUNet


# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "siamese_unet_best.pth"
IMG_SIZE = 512          # higher resolution for tiny objects
THRESH = 0.5            # mask threshold
MIN_AREA = 50           # minimum change area (in mask coords)
ZOOM = 2.0              # PDF render zoom
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# PDF → BGR image (full page)
# ----------------------------
def pdf_page_to_image(doc, page_no, zoom=ZOOM):
    page = doc.load_page(page_no)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# ----------------------------
# Content-aware cropping (remove margins)
# returns: cropped_img, (x0, y0, w, h) in original coords
# ----------------------------
def extract_content_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # invert threshold to find non-white content
    _, th = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = img.shape[:2]
        return img, (0, 0, w, h)

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cropped = img[y:y + h, x:x + w]
    return cropped, (x, y, w, h)


# ----------------------------
# Preprocess for model
# ----------------------------
def preprocess(img):
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = (img_resized - 0.5) / 0.5  # normalize (mean=0.5,std=0.5)
    img_resized = img_resized.transpose(2, 0, 1)  # HWC -> CHW
    return torch.tensor(img_resized).unsqueeze(0)  # [1,3,H,W]


# ----------------------------
# Predict change mask for cropped region
# ----------------------------
@torch.no_grad()
def predict_change_mask(model, imgA_crop, imgB_crop):
    imgA_t = preprocess(imgA_crop).to(DEVICE)
    imgB_t = preprocess(imgB_crop).to(DEVICE)

    logits = model(imgA_t, imgB_t)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (probs > THRESH).astype(np.uint8) * 255

    # morphological closing to clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# ----------------------------
# Mask → bounding boxes in mask coords
# ----------------------------
def mask_to_bboxes(mask, min_area=MIN_AREA):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append((x, y, w, h))
    return bboxes


# ----------------------------
# Map mask bboxes → full page coords
# crop_box = (x0,y0,w,h) in full page
# ----------------------------
def map_bboxes_to_page(bboxes_mask, crop_box, crop_shape, mask_shape):
    x0, y0, cw, ch = crop_box
    mh, mw = mask_shape
    scale_x = cw / mw
    scale_y = ch / mh

    mapped = []
    for (x, y, w, h) in bboxes_mask:
        px = int(x * scale_x) + x0
        py = int(y * scale_y) + y0
        pw = int(w * scale_x)
        ph = int(h * scale_y)
        mapped.append((px, py, pw, ph))
    return mapped


# ----------------------------
# Convert BGR numpy image → PNG bytes
# ----------------------------
def image_to_png_bytes(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------
# Process a pair of PDFs (multi-page)
# Returns list of annotated page images
# ----------------------------
def process_pdfs(pdfA_path, pdfB_path, model):
    docA = fitz.open(pdfA_path)
    docB = fitz.open(pdfB_path)

    n_pages = min(len(docA), len(docB))
    print(f"Processing {n_pages} page(s)...")

    annotated_pages = []

    for page_no in range(n_pages):
        print(f"  Page {page_no+1}/{n_pages}...")

        pageA_img = pdf_page_to_image(docA, page_no)
        pageB_img = pdf_page_to_image(docB, page_no)

        # match size
        pageB_img = cv2.resize(pageB_img, (pageA_img.shape[1], pageA_img.shape[0]))

        # crop content regions (union of both)
        cropA, boxA = extract_content_region(pageA_img)
        cropB, boxB = extract_content_region(pageB_img)

        # union of both crop boxes (in case they differ slightly)
        x0 = min(boxA[0], boxB[0])
        y0 = min(boxA[1], boxB[1])
        x1 = max(boxA[0] + boxA[2], boxB[0] + boxB[2])
        y1 = max(boxA[1] + boxA[3], boxB[1] + boxB[3])

        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(pageA_img.shape[1], x1)
        y1 = min(pageA_img.shape[0], y1)

        cropA_union = pageA_img[y0:y1, x0:x1]
        cropB_union = pageB_img[y0:y1, x0:x1]
        crop_box_union = (x0, y0, x1 - x0, y1 - y0)

        # predict mask on the cropped region
        mask = predict_change_mask(model, cropA_union, cropB_union)

        # get bboxes in mask coords
        bboxes_mask = mask_to_bboxes(mask)
        # map to full-page coords
        full_bboxes = map_bboxes_to_page(
            bboxes_mask,
            crop_box_union,
            cropA_union.shape[:2],
            mask.shape[:2],
        )

        # draw on a copy of page A
        overlay = pageA_img.copy()
        for (x, y, w, h) in full_bboxes:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 4)

        annotated_pages.append(overlay)

    docA.close()
    docB.close()
    return annotated_pages


# ----------------------------
# Create annotated PDF from page images
# ----------------------------
def save_annotated_pdf(annotated_pages, output_path="annotated_output.pdf"):
    doc = fitz.open()
    for img in annotated_pages:
        img_bytes = image_to_png_bytes(img)
        h, w = img.shape[:2]
        page = doc.new_page(width=w, height=h)
        page.insert_image(page.rect, stream=img_bytes)
    doc.save(output_path)
    doc.close()
    print(f"Annotated PDF saved to: {output_path}")


# ----------------------------
# MAIN
# ----------------------------
def main():
    # load model
    model = SiameseUNet(in_ch=3, base_ch=32, out_ch=1).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.")

    # get PDF paths
    if len(sys.argv) >= 3:
        pdfA_path = sys.argv[1]
        pdfB_path = sys.argv[2]
    else:
        pdfA_path = input("Enter first PDF path: ").strip()
        pdfB_path = input("Enter second PDF path: ").strip()

    if not os.path.exists(pdfA_path):
        print(f"ERROR: {pdfA_path} not found")
        return
    if not os.path.exists(pdfB_path):
        print(f"ERROR: {pdfB_path} not found")
        return

    annotated_pages = process_pdfs(pdfA_path, pdfB_path, model)

    # save annotated pdf
    save_annotated_pdf(annotated_pages, "annotated_output.pdf")

    # also save first page as PNG preview
    if annotated_pages:
        cv2.imwrite("annotated_page1.png", annotated_pages[0])
        print("First annotated page saved as annotated_page1.png")


if __name__ == "__main__":
    main()
