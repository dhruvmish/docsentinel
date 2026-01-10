# docsentinel2/visual_diff.py

import os
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image
import fitz
import imagehash
from skimage.metrics import structural_similarity as ssim
import cv2

from .ingestion import load_pdf_layout
from .visual_siamese import siamese_similarity, diff_heatmap
from .image_ocr import ocr_image
from .embeddings import EmbeddingModel
from docsentinel2.image_change_heatmap import generate_heatmap_for_page



# =========================================================
# Thresholds
# =========================================================

PHASH_THRESHOLD_STRONG = 12
PHASH_THRESHOLD_WEAK = 3
SHIFT_THRESHOLD = 20
PAGE_SSIM_THRESHOLD = 0.92
TEXT_SIM_THRESHOLD = 0.95


# =========================================================
# Output paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "visual"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

embedder = EmbeddingModel()


# =========================================================
# Helpers
# =========================================================

def _full_image_bbox_payload(old_blk, new_blk):
    return {
        "page_old": old_blk.page_number,
        "bbox_old": list(old_blk.bbox),   # PAGE-level bbox
        "page_new": new_blk.page_number,
        "bbox_new": list(new_blk.bbox),
    }



def _phash(path):
    try:
        return imagehash.phash(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _ssim_paths(path1, path2, size=(500, 500)):
    try:
        img1 = np.array(Image.open(path1).convert("L").resize(size))
        img2 = np.array(Image.open(path2).convert("L").resize(size))
        return ssim(img1, img2)
    except Exception:
        return 1.0


def render_pdf_page(pdf_path, page_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    out_path = os.path.join(out_dir, f"page_{page_idx + 1}.png")
    pix.save(out_path)
    doc.close()
    return out_path


# =========================================================
# Main visual diff
# =========================================================

def run_visual_diff(
    old_path: str,
    new_path: str,
    tmp_old_dir="tmp_old",
    tmp_new_dir="tmp_new"
) -> List[Dict]:

    changes: List[Dict] = []

    old_doc = load_pdf_layout(old_path, os.path.join(tmp_old_dir, "imgs"))
    new_doc = load_pdf_layout(new_path, os.path.join(tmp_new_dir, "imgs"))

    old_imgs = {b.id: b for p in old_doc.pages for b in p.blocks if b.type == "image"}
    new_imgs = {b.id: b for p in new_doc.pages for b in p.blocks if b.type == "image"}

    # -----------------------------------------------------
    # 1️⃣ Image-by-image comparison
    # -----------------------------------------------------
    for img_id, old_blk in old_imgs.items():
        boxes = []  # ✅ ALWAYS defined
        highlight_path = None

        if img_id not in new_imgs:
            changes.append({
                "type": "IMAGE_REMOVED",
                "page": old_blk.page_number,
                "element_id": img_id
            })
            continue

        new_blk = new_imgs[img_id]

        # ---------- Movement ----------
        shift = abs(old_blk.bbox[0] - new_blk.bbox[0]) + abs(old_blk.bbox[1] - new_blk.bbox[1])
        if shift > SHIFT_THRESHOLD:
            changes.append({
                "type": "IMAGE_SHIFTED",
                "page": new_blk.page_number,
                "element_id": img_id,
                "pixels_moved": float(shift)


            })

        # ---------- Strong replacement ----------
        h_old = _phash(old_blk.path)
        h_new = _phash(new_blk.path)
        phash_diff = (h_old - h_new) if h_old and h_new else 0

        if phash_diff >= PHASH_THRESHOLD_STRONG:
            changes.append({
                "type": "IMAGE_REPLACED",
                "page": new_blk.page_number,
                "element_id": img_id,
                "pixel_change_score": float(phash_diff),
                **_full_image_bbox_payload(old_blk, new_blk)
            })

            continue

        # ---------- OCR-based text change ----------
        old_text = ocr_image(old_blk.path)
        new_text = ocr_image(new_blk.path)

        # --- STRICT TEXT DIFFERENCE CHECK (NO SEMANTICS) ---
        if old_text.strip() != new_text.strip():
            img2, mask, boxes = diff_heatmap(old_blk.path, new_blk.path)

            highlight_path = None
            if boxes:
                for (x, y, w, h) in boxes:
                    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 🔴 RED BOX

                highlight_path = OUTPUT_DIR / f"{img_id}_text_diff.png"
                cv2.imwrite(str(highlight_path), img2)

            changes.append({
                "type": "IMAGE_TEXT_CHANGED",
                "page": new_blk.page_number,
                "element_id": img_id,
                "ocr_old_text": old_text,
                "ocr_new_text": new_text,
                "highlight_path": str(highlight_path) if highlight_path else None,
                "regions": boxes if boxes else [],
                **_full_image_bbox_payload(old_blk, new_blk)
            })

            continue

        # ---------- Siamese CNN fallback ----------
        siam_sim = siamese_similarity(old_blk.path, new_blk.path)
        if siam_sim is not None and siam_sim < 0.98:
            heatmap_path = generate_heatmap_for_page(
                old_path,
                new_path,
                new_blk.page_number
            )

            changes.append({
                "type": "IMAGE_CHANGE_REGION",
                "page": new_blk.page_number,
                "element_id": img_id,
                "highlight_path": heatmap_path,
                "regions": boxes if boxes else [],
                **_full_image_bbox_payload(old_blk, new_blk)
            })

            continue

        # ---------- Minor tweak ----------
        if phash_diff > PHASH_THRESHOLD_WEAK:
            changes.append({
                "type": "IMAGE_MINOR_TWEAK",
                "page": new_blk.page_number,
                "element_id": img_id,
                "pixel_change_score": float(phash_diff)

            })

    # -----------------------------------------------------
    # 2️⃣ Newly added images
    # -----------------------------------------------------
    for img_id, blk in new_imgs.items():
        if img_id not in old_imgs:
            changes.append({
                "type": "IMAGE_ADDED",
                "page": blk.page_number,
                "element_id": img_id

            })

    # -----------------------------------------------------
    # 3️⃣ Page-level layout change
    # -----------------------------------------------------
    for p in range(min(len(old_doc.pages), len(new_doc.pages))):
        o = render_pdf_page(old_path, p, tmp_old_dir)
        n = render_pdf_page(new_path, p, tmp_new_dir)
        score = _ssim_paths(o, n)

        if score < PAGE_SSIM_THRESHOLD:
            changes.append({
                "type": "PAGE_LAYOUT_CHANGED",
                "page": p + 1,
                "ssim_score": float(score)
            })

    return changes
