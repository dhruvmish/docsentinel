import fitz
import os
from typing import List, Dict, Tuple

OUTPUT_ANN = "outputs/annotated_pages"
os.makedirs(OUTPUT_ANN, exist_ok=True)


def _normalize_bboxes(change: Dict) -> List[Tuple[float, float, float, float]]:
    # Accept various formats: bbox, bboxes, regions (legacy)
    bboxes = []

    if "bbox" in change and change["bbox"]:
        bboxes.append(change["bbox"])

    if "bboxes" in change and isinstance(change["bboxes"], list):
        bboxes.extend(change["bboxes"])

    if "regions" in change and isinstance(change["regions"], list):
        # Convert (x,y,w,h) → (x0,y0,x1,y1)
        for (x, y, w, h) in change["regions"]:
            bboxes.append((x, y, x + w, y + h))

    # Filter out invalid ones
    return [b for b in bboxes if len(b) == 4]


def annotate_pdf_with_visual_changes(old_pdf: str, new_pdf: str, changes: List[Dict]):
    old_doc = fitz.open(old_pdf)
    new_doc = fitz.open(new_pdf)

    # Colors: Red = removed/changed, Green = added/changed
    for ch in changes:
        bboxes = _normalize_bboxes(ch)
        if not bboxes:
            continue

        page_num = ch.get("page")
        if not page_num or page_num < 1:
            continue

        idx = page_num - 1

        page_old = old_doc[idx]
        page_new = new_doc[idx]

        for bbox in bboxes:
            rect = fitz.Rect(*bbox)

            # annotate on both PDFs for comparison
            page_old.draw_rect(rect, color=(1, 0, 0), width=2)
            page_new.draw_rect(rect, color=(0, 1, 0), width=2)

    out_old = os.path.join(OUTPUT_ANN, "old_annotated.pdf")
    out_new = os.path.join(OUTPUT_ANN, "new_annotated.pdf")

    old_doc.save(out_old)
    new_doc.save(out_new)
    old_doc.close()
    new_doc.close()

    return out_old, out_new
