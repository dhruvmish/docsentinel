# docsentinel2/table_ocr.py

from typing import List, Tuple
import cv2
import numpy as np
import pytesseract

from .ingestion import CellRep, TableRep


def _preprocess(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # adaptive threshold to handle scan brightness variations
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )
    return img, th


def _detect_cell_boxes(th_bin) -> List[Tuple[int, int, int, int]]:
    """
    Heuristic: find rectangular cell-like contours.
    Returns list of (x, y, w, h).
    """
    contours, _ = cv2.findContours(
        th_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    h_img, w_img = th_bin.shape[:2]
    min_w = w_img * 0.03
    min_h = h_img * 0.03

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h:
            continue
        # ignore huge boxes (likely whole table / background)
        if w > 0.95 * w_img or h > 0.95 * h_img:
            continue
        boxes.append((x, y, w, h))

    # sort top→bottom, left→right
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def _group_boxes_into_grid(boxes: List[Tuple[int, int, int, int]],
                           y_tol: int = 10) -> List[List[Tuple[int, int, int, int]]]:
    """
    Cluster boxes into rows based on y coordinate, then sort each row by x.
    """
    rows: List[List[Tuple[int, int, int, int]]] = []

    for box in boxes:
        x, y, w, h = box
        placed = False
        for row in rows:
            # compare with first box in the row
            _, ry, _, rh = row[0]
            if abs(y - ry) < y_tol or abs((y + h / 2) - (ry + rh / 2)) < y_tol:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    # sort rows by y, and boxes in each row by x
    rows.sort(key=lambda r: r[0][1])
    for r in rows:
        r.sort(key=lambda b: b[0])

    return rows


def _ocr_cell(image_gray, box) -> str:
    x, y, w, h = box
    pad = 2
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, image_gray.shape[1])
    y1 = min(y + h + pad, image_gray.shape[0])

    crop = image_gray[y0:y1, x0:x1]
    if crop.size == 0:
        return ""

    # binarize crop for better OCR
    _, crop_th = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(
        crop_th,
        config="--psm 7"  # single text line / cell
    )
    # clean up
    return " ".join(text.replace("\n", " ").split())


def extract_table_from_image(image_path: str,
                             sheet_name: str = "image_table") -> TableRep:
    """
    Extract a logical table (rows/cols + CellRep list) from a table image.
    Returns a TableRep that is compatible with XLSX table diff.
    """
    img_gray, th = _preprocess(image_path)
    boxes = _detect_cell_boxes(th)
    if not boxes:
        # fallback – return empty table
        return TableRep(sheet_name=sheet_name, cells=[], max_row=0, max_col=0)

    grid = _group_boxes_into_grid(boxes)

    cells: List[CellRep] = []
    max_row = len(grid)
    max_col = max(len(r) for r in grid)

    for r_idx, row in enumerate(grid, start=1):
        for c_idx, box in enumerate(row, start=1):
            value = _ocr_cell(img_gray, box)
            if value.strip() == "":
                continue
            cells.append(CellRep(row=r_idx, col=c_idx, value=value))

    return TableRep(
        sheet_name=sheet_name,
        cells=cells,
        max_row=max_row,
        max_col=max_col
    )


def extract_table_matrix(image_path: str) -> List[List[str]]:
    """
    Convenience: return a 2D list of strings for quick inspection.
    """
    table = extract_table_from_image(image_path)
    matrix = [
        ["" for _ in range(table.max_col)]
        for _ in range(table.max_row)
    ]
    for cell in table.cells:
        matrix[cell.row - 1][cell.col - 1] = str(cell.value)
    return matrix
