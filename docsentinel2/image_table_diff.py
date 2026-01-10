# docsentinel2/image_table_diff.py

from typing import List, Dict, Tuple, Optional
import math

from .ingestion import CellRep, TableRep, WorkbookRep, load_pdf_layout
from .table_ocr import extract_table_from_image
from .embeddings import EmbeddingModel


def _table_to_dict(table: TableRep) -> Dict[Tuple[int, int], str]:
    return {(c.row, c.col): str(c.value) for c in table.cells}


def _is_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))  # allow 1,234.56
        return True
    except Exception:
        return False


def _to_number(s: str) -> float:
    return float(s.replace(",", ""))


def diff_image_tables(
    old_image_path: str,
    new_image_path: str,
    embedder: Optional[EmbeddingModel] = None,
    sheet_name: str = "image_table"
) -> List[Dict]:
    """
    Cell-by-cell diff between two table images.

    Returns a list of dicts:
      {
        "type": "CELL_TEXT_CHANGED" | "CELL_NUM_CHANGED" | "CELL_ADDED" | "CELL_REMOVED",
        "row": int,
        "col": int,
        "old": str or None,
        "new": str or None,
        "delta": float (for numeric),
        "percent_change": float (for numeric),
        "semantic_similarity": float or None
      }
    """
    if embedder is None:
        embedder = EmbeddingModel()

    old_table = extract_table_from_image(old_image_path, sheet_name=sheet_name + "_old")
    new_table = extract_table_from_image(new_image_path, sheet_name=sheet_name + "_new")

    old_map = _table_to_dict(old_table)
    new_map = _table_to_dict(new_table)

    all_keys = set(old_map.keys()) | set(new_map.keys())
    changes: List[Dict] = []

    for (r, c) in sorted(all_keys):
        old_val = old_map.get((r, c))
        new_val = new_map.get((r, c))

        if old_val is None and new_val is not None:
            changes.append({
                "type": "CELL_ADDED",
                "row": r,
                "col": c,
                "old": None,
                "new": new_val
            })
            continue

        if old_val is not None and new_val is None:
            changes.append({
                "type": "CELL_REMOVED",
                "row": r,
                "col": c,
                "old": old_val,
                "new": None
            })
            continue

        # both present
        if str(old_val).strip() == str(new_val).strip():
            continue  # unchanged cell

        # numeric vs text change
        if _is_number(old_val) and _is_number(new_val):
            old_num = _to_number(old_val)
            new_num = _to_number(new_val)
            delta = new_num - old_num
            pct = (delta / old_num * 100.0) if old_num != 0 else math.inf

            changes.append({
                "type": "CELL_NUM_CHANGED",
                "row": r,
                "col": c,
                "old": old_val,
                "new": new_val,
                "delta": delta,
                "percent_change": pct
            })
        else:
            sim = embedder.similarity(str(old_val), str(new_val))
            changes.append({
                "type": "CELL_TEXT_CHANGED",
                "row": r,
                "col": c,
                "old": old_val,
                "new": new_val,
                "semantic_similarity": float(sim)
            })

    return changes


# ---- Apply this to tables embedded as images in PDFs ----

def diff_tables_in_pdf_images(
    old_pdf_path: str,
    new_pdf_path: str,
    embedder: Optional[EmbeddingModel] = None,
    tmp_old_dir: str = "tmp_old_imgs",
    tmp_new_dir: str = "tmp_new_imgs"
) -> List[Dict]:
    """
    Heuristic: for every image in old/new PDFs that looks like a table,
    run diff_image_tables on the image pair.

    Returns list of changes with page / element identifiers.
    """
    if embedder is None:
        embedder = EmbeddingModel()

    old_doc = load_pdf_layout(old_pdf_path, tmp_old_dir)
    new_doc = load_pdf_layout(new_pdf_path, tmp_new_dir)

    old_imgs = {b.id: b for p in old_doc.pages for b in p.blocks if b.type == "image"}
    new_imgs = {b.id: b for p in new_doc.pages for b in p.blocks if b.type == "image"}

    all_ids = set(old_imgs.keys()) & set(new_imgs.keys())
    all_changes: List[Dict] = []

    for img_id in sorted(all_ids):
        old_blk = old_imgs[img_id]
        new_blk = new_imgs[img_id]

        # For PoC, we *try* table extraction on every image.
        # In future, we can add a "looks_like_table" heuristic using line detection.
        try:
            cell_changes = diff_image_tables(
                old_blk.path,
                new_blk.path,
                embedder=embedder,
                sheet_name=f"{img_id}"
            )
        except Exception:
            continue

        for ch in cell_changes:
            ch_with_meta = {
                "page": new_blk.page_number,
                "element_id": img_id,
                **ch
            }
            all_changes.append(ch_with_meta)

    return all_changes
