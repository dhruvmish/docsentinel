# docsentinel2/diff_engine_simple.py

from typing import List, Dict
import fitz  # PyMuPDF
from difflib import SequenceMatcher


class DiffEngineSimple:
    """
    Simple, fast text diff:
    - Works on PDF text blocks (no embeddings, no NLI)
    - Exact text comparison (option A)
    - Outputs ONE merged record per change:
        - TEXT_MODIFIED  -> old + new
        - TEXT_REMOVED   -> old only
        - TEXT_ADDED     -> new only
    - Includes page + bbox for side-by-side PDF overlay
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    # Internal: extract text segments with (text, page, bbox) from PDF   #
    # ------------------------------------------------------------------ #
    def _extract_segments(self, pdf_path: str) -> List[Dict]:
        """
        Returns list of dicts:
          { "text": str, "page": int, "bbox": (x0, y0, x1, y1) }
        using PyMuPDF text blocks.
        """
        doc = fitz.open(pdf_path)
        segments: List[Dict] = []

        for page_index, page in enumerate(doc):
            page_num = page_index + 1
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, ...)
            for block in blocks:
                x0, y0, x1, y1, text, *rest = block
                text = (text or "").strip()
                if not text:
                    continue

                segments.append(
                    {
                        "text": text,
                        "page": page_num,
                        "bbox": (x0, y0, x1, y1),
                    }
                )

        doc.close()
        return segments

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def detect_changes(self, old_path: str, new_path: str) -> List[Dict]:
        """
        Main entry point.

        Uses SequenceMatcher over lists of block texts to detect:
        - equal   -> ignored
        - delete  -> TEXT_REMOVED
        - insert  -> TEXT_ADDED
        - replace -> TEXT_MODIFIED / TEXT_REMOVED / TEXT_ADDED (paired)
        """
        old_segs = self._extract_segments(old_path)
        new_segs = self._extract_segments(new_path)

        old_texts = [s["text"] for s in old_segs]
        new_texts = [s["text"] for s in new_segs]

        matcher = SequenceMatcher(None, old_texts, new_texts)
        changes: List[Dict] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            # DELETE: blocks only in old
            if tag == "delete":
                for idx in range(i1, i2):
                    o = old_segs[idx]
                    changes.append(
                        {
                            "label": "TEXT_REMOVED",
                            "old": o["text"],
                            "new": "",
                            "page_old": o["page"],
                            "bbox_old": o["bbox"],
                            "page_new": None,
                            "bbox_new": None,
                        }
                    )

            # INSERT: blocks only in new
            elif tag == "insert":
                for idx in range(j1, j2):
                    n = new_segs[idx]
                    changes.append(
                        {
                            "label": "TEXT_ADDED",
                            "old": "",
                            "new": n["text"],
                            "page_old": None,
                            "bbox_old": None,
                            "page_new": n["page"],
                            "bbox_new": n["bbox"],
                        }
                    )

            # REPLACE: old[i1:i2] <-> new[j1:j2]
            elif tag == "replace":
                len_old = i2 - i1
                len_new = j2 - j1
                max_len = max(len_old, len_new)

                for k in range(max_len):
                    o = old_segs[i1 + k] if k < len_old else None
                    n = new_segs[j1 + k] if k < len_new else None

                    if o and n:
                        # Paired modification: ONE merged record
                        changes.append(
                            {
                                "label": "TEXT_MODIFIED",
                                "old": o["text"],
                                "new": n["text"],
                                "page_old": o["page"],
                                "bbox_old": o["bbox"],
                                "page_new": n["page"],
                                "bbox_new": n["bbox"],
                            }
                        )
                    elif o and not n:
                        # Extra old -> removed
                        changes.append(
                            {
                                "label": "TEXT_REMOVED",
                                "old": o["text"],
                                "new": "",
                                "page_old": o["page"],
                                "bbox_old": o["bbox"],
                                "page_new": None,
                                "bbox_new": None,
                            }
                        )
                    elif n and not o:
                        # Extra new -> added
                        changes.append(
                            {
                                "label": "TEXT_ADDED",
                                "old": "",
                                "new": n["text"],
                                "page_old": None,
                                "bbox_old": None,
                                "page_new": n["page"],
                                "bbox_new": n["bbox"],
                            }
                        )

        return changes
