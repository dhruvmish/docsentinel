# docsentinel2/diff_engine_onnx.py

from typing import List, Dict, Optional

from docsentinel2.nli_model import NLIModel
from docsentinel2.text_features import diff_words, extract_numbers
from docsentinel2.severity import classify_change


def _ensure_nli_keys(p: Dict) -> Dict[str, float]:
    # Guarantee all 3 NLI keys exist
    return {
        "entail": float(p.get("entail", 0.0)),
        "neutral": float(p.get("neutral", 0.0)),
        "contradict": float(p.get("contradict", 0.0)),
    }


def _lookup_meta(idx: Optional[int], meta: Optional[List[Dict]]):
    """
    Safely get page + bbox given a sentence index and meta list.
    meta[i] is expected to be dict with keys: page, bbox (from ingestion).
    """
    if meta is None or idx is None:
        return None, None

    if idx < 0 or idx >= len(meta):
        return None, None

    info = meta[idx] or {}
    return info.get("page"), info.get("bbox")


class DiffEngineONNX:
    def __init__(self, embedder, aligner):
        self.embedder = embedder
        self.aligner = aligner
        self.nli = NLIModel(use_onnx=True)

    def detect_changes(
        self,
        old_sentences: List[str],
        new_sentences: List[str],
        old_meta: Optional[List[Dict]] = None,
        new_meta: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Detect semantic changes between two sentence lists.

        old_meta / new_meta are optional lists aligned with sentences:
          meta[i] = {"page": int, "bbox": (x0,y0,x1,y1), "block_id": str}

        If meta is not provided, page/bbox in the output will be None.
        """
        alignments = self.aligner(old_sentences, new_sentences)
        changes: List[Dict] = []

        for idx_old, idx_new in alignments:
            # Added sentence
            if idx_old is None:
                page, bbox = _lookup_meta(idx_new, new_meta)
                changes.append({
                    "old": "",
                    "new": new_sentences[idx_new],
                    "label": "ADDED_SENTENCE",
                    "cosine": 0.0,
                    "nli": None,
                    "page": page,
                    "bbox": bbox,
                })
                continue

            # Removed sentence
            if idx_new is None:
                page, bbox = _lookup_meta(idx_old, old_meta)
                changes.append({
                    "old": old_sentences[idx_old],
                    "new": "",
                    "label": "REMOVED_SENTENCE",
                    "cosine": 0.0,
                    "nli": None,
                    "page": page,
                    "bbox": bbox,
                })
                continue

            old = old_sentences[idx_old]
            new = new_sentences[idx_new]

            cosine = self.embedder.similarity(old, new)
            removed, added = diff_words(old, new)
            nums_changed = extract_numbers(old) != extract_numbers(new)

            # Always run NLI (ONNX powered)
            p1, p2 = self.nli.bidirectional(old, new)
            p1 = _ensure_nli_keys(p1)
            p2 = _ensure_nli_keys(p2)

            label = classify_change(
                cosine, p1, p2,
                removed, added,
                nums_changed
            )

            if label != "NO_CHANGE":
                page, bbox = _lookup_meta(idx_old, old_meta)
                # fallback to new_meta if old_meta missing
                if page is None and new_meta is not None:
                    page, bbox = _lookup_meta(idx_new, new_meta)

                changes.append({
                    "old": old,
                    "new": new,
                    "label": label,
                    "cosine": float(cosine),
                    "nli": (p1, p2),
                    "page": page,
                    "bbox": bbox,
                })

        return changes
