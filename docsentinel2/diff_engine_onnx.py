from typing import Dict, Tuple, List, Optional

from docsentinel2.nli_model import NLIModel
from docsentinel2.text_features import diff_words, extract_numbers
from docsentinel2.severity import classify_change
from docsentinel2.ingestion import load_pdf_sentences_with_positions


def _ensure_nli_keys(p: Dict) -> Dict[str, float]:
    # Guarantee all 3 NLI keys exist
    return {
        "entail": float(p.get("entail", 0.0)),
        "neutral": float(p.get("neutral", 0.0)),
        "contradict": float(p.get("contradict", 0.0)),
    }


class DiffEngineONNX:
    """
    Same API as your old DiffEngine, but:
    - Uses ONNX-accelerated NLI
    - Can enrich text changes with page + bbox (for PDF)
    """

    def __init__(self, embedder, aligner,
                 old_path: Optional[str] = None,
                 new_path: Optional[str] = None):
        self.embedder = embedder
        self.aligner = aligner
        self.nli = NLIModel(use_onnx=True)

        self.old_meta: Optional[List[dict]] = None
        self.new_meta: Optional[List[dict]] = None

        # If paths are provided and PDF-based, pre-load metadata
        if old_path is not None:
            try:
                self.old_meta = load_pdf_sentences_with_positions(old_path)
            except Exception:
                self.old_meta = None

        if new_path is not None:
            try:
                self.new_meta = load_pdf_sentences_with_positions(new_path)
            except Exception:
                self.new_meta = None

    def _get_meta(self, idx: Optional[int], which: str):
        """
        Helper: safely fetch page/bbox metadata by sentence index.
        `which` is "old" or "new".
        """
        meta_list = self.old_meta if which == "old" else self.new_meta
        if meta_list is None or idx is None:
            return None, None
        if idx < 0 or idx >= len(meta_list):
            return None, None
        entry = meta_list[idx]
        return entry.get("page"), entry.get("bbox")

    def detect_changes(self, old_sentences, new_sentences):
        alignments = self.aligner(old_sentences, new_sentences)
        changes = []

        for idx_old, idx_new in alignments:
            # Added sentence (only in new)
            if idx_old is None:
                page_new, bbox_new = self._get_meta(idx_new, "new")
                changes.append({
                    "old": "",
                    "new": new_sentences[idx_new],
                    "label": "ADDED_SENTENCE",
                    "cosine": 0.0,
                    "nli": None,
                    "page_old": None,
                    "bbox_old": None,
                    "page_new": page_new,
                    "bbox_new": bbox_new,
                })
                continue

            # Removed sentence (only in old)
            if idx_new is None:
                page_old, bbox_old = self._get_meta(idx_old, "old")
                changes.append({
                    "old": old_sentences[idx_old],
                    "new": "",
                    "label": "REMOVED_SENTENCE",
                    "cosine": 0.0,
                    "nli": None,
                    "page_old": page_old,
                    "bbox_old": bbox_old,
                    "page_new": None,
                    "bbox_new": None,
                })
                continue

            # Paired sentences
            old = old_sentences[idx_old]
            new = new_sentences[idx_new]

            cosine = self.embedder.similarity(old, new)
            removed, added = diff_words(old, new)
            nums_changed = extract_numbers(old) != extract_numbers(new)

            # Always run NLI (ONNX-powered)
            p1, p2 = self.nli.bidirectional(old, new)
            p1 = _ensure_nli_keys(p1)
            p2 = _ensure_nli_keys(p2)

            label = classify_change(
                cosine, p1, p2,
                removed, added,
                nums_changed
            )

            if label != "NO_CHANGE":
                page_old, bbox_old = self._get_meta(idx_old, "old")
                page_new, bbox_new = self._get_meta(idx_new, "new")

                changes.append({
                    "old": old,
                    "new": new,
                    "label": label,
                    "cosine": float(cosine),
                    "nli": (p1, p2),
                    "page_old": page_old,
                    "bbox_old": bbox_old,
                    "page_new": page_new,
                    "bbox_new": bbox_new,
                })

        return changes
