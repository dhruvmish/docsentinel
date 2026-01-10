# docsentinel2/summarizer.py

from __future__ import annotations
from typing import Dict, List
from .text_features import diff_words, extract_numbers


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _format_number_change(old: str, new: str) -> str | None:
    old_nums = extract_numbers(old)
    new_nums = extract_numbers(new)

    if old_nums == new_nums or (not old_nums and not new_nums):
        return None

    parts = []
    for i, (o, n) in enumerate(zip(old_nums, new_nums)):
        if o != n:
            parts.append(f"value {i+1}: {o} → {n}")

    if len(new_nums) > len(old_nums):
        parts.append(f"{len(new_nums) - len(old_nums)} numeric value(s) added")
    elif len(old_nums) > len(new_nums):
        parts.append(f"{len(old_nums) - len(new_nums)} numeric value(s) removed")

    return "; ".join(parts) if parts else None


# ---------------------------------------------------------
# TEXT CHANGES
# ---------------------------------------------------------

def summarize_text_change(change: Dict) -> str:
    label = change.get("label", "CHANGE")
    old = (change.get("old") or "").strip()
    new = (change.get("new") or "").strip()

    removed, added = diff_words(old, new)
    num_summary = _format_number_change(old, new)

    pieces: List[str] = []

    # ---- Primary classification ----
    if label in {"TEXT_ADDED", "ADDED_SENTENCE"}:
        pieces.append(f"➕ New text added: “{new}”.")
    elif label in {"TEXT_REMOVED", "REMOVED_SENTENCE"}:
        pieces.append(f"➖ Text removed: “{old}”.")
    elif label in {"NUMERIC_CHANGE"}:
        pieces.append("🔢 Numeric content modified.")
    elif label in {"MAJOR_SEMANTIC_CHANGE"}:
        pieces.append("⚠️ Major semantic change detected.")
    else:
        pieces.append("✏️ Text modified.")

    # ---- Lexical changes ----
    if removed:
        pieces.append("Removed terms: " + ", ".join(sorted(set(removed))) + ".")
    if added:
        pieces.append("Added terms: " + ", ".join(sorted(set(added))) + ".")

    # ---- Numeric impact ----
    if num_summary:
        pieces.append("Numeric impact: " + num_summary + ".")

    # ---- Impact framing (industry-grade) ----
    impact_flags = []
    if num_summary:
        impact_flags.append("affects numeric values")
    if removed or added:
        impact_flags.append("alters terminology or meaning")

    if impact_flags:
        pieces.append("Impact: " + ", ".join(impact_flags) + ".")

    return " ".join(pieces)


# ---------------------------------------------------------
# VISUAL CHANGES
# ---------------------------------------------------------

def summarize_visual_change(change: Dict) -> str:
    ctype = change.get("type")
    page = change.get("page")
    prefix = f"📄 Page {page}: " if page else ""

    ssim_score = change.get("ssim_score")
    score_str = f"{ssim_score:.2f}" if isinstance(ssim_score, (float, int)) else "—"

    if ctype == "IMAGE_REPLACED":
        score = change.get("pixel_change_score")
        return (
            f"{prefix}🖼 Image replaced "
            f"(significant visual difference, pHash ≈ {score})."
        )

    if ctype == "IMAGE_TEXT_CHANGED":
        regions = change.get("regions", [])
        region_info = f"{len(regions)} region(s)" if regions else "localized region(s)"
        return (
            f"{prefix}✍️ Text embedded within image modified "
            f"({region_info}). This may affect interpretation of visual content."
        )

    if ctype == "IMAGE_CHANGE_REGION":
        regions = change.get("regions", [])
        region_info = f"{len(regions)} region(s)" if regions else "localized areas"
        return (
            f"{prefix}🖼 Visual content altered in {region_info}. "
            "Layout or visual semantics may be impacted."
        )

    if ctype == "IMAGE_ADDED":
        return f"{prefix}🆕 New image added."

    if ctype == "IMAGE_REMOVED":
        return f"{prefix}🗑 Image removed."

    if ctype == "IMAGE_SHIFTED":
        px = change.get("pixels_moved")
        return (
            f"{prefix}↔️ Image repositioned "
            f"(approx. {px} px shift)."
        )

    if ctype == "IMAGE_MINOR_TWEAK":
        score = change.get("pixel_change_score")
        return (
            f"{prefix}🖌 Minor visual adjustments detected "
            f"(pHash ≈ {score}). No major content change inferred."
        )

    if ctype == "PAGE_LAYOUT_CHANGED":
        return (
            f"{prefix}📄 Page layout modified "
            f"(structural similarity ≈ {score_str})."
        )

    return prefix + f"Visual change detected ({ctype})."


# ---------------------------------------------------------
# TABLE CHANGES
# ---------------------------------------------------------

def summarize_table_change(change: Dict) -> str:
    ctype = change.get("type", "")
    sheet = change.get("sheet") or change.get("sheet_name") or "Unknown sheet"

    if ctype == "CELL_UPDATED":
        addr = change.get("cell")
        old = change.get("old")
        new = change.get("new")
        return (
            f"🔢 {sheet}: Cell {addr} updated from “{old}” to “{new}”. "
            "This may affect calculations or downstream reporting."
        )

    if ctype == "ROW_ADDED":
        idx = change.get("row_index")
        return f"➕ {sheet}: Row {idx} added."

    if ctype == "ROW_REMOVED":
        idx = change.get("row_index")
        return f"➖ {sheet}: Row {idx} removed."

    if ctype == "SHEET_ADDED":
        return f"➕ New worksheet added: {sheet}."

    if ctype == "SHEET_REMOVED":
        return f"➖ Worksheet removed: {sheet}."

    return f"📊 Table change detected in {sheet}."


# ---------------------------------------------------------
# ROUTER (UNCHANGED SIGNATURE)
# ---------------------------------------------------------

def summarize_change(change: Dict) -> str:
    if "old" in change and "new" in change and "label" in change:
        return summarize_text_change(change)

    ctype = change.get("type", "")
    if ctype.startswith("IMAGE_") or ctype == "PAGE_LAYOUT_CHANGED":
        return summarize_visual_change(change)

    if "sheet" in change or "sheet_name" in change:
        return summarize_table_change(change)

    return "Change detected."
