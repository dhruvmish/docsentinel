# docsentinel2/report.py

import csv
import os
from typing import List, Dict

from .summarizer import summarize_change


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _infer_modality(change: Dict) -> str:
    if "label" in change:
        return "TEXT"
    ctype = change.get("type", "")
    if ctype.startswith("IMAGE_") or ctype == "PAGE_LAYOUT_CHANGED":
        return "IMAGE"
    if "sheet" in change or "sheet_name" in change:
        return "TABLE"
    return "UNKNOWN"


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------------------------------------------
# ORIGINAL FUNCTION (UNCHANGED)
# ---------------------------------------------------------

def generate_report(results: List[Dict], output_path: str):
    """
    Backward-compatible single CSV report.
    """
    fieldnames = ["old", "new", "label", "cosine"]

    optional_fields = ["old_numbers", "new_numbers"]
    for field in optional_fields:
        if any(field in r for r in results):
            fieldnames.append(field)

    audit_fields = ["change_id", "modality", "page", "severity", "summary"]
    for f in audit_fields:
        if f not in fieldnames:
            fieldnames.append(f)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, r in enumerate(results, start=1):
            r = dict(r)
            r["change_id"] = f"CHG-{idx:04d}"
            r["modality"] = _infer_modality(r)
            r["summary"] = summarize_change(r)
            writer.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------------------------------------------
# ✅ NEW: AUDIT-GRADE SPLIT REPORTS
# ---------------------------------------------------------

def generate_audit_reports(results: List[Dict], output_dir: str):
    """
    Generates three separate CSVs:
    - text_changes.csv
    - image_changes.csv
    - table_changes.csv

    Industry-grade audit format.
    """

    os.makedirs(output_dir, exist_ok=True)

    text_rows = []
    image_rows = []
    table_rows = []

    for idx, r in enumerate(results, start=1):
        r = dict(r)
        r["change_id"] = f"CHG-{idx:04d}"
        r["modality"] = _infer_modality(r)
        r["summary"] = summarize_change(r)

        if r["modality"] == "TEXT":
            text_rows.append(r)
        elif r["modality"] == "IMAGE":
            image_rows.append(r)
        elif r["modality"] == "TABLE":
            table_rows.append(r)

    # ---------------- TEXT CSV ----------------
    if text_rows:
        text_fields = [
            "change_id",
            "page",
            "severity",
            "label",
            "old",
            "new",
            "cosine",
            "summary",
        ]
        _write_csv(
            os.path.join(output_dir, "text_changes.csv"),
            text_rows,
            text_fields,
        )

    # ---------------- IMAGE CSV ----------------
    if image_rows:
        image_fields = [
            "change_id",
            "page",
            "type",
            "severity",
            "regions",
            "highlight_path",
            "summary",
        ]
        _write_csv(
            os.path.join(output_dir, "image_changes.csv"),
            image_rows,
            image_fields,
        )

    # ---------------- TABLE CSV ----------------
    if table_rows:
        table_fields = [
            "change_id",
            "sheet",
            "type",
            "cell",
            "old",
            "new",
            "severity",
            "summary",
        ]
        _write_csv(
            os.path.join(output_dir, "table_changes.csv"),
            table_rows,
            table_fields,
        )
