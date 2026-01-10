# main_unified.py — Unified Runner (PDF + XLSX, ONNX-Accelerated)

import os
from docsentinel2.ingestion import load_document, load_xlsx
from docsentinel2.embeddings import EmbeddingModel
from docsentinel2.alignment import SemanticAligner
from docsentinel2.diff_engine_onnx import DiffEngineONNX
from docsentinel2.table_diff import diff_workbooks
from docsentinel2.visual_diff import run_visual_diff
from docsentinel2.report import generate_report


def run_semantic_pdf_diff(old_path, new_path):
    print("\n▶ Running Semantic Text Diff (ONNX-Accelerated)...\n")

    old_sentences = load_document(old_path)
    new_sentences = load_document(new_path)

    print(f"Old sentences: {len(old_sentences)} | New sentences: {len(new_sentences)}")

    embed_model = EmbeddingModel()
    aligner = SemanticAligner(embed_model)

    # Pass paths so engine can attach page + bbox
    engine = DiffEngineONNX(
        embedder=embed_model,
        aligner=aligner,
        old_path=old_path,
        new_path=new_path,
    )

    return engine.detect_changes(old_sentences, new_sentences)


def run_visual_pdf_diff(old_path, new_path):
    print("\n▶ Running Visual Diff...\n")
    return run_visual_diff(old_path, new_path)


def run_excel_diff(old_path, new_path):
    print("\n▶ Running Spreadsheet Table Diff...\n")

    embedder = EmbeddingModel()
    old_wb = load_xlsx(old_path)
    new_wb = load_xlsx(new_path)

    return diff_workbooks(old_wb, new_wb, embedder)


def main():
    # Change these inputs as needed
    old_path = "data/Untitled document (21).pdf"
    new_path = "data/Untitled document (22).pdf"

    print("=== DocSentinel Unified Audit ===")
    print(f"OLD: {old_path}")
    print(f"NEW: {new_path}\n")

    ext = os.path.splitext(old_path)[1].lower()
    all_changes = []

    if ext == ".pdf":
        text_changes = run_semantic_pdf_diff(old_path, new_path)
        visual_changes = run_visual_pdf_diff(old_path, new_path)
        all_changes.extend(text_changes + visual_changes)

    elif ext == ".xlsx":
        excel_changes = run_excel_diff(old_path, new_path)
        all_changes.extend(excel_changes)

    else:
        raise ValueError("Unsupported file format (Only PDF/XLSX).")

    # Filter important changes
    real_changes = [
        c for c in all_changes
        if c.get("label") not in ["NO_CHANGE", "MINOR_CHANGE"]
    ]

    if not real_changes:
        print("\n🎉 No meaningful changes detected!")
        print("✔ Documents match semantically and visually.\n")
        return

    print(f"\nDetected {len(real_changes)} meaningful changes:\n")

    for i, ch in enumerate(real_changes, 1):
        print(f"--- Change #{i} ---")
        for k, v in ch.items():
            print(f"{k}: {v}")
        print()

    os.makedirs("outputs", exist_ok=True)
    generate_report(real_changes, "outputs/unified_report.csv")
    print("\n📄 Report saved → outputs/unified_report.csv\n")
    print("✔ Audit Completed Successfully\n")


if __name__ == "__main__":
    main()
