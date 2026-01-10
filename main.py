# main.py

import os
from docsentinel2.ingestion import load_document, load_xlsx
from docsentinel2.embeddings import EmbeddingModel
from docsentinel2.alignment import SemanticAligner
from docsentinel2.diff_engine import DiffEngine
from docsentinel2.table_diff import diff_workbooks
from docsentinel2.report import generate_report


def run_semantic_pdf_diff(old_path, new_path):
    print("\nRunning semantic diff on PDF...\n")

    old_sentences = load_document(old_path)
    new_sentences = load_document(new_path)

    print(f"Old sentences: {len(old_sentences)} | New sentences: {len(new_sentences)}")

    embed_model = EmbeddingModel()
    aligner = SemanticAligner(embed_model)
    engine = DiffEngine(embedder=embed_model, aligner=aligner)

    results = engine.detect_changes(old_sentences, new_sentences)
    return results


def run_excel_table_diff(old_path, new_path):
    print("\nRunning table diff...\n")

    embedder = EmbeddingModel()
    old_wb = load_xlsx(old_path)
    new_wb = load_xlsx(new_path)

    results = diff_workbooks(old_wb, new_wb, embedder)
    return results


def main():
    old_doc_path = "data/old.xlsx"
    new_doc_path = "data/new.xlsx"

    print(f"Loading documents:\n  OLD: {old_doc_path}\n  NEW: {new_doc_path}")

    ext = os.path.splitext(old_doc_path)[1].lower()

    if ext == ".xlsx":
        results = run_excel_table_diff(old_doc_path, new_doc_path)
    elif ext == ".pdf":
        results = run_semantic_pdf_diff(old_doc_path, new_doc_path)
    else:
        raise ValueError("Unsupported format. Only .pdf and .xlsx supported")

    if not results:
        print("\nNo significant changes detected 🎉")
    else:
        for i, ch in enumerate(results, 1):
            print(f"=== Change #{i} ===")
            for k, v in ch.items():
                print(f"{k}: {v}")
            print()

        os.makedirs("outputs", exist_ok=True)
        generate_report(results, "outputs/report.csv")
        print("Report saved to outputs/report.csv")

    print("\nCompleted successfully ✅")


if __name__ == "__main__":
    main()
