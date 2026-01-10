# main_nli_onnx_test.py

import os

from docsentinel2.ingestion import load_document
from docsentinel2.embeddings import EmbeddingModel
from docsentinel2.alignment import SemanticAligner
from docsentinel2.diff_engine import DiffEngine


def run_semantic_pdf_diff(old_path, new_path):
    print("\n▶ Running Semantic Text Diff (ONNX-accelerated if available)...\n")

    old_sentences = load_document(old_path)
    new_sentences = load_document(new_path)

    print(f"Old sentences: {len(old_sentences)} | New sentences: {len(new_sentences)}")

    embed_model = EmbeddingModel()
    aligner = SemanticAligner(embed_model)
    engine = DiffEngine(embedder=embed_model, aligner=aligner)

    changes = engine.detect_changes(old_sentences, new_sentences)
    return changes


def main():
    old_path = "data/old.pdf"
    new_path = "data/new.pdf"

    print("=== DocSentinel NLI ONNX Test ===")
    print(f"OLD: {old_path}")
    print(f"NEW: {new_path}")

    if not (os.path.exists(old_path) and os.path.exists(new_path)):
        print("\n[ERROR] Please ensure data/old.pdf and data/new.pdf exist.")
        return

    changes = run_semantic_pdf_diff(old_path, new_path)

    if not changes:
        print("\n✅ No semantic changes detected.")
    else:
        print(f"\nDetected {len(changes)} semantic changes:")
        for i, ch in enumerate(changes, 1):
            print(f"\n--- Change #{i} ---")
            for k, v in ch.items():
                print(f"{k}: {v}")

    print("\n✔ NLI ONNX Test Completed")


if __name__ == "__main__":
    main()
