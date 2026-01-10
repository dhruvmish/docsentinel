# export_nli_onnx.py
"""
One-time script to export your large NLI model to ONNX for faster inference.

Usage (from project root, with venv activated):

    python export_nli_onnx.py

This will create:
    models/nli_roberta_large.onnx
"""

from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from docsentinel2.nli_model import DEFAULT_MODEL_NAME, DEFAULT_ONNX_PATH


def main():
    models_dir = DEFAULT_ONNX_PATH.parent
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXPORT] Using HF model: {DEFAULT_MODEL_NAME}")
    print(f"[EXPORT] ONNX output path: {DEFAULT_ONNX_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL_NAME)
    model.eval()

    # Dummy input for tracing
    example = tokenizer(
        "premise sentence",
        "hypothesis sentence",
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    input_ids = example["input_ids"]
    attention_mask = example["attention_mask"]

    # We export with explicit input names for clarity
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]

    print("[EXPORT] Exporting to ONNX...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(DEFAULT_ONNX_PATH),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )

    print(f"[EXPORT] Done. ONNX model saved to: {DEFAULT_ONNX_PATH}")


if __name__ == "__main__":
    main()
