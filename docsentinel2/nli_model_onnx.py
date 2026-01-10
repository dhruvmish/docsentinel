# docsentinel2/nli_model_onnx.py

import os
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ONNX_PATH = "models/nli_roberta_large.onnx"
HF_MODEL = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"


class NLIModelONNX:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

        if os.path.exists(ONNX_PATH):
            print(f"[NLI] Using ONNX Runtime model: {ONNX_PATH}")
            self.session = ort.InferenceSession(
                ONNX_PATH,
                providers=["CPUExecutionProvider"]
            )
            self.use_onnx = True
        else:
            print("[NLI] ONNX model missing — falling back to PyTorch")
            self.model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
            self.model.eval()
            self.use_onnx = False

    def _infer(self, premise, hypothesis):
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        if self.use_onnx:
            ort_inputs = {
                k: v.cpu().numpy()
                for k, v in inputs.items()
            }
            logits = self.session.run(None, ort_inputs)[0]
            probs = torch.softmax(torch.tensor(logits), dim=1).squeeze()
        else:
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=1).squeeze()

        return {
            "entail": float(probs[0]),
            "neutral": float(probs[1]),
            "contradict": float(probs[2]),
        }

    def bidirectional(self, a, b):
        return self._infer(a, b), self._infer(b, a)
