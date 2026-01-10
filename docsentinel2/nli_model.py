from typing import Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEFAULT_MODEL_NAME = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ONNX_PATH = BASE_DIR / "models" / "nli_roberta_large.onnx"


class NLIModel:
    """
    Wrapper around large NLI model with optional ONNX acceleration.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        onnx_path: Optional[str] = None,
        use_onnx: bool = True,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.use_onnx = False
        self.onnx_session = None
        self.onnx_input_names = []
        self.onnx_output_name = None
        self._ort = None

        # Try ONNX
        if use_onnx:
            try:
                import onnxruntime as ort
                self._ort = ort

                if onnx_path is None:
                    onnx_path = str(DEFAULT_ONNX_PATH)

                onnx_file = Path(onnx_path)
                if onnx_file.exists():
                    print(f"[NLI] Using ONNX Runtime model: {onnx_file}")
                    self.onnx_session = ort.InferenceSession(
                        str(onnx_file),
                        providers=["CPUExecutionProvider"]
                    )
                    self.onnx_input_names = [i.name for i in self.onnx_session.get_inputs()]
                    self.onnx_output_name = self.onnx_session.get_outputs()[0].name
                    self.use_onnx = True
                else:
                    print(f"[NLI] ONNX model not found at {onnx_file}. Falling back to PyTorch.")
            except ImportError:
                print("[NLI] ONNXRuntime not installed. Using PyTorch.")

        # If ONNX failed → load PyTorch model
        if not self.use_onnx:
            print(f"[NLI] Loading PyTorch model: {model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()

    @staticmethod
    def _softmax_np(logits: np.ndarray) -> np.ndarray:
        exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def _infer_onnx(self, premise: str, hypothesis: str) -> Dict[str, float]:
        inputs_pt = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True, max_length=512,
        )
        inputs_np = {k: v.cpu().numpy() for k, v in inputs_pt.items()}

        onnx_inputs = {}
        for name in self.onnx_input_names:
            key = name
            if key not in inputs_np:
                for k in inputs_np:
                    if name.endswith(k):
                        key = k
                        break
            onnx_inputs[name] = inputs_np[key]

        logits = self.onnx_session.run([self.onnx_output_name], onnx_inputs)[0]
        probs = self._softmax_np(logits)[0]

        return {
            "entail": float(probs[0]),
            "neutral": float(probs[1]),
            "contradict": float(probs[2]),
        }

    def _infer_torch(self, premise: str, hypothesis: str) -> Dict[str, float]:
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True, max_length=512,
        )
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=1).squeeze()

        return {
            "entail": probs[0].item(),
            "neutral": probs[1].item(),
            "contradict": probs[2].item(),
        }

    def _infer(self, premise, hypothesis):
        if self.use_onnx and self.onnx_session is not None:
            return self._infer_onnx(premise, hypothesis)
        return self._infer_torch(premise, hypothesis)

    def bidirectional(self, a: str, b: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        p1 = self._infer(a, b)
        p2 = self._infer(b, a)

        # 🔥 Guarantee all expected keys always exist
        for p in (p1, p2):
            for k in ["entail", "neutral", "contradict"]:
                if k not in p:
                    p[k] = 0.0

        return p1, p2
