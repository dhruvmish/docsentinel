# docsentinel2/nli_model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Tuple


class NLIModel:
    def __init__(self, model_name: str = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def _infer(self, premise: str, hypothesis: str) -> Dict[str, float]:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze()
        return {
            "entail": probs[0].item(),
            "neutral": probs[1].item(),
            "contradict": probs[2].item(),
        }

    def bidirectional(self, a: str, b: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        p1 = self._infer(a, b)
        p2 = self._infer(b, a)
        return p1, p2
