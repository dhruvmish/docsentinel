# docsentinel2/visual_model.py

import os
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import cv2
except ImportError:
    cv2 = None


class SiameseONNX:
    """
    Lightweight wrapper around an ONNX Siamese / embedding model.

    - If onnxruntime or the .onnx file is missing, all methods return None
      and the visual diff falls back to pHash + SSIM only.
    """
    def __init__(self, model_path: str, input_size=(224, 224)):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.session: Optional["ort.InferenceSession"] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None

        if ort is None:
            print("[visual_model] onnxruntime not installed; Siamese model disabled.")
            return

        if not self.model_path.exists():
            print(f"[visual_model] ONNX model not found at {self.model_path}. "
                  "Visual Siamese checks will be skipped.")
            return

        if cv2 is None:
            print("[visual_model] OpenCV not installed; Siamese model disabled.")
            return

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[visual_model] Loaded Siamese model from {self.model_path}")

    def _preprocess(self, img_path: str) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV not available")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.astype("float32") / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, 0)        # (1, C, H, W)
        return img

    def embed(self, img_path: str) -> Optional[np.ndarray]:
        if self.session is None:
            return None
        x = self._preprocess(img_path)
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        # assume output shape (1, D)
        return out[0]

    def similarity(self, img1: str, img2: str) -> Optional[float]:
        """
        Returns cosine similarity between two images in [−1, 1].
        Returns None if the model/session is not available.
        """
        emb1 = self.embed(img1)
        emb2 = self.embed(img2)
        if emb1 is None or emb2 is None:
            return None
        denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2)) + 1e-9
        sim = float(np.dot(emb1, emb2) / denom)
        return sim
