import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from safetensors.torch import load_file

import cv2

def heatmap_to_boxes(heatmap, threshold=0.45):
    # Normalize 0-255
    hmap = (heatmap * 255).astype("uint8")
    _, mask = cv2.threshold(hmap, int(threshold * 255), 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x + w, y + h))

    return boxes, mask



class ImageChangeModel:
    def __init__(self, model_path: str, device="cpu"):
        self.device = device

        # Load safetensors
        state = load_file(model_path)

        # Rebuild model architecture
        from your_model_def import MyChangeDetector  # <- your original class
        self.model = MyChangeDetector()
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def predict_heatmap(self, imgA_path, imgB_path):
        imgA = Image.open(imgA_path).convert("RGB")
        imgB = Image.open(imgB_path).convert("RGB")

        tA = self.transform(imgA).unsqueeze(0).to(self.device)
        tB = self.transform(imgB).unsqueeze(0).to(self.device)

        with torch.no_grad():
            heatmap = self.model(tA, tB)  # shape [1,1,H,W]

        return heatmap.squeeze(0).squeeze(0).cpu().numpy()
