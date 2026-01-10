import torch
import cv2
import numpy as np

class ImageChangeModel:
    def __init__(self, model, weights_path, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.eval()

    @torch.no_grad()
    def predict_heatmap(self, img_old_path, img_new_path):
        """
        Returns:
            heatmap (H, W) numpy array in [0,1]
        """

        img1 = cv2.imread(img_old_path)
        img2 = cv2.imread(img_new_path)

        if img1 is None or img2 is None:
            return None

        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))

        img1 = img1.transpose(2, 0, 1) / 255.0
        img2 = img2.transpose(2, 0, 1) / 255.0

        x1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).to(self.device)
        x2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 🔥 Forward pass
        out = self.model(x1, x2)  # shape: (1, 1, H, W)

        heatmap = out.squeeze().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)

        return heatmap
