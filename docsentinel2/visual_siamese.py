import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2


class SiameseMobileNet(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(1280, embedding_dim)

    def forward_once(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x

    def forward(self, x1, x2):
        f1 = self.forward_once(x1)
        f2 = self.forward_once(x2)
        return f1, f2


# ===== Utility: Feature Embeddings =====

_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

_device = torch.device("cpu")
_model = SiameseMobileNet().to(_device)
_model.eval()


def load_image(path):
    img = Image.open(path).convert("RGB")
    return _transform(img).unsqueeze(0).to(_device)


def siamese_similarity(path1, path2):
    try:
        t1 = load_image(path1)
        t2 = load_image(path2)
        with torch.no_grad():
            f1, f2 = _model(t1, t2)
        cos = torch.nn.functional.cosine_similarity(f1, f2).item()
        return float(cos)
    except:
        return None


# ===== Change Heatmap + Bounding Boxes =====

def diff_heatmap(path1, path2, threshold=25):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        return None, None, []

    # Resize to match neural input
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

    # Absolute image difference
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Binary mask
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Cleanup noise
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    # Detect change contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h > 50:  # ignore tiny artifacts
            boxes.append((x, y, w, h))

    return img2, mask, boxes
