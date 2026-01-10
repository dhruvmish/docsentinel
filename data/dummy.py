import cv2
import fitz  # PyMuPDF
import numpy as np
import imutils
from skimage.metrics import structural_similarity as compare_ssim

def pdf_to_image(pdf_path, page_no=0, zoom=2.0):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_no)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape((pix.height, pix.width, pix.n))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img

# -------- Load Page 1 from both PDFs -------- #
imgA = pdf_to_image("data/Untitled document (21).pdf")
imgB = pdf_to_image("data/Untitled document (22).pdf")

# Resize B to match A
imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))

# -------- SSIM on Color Channels (RGB) -------- #
channelsA = cv2.split(imgA)
channelsB = cv2.split(imgB)
diff_channels = []

for chA, chB in zip(channelsA, channelsB):
    score, diff = compare_ssim(chA, chB, full=True)
    diff_channels.append((diff * 255).astype("uint8"))

diff = cv2.merge(diff_channels)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# -------- Clean noise: Morphology + Threshold -------- #
blur = cv2.GaussianBlur(diff_gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# -------- Find & Filter Contours -------- #
cnts = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

MIN_AREA = 600  # ignore tiny noise changes

for c in cnts:
    if cv2.contourArea(c) < MIN_AREA:
        continue
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(imgA, (x, y), (x+w, y+h), (0, 0, 255), 2)

# -------- Save and show result -------- #
cv2.imwrite("visual_diff_output.png", imgA)
print("✅ Visual diff saved as: visual_diff_output.png")
