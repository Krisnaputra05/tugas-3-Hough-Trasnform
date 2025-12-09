import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Baca gambar ===
img = cv2.imread("./img/lingkaran.jpg")
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Pastikan path './img/koin4.jpg' benar.")

# === Preprocessing ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# CLAHE untuk meningkatkan kontras
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# Blur agar lingkaran lebih stabil terdeteksi
gray_blur = cv2.medianBlur(gray, 7)

# === Hough Circle ===
circles = cv2.HoughCircles(
    gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,         # resolusi akumulator
    minDist=60,     # jarak minimum antar pusat lingkaran
    param1=150,     # ambang atas Canny edge
    param2=40,      # ambang voting lingkaran
    minRadius=25,   # radius minimum lingkaran
    maxRadius=90    # radius maksimum lingkaran
)

if circles is not None:
    circles = np.uint16(np.around(circles))

    print("\n=== HASIL DETEKSI KOIN ===")
    for idx, (x, y, r) in enumerate(circles[0, :], start=1):

        # Gambar lingkaran hijau
        cv2.circle(img, (x, y), r, (0, 255, 0), 3)

        # Titik pusat (biru)
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)


plt.figure(figsize=(7, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Deteksi & Penomoran Lingkaran (Koin)')
plt.axis('off')
plt.show()
