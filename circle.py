import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("./img/koin4.jpg")
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Pastikan path './img/koin4.jpg' benar.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)


gray_blur = cv2.medianBlur(gray, 7)  # median blur lebih cocok untuk lingkaran

circles = cv2.HoughCircles(
    gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,         # resolusi akumulator (1.2 cukup presisi)
    minDist=60,     # jarak minimum antar pusat lingkaran
    param1=150,     # ambang atas Canny edge
    param2=40,      # ambang voting lingkaran (semakin kecil = lebih sensitif)
    minRadius=25,   # radius minimum lingkaran
    maxRadius=90    # radius maksimum lingkaran
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        # Gambar lingkaran luar (hijau)
        cv2.circle(img, (x, y), r, (0, 255, 0), 3)
        # Gambar pusat lingkaran (merah)
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)


plt.figure(figsize=(7, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Deteksi Lingkaran dengan Hough Transform (Versi Stabil)')
plt.axis('off')
plt.show()
