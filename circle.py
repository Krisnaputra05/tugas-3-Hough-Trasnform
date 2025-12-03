import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Baca gambar ===
img = cv2.imread("./img/lingkaran.jpg")
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Pastikan path './img/lingkaran.jpg' benar.")

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
    dp=1.2,
    minDist=60,
    param1=150,
    param2=40,
    minRadius=25,
    maxRadius=90
)

# === Jika lingkaran ditemukan ===
if circles is not None:
    circles = np.uint16(np.around(circles))

    print("\n=== HASIL DETEKSI KOIN ===")
    for idx, (x, y, r) in enumerate(circles[0, :], start=1):

        # Gambar lingkaran hijau
        cv2.circle(img, (x, y), r, (0, 255, 0), 3)

        # Titik pusat (biru)
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

        # Nomor koin (merah) sedikit di atas lingkaran
        cv2.putText(img, f"{idx}", (x - 10, y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)

        # Hitung luas
        luas = np.pi * (r ** 2)

        print(f"Koin {idx}: Pusat=({x},{y}), Radius={r}px, Luas={luas:.2f} px²")
    print("============================\n")

# === Tampilkan hasil ===
plt.figure(figsize=(7, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Deteksi & Penomoran Lingkaran (Koin)')
plt.axis('off')
plt.show()
