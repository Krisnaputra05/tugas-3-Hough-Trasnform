import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. Baca dan ubah ke grayscale ===
img = cv2.imread("./img/jendela2.jpg")
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Pastikan path './img/jendela2.jpg' benar.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === 2. Kurangi noise ===
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# === 3. Deteksi tepi (Canny Edge Detection) ===
edges = cv2.Canny(blur, 50, 50)

# === 4. Deteksi garis dengan Hough Transform Probabilistik ===
lines = cv2.HoughLinesP(
    edges,
    rho=1,             # resolusi jarak dalam piksel
    theta=np.pi / 180, # resolusi sudut dalam radian
    threshold=125,      # jumlah minimum voting agar dianggap garis
    minLineLength=5,  # panjang minimum garis
    maxLineGap=80      # jarak maksimum antar segmen agar dianggap satu garis
)

# === 5. Gambar garis hasil deteksi ===
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# === 6. Tampilkan hasil ===
plt.figure(figsize=(7, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Deteksi Garis dengan Hough Transform (Versi Stabil)')
plt.axis('off')
plt.show()

# (Opsional) tampilkan tepi hasil Canny juga
# plt.imshow(edges, cmap='gray')
# plt.title("Deteksi Tepi (Canny)")
# plt.axis('off')
# plt.show()
