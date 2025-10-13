import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. Baca dan ubah ke grayscale ===
img = cv2.imread("./img/jalan.jpg")
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Pastikan path './img/jalan.jpg' benar.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === 2. Kurangi noise ===
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# === 3. Deteksi tepi (Canny Edge Detection) ===
# Gunakan threshold lebih tinggi agar hanya tepi jalan yang jelas terdeteksi
edges = cv2.Canny(blur, 70, 180)

# === 4. Deteksi garis dengan Hough Transform Probabilistik ===
lines = cv2.HoughLinesP(
    edges,
    rho=1,                  # resolusi jarak (piksel)
    theta=np.pi / 180,      # resolusi sudut (radian)
    threshold=150,          # jumlah minimum voting agar dianggap garis
    minLineLength=120,       # panjang minimum garis
    maxLineGap=50           # jarak maksimum antar segmen agar dianggap satu garis
)

# === 5. Gambar garis hasil deteksi ===
output = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

# === 6. Tampilkan hasil ===
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Deteksi Garis Jalan (HoughLinesP)')
plt.axis('off')
plt.show()

# (Opsional) tampilkan tepi hasil Canny juga
# plt.imshow(edges, cmap='gray')
# plt.title("Deteksi Tepi (Canny)")
# plt.axis('off')
# plt.show()
