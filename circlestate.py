import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. Load Gambar ===
img = cv2.imread("./img/lingkaran.jpg")
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Pastikan path benar!")

# === 2. Preprocessing (CLAHE + median blur) ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)
gray_blur = cv2.medianBlur(gray, 7)

# === 3. Hough Circle Detection ===
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

if circles is None:
    print("Tidak ada lingkaran terdeteksi!")
    exit()

circles = np.around(circles[0, :]).astype(float)

print("\n=== DATA KOIN TERDETEKSI ===")
for i, (x, y, r) in enumerate(circles, start=1):
    print(f"Koin {i}: Pusat=({x:.0f},{y:.0f}), Radius={r:.0f}")

# === Fungsi input ===
def pilih_koin(pesan, max_idx):
    while True:
        try:
            idx = int(input(pesan))
            if 1 <= idx <= max_idx:
                return idx
            else:
                print(f"Masukkan angka antara 1 - {max_idx}")
        except:
            print("Input harus angka!")

# Pilih lingkaran
k1 = pilih_koin("\nPilih koin pertama: ", len(circles))
k2 = pilih_koin("Pilih koin kedua : ", len(circles))

(x1, y1, r1) = circles[k1 - 1]
(x2, y2, r2) = circles[k2 - 1]

# === 6. Hitung jarak antar pusat ===
d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
print(f"\nJarak antara Koin {k1} dan Koin {k2} = {d:.2f} px")

# === 7. Hitung Titik Perpotongan Lingkaran ===
def titik_perpotongan(x1, y1, r1, x2, y2, r2):
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if d > r1 + r2:
        return None  # tidak berpotongan (terpisah)
    if d < abs(r1 - r2):
        return None  # satu di dalam lain
    if d == 0 and r1 == r2:
        return None  # lingkaran sama

    # rumus geometri titik potong
    a = (r1*r1 - r2*r2 + d*d) / (2 * d)
    h = np.sqrt(abs(r1*r1 - a*a))

    x3 = x1 + a * (x2 - x1) / d
    y3 = y1 + a * (y2 - y1) / d

    p1x = x3 + h * (y2 - y1) / d
    p1y = y3 - h * (x2 - x1) / d

    p2x = x3 - h * (y2 - y1) / d
    p2y = y3 + h * (x2 - x1) / d

    return (p1x, p1y), (p2x, p2y)

intersections = titik_perpotongan(x1, y1, r1, x2, y2, r2)

output = img.copy()

# gambar lingkaran
for i, (x, y, r) in enumerate(circles, start=1):
    cv2.circle(output, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.putText(output, str(i), (int(x-10), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

# Jika ada 2 titik perpotongan
if intersections is not None:
    (p1x, p1y), (p2x, p2y) = intersections
    p1 = (int(p1x), int(p1y))
    p2 = (int(p2x), int(p2y))

    # gambar titik potong
    cv2.circle(output, p1, 6, (0, 0, 255), -1)
    cv2.circle(output, p2, 6, (0, 0, 255), -1)

    # gambar garis penghubung titik potong
    cv2.line(output, p1, p2, (0, 0, 255), 3)

    # hitung panjang garis
    panjang = np.sqrt((p2x - p1x)**2 + (p2y - p1y)**2)
    print(f"\n=== GARIS PERPOTONGAN ===")
    print(f"Titik 1 = {p1}")
    print(f"Titik 2 = {p2}")
    print(f"Panjang garis perpotongan = {panjang:.2f} px")

    # label teks
    mx = int((p1x + p2x) / 2)
    my = int((p1y + p2y) / 2)
    cv2.putText(output, f"{panjang:.1f}px", (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

else:
    print("\nLingkaran TIDAK berpotongan!")

# tampilkan hasil
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Perpotongan Lingkaran")
plt.axis("off")
plt.show()
