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

# Variabel global untuk menyimpan properti geometri jika perpotongan terjadi
geo_properties = {}

# === 7. Hitung Titik Perpotongan Lingkaran (Dimodifikasi untuk menyimpan properti) ===
def titik_perpotongan(x1, y1, r1, x2, y2, r2):
    global geo_properties
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Cek kondisi perpotongan
    if d > r1 + r2 or d < abs(r1 - r2) or (d == 0 and r1 == r2):
        geo_properties = {'d': d, 'status': 'Tidak berpotongan'}
        return None 
    
    # 1. Hitung Jarak Proyeksi 'a' (Jarak dari C1 ke Tali Busur Umum)
    a = (r1*r1 - r2*r2 + d*d) / (2 * d)
    
    # 2. Hitung Setengah Tali Busur 'h'
    h = np.sqrt(abs(r1*r1 - a*a))

    # 3. Hitung Titik Proyeksi (x3, y3) pada garis pusat
    x3 = x1 + a * (x2 - x1) / d
    y3 = y1 + a * (y2 - y1) / d

    # 4. Hitung Dua Titik Perpotongan (P1 dan P2)
    p1x = x3 + h * (y2 - y1) / d
    p1y = y3 - h * (x2 - x1) / d

    p2x = x3 - h * (y2 - y1) / d
    p2y = y3 + h * (x2 - x1) / d
    
    # Simpan semua properti yang dihitung
    geo_properties = {
        'd': d,                 # Jarak antar pusat
        'a': a,                 # Jarak proyeksi dari C1 ke tali busur
        'h': h,                 # Setengah panjang tali busur
        'x3': x3,               # Koordinat X Titik Proyeksi (pusat tali busur)
        'y3': y3,               # Koordinat Y Titik Proyeksi (pusat tali busur)
        'p1': (p1x, p1y),       # Titik Perpotongan 1
        'p2': (p2x, p2y),       # Titik Perpotongan 2
        'status': 'Berpotongan'
    }

    return (p1x, p1y), (p2x, p2y)

intersections = titik_perpotongan(x1, y1, r1, x2, y2, r2)

# --- Pencetakan Properti Geometri ---
print("\n=============================================")
print(f"=== PROPERTI GEOMETRI (Koin {k1} vs Koin {k2}) ===")
print("=============================================")
print(f"Pusat Koin {k1} (C1) : ({x1:.2f}, {y1:.2f}) | R1 = {r1:.2f}")
print(f"Pusat Koin {k2} (C2) : ({x2:.2f}, {y2:.2f}) | R2 = {r2:.2f}")
print("--- Variabel Utama ---")

if geo_properties['status'] != 'Tidak berpotongan':
    # Variabel Proyeksi
    a = geo_properties['a']
    h = geo_properties['h']
    x3 = geo_properties['x3']
    y3 = geo_properties['y3']
    
    # Titik Perpotongan
    p1x, p1y = geo_properties['p1']
    p2x, p2y = geo_properties['p2']
    panjang = 2 * h
    
    print(f"d (Jarak Pusat)          : {d:.2f} px")
    print(f"a (Jarak Proyeksi dari C1) : {a:.2f} px")
    print(f"h (Setengah Tali Busur)  : {h:.2f} px")
    print("--- Koordinat ---")
    print(f"Pusat Tali Busur (x3, y3): ({x3:.2f}, {y3:.2f})")
    print(f"Titik Perpotongan P1     : ({p1x:.2f}, {p1y:.2f})")
    print(f"Titik Perpotongan P2     : ({p2x:.2f}, {p2y:.2f})")
    print("--- Hasil Akhir ---")
    print(f"Panjang Tali Busur       : {panjang:.2f} px")
else:
    print(f"d (Jarak Pusat)          : {d:.2f} px")
    print("Status: Lingkaran tidak berpotongan!")
print("=============================================")
# --- Akhir Pencetakan Properti Geometri ---


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
    
    # label teks
    mx = int((p1x + p2x) / 2)
    my = int((p1y + p2y) / 2)
    cv2.putText(output, f"{panjang:.1f}px", (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

else:
    # Jika tidak berpotongan, teks sudah dicetak di bagian properti
    pass

# tampilkan hasil
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title(f"Perpotongan Lingkaran (Koin {k1} & Koin {k2})")
plt.axis("off")
plt.show()