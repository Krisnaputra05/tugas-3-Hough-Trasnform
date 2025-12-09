import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. Baca dan ubah ke grayscale ===
img = cv2.imread("./img/jendela2.jpg")

if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Pastikan path benar.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 80, 200)

# 2. Hough Lines
lines = cv2.HoughLinesP(
    edges, 1, np.pi/180,
    threshold=300,
    minLineLength=150,
    maxLineGap=50
)

print("\n=== DETEKSI GARIS ===")

# 3. Pisahkan garis vertikal & horizontal
vertical_lines = []
horizontal_lines = []

for x1, y1, x2, y2 in lines[:,0]:
    if abs(x1 - x2) < 10:  # vertical
        vertical_lines.append(x1)
    if abs(y1 - y2) < 10:  # horizontal
        horizontal_lines.append(y1)

# 4. Sort dan buang duplikat
vertical_lines = sorted(list(set(vertical_lines)))
horizontal_lines = sorted(list(set(horizontal_lines)))

# Ambil 9 garis (untuk 8 kotak)
vertical_lines = vertical_lines[:9]
horizontal_lines = horizontal_lines[:9]

print(f"Jumlah garis vertikal   : {len(vertical_lines)}")
print("Posisi garis vertikal   :", vertical_lines)

print(f"\nJumlah garis horizontal : {len(horizontal_lines)}")
print("Posisi garis horizontal :", horizontal_lines)

# 5. Gambar grid linier
only_lines = np.zeros_like(img)
for x in vertical_lines:
    cv2.line(only_lines, (x, horizontal_lines[0]), (x, horizontal_lines[-1]), (0,255,0), 3)

for y in horizontal_lines:
    cv2.line(only_lines, (vertical_lines[0], y), (vertical_lines[-1], y), (0,255,0), 3)

# 6. Pilih kotak tertentu
row = 3
col = 5

x_start = vertical_lines[col]
x_end   = vertical_lines[col+1]
y_start = horizontal_lines[row]
y_end   = horizontal_lines[row+1]

box_w = x_end - x_start
box_h = y_end - y_start
box_area = box_w * box_h

print("\n=== DETAIL KOTAK TERPILIH ===")
print(f"Baris, Kolom               : ({row}, {col})")
print(f"x_start                    : {x_start}")
print(f"x_end                      : {x_end}")
print(f"y_start                    : {y_start}")
print(f"y_end                      : {y_end}")
print(f"Lebar kotak (px)           : {box_w}")
print(f"Tinggi kotak (px)          : {box_h}")
print(f"Luas kotak (px)            : {box_area}\n")

# 7. Tandai pada only_lines
only_lines_marked = only_lines.copy()
cv2.rectangle(only_lines_marked, (x_start, y_start), (x_end, y_end), (0,0,255), 3)
cv2.putText(only_lines_marked, f"({row},{col})", (x_start, y_start - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

# 8. Tandai pada gambar asli
marked = img.copy()
cv2.rectangle(marked, (x_start, y_start), (x_end, y_end), (0,0,255), 3)
cv2.putText(marked, f"Luas: {box_area} px", (x_start, y_start - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

# 9. Tampilkan hasil
plt.figure(figsize=(8, 8))  # ukuran lebih square agar proporsional

plt.imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
plt.title("Kotak yang dihitung (Akurat)")
plt.axis("off")

plt.tight_layout()
plt.show()