# Dataset latihan anotasi

`gambar/` berisi 120 gambar kamera depan WAM-V dari simulasi VRX (1280 x 720), **belum dilabel**.
Sebagian kecil gambar sengaja tidak berisi buoy (contoh negatif): tandai sebagai null di Roboflow, jangan dihapus.

## Kelas

Pakai nama kelas persis seperti ini (huruf kecil):

| Kelas | Objek |
|---|---|
| `red` | buoy merah (kerucut tinggi) |
| `green` | buoy hijau (kerucut tinggi) |
| `orange` | buoy oranye (bulat kecil) |
| `black` | buoy hitam (bulat kecil) |

`contoh_label.png` menunjukkan contoh kotak yang benar: rapat, dan semua buoy dilabel, termasuk yang kecil di kejauhan.

## Cara pakai

1. Roboflow: **Create New Project**, tipe **Object Detection**.
2. Halaman **Upload**: seret seluruh folder `gambar/`.
3. **Annotate**: beri kotak pada setiap buoy dengan kelas di atas.
4. Lanjutkan sesuai `README.md` di root repo (split, versi, ekspor YOLOv8, training).
