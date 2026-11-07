# Hands-on Perception 2: Roboflow sampai best.pt

Tidak butuh VRX atau Gazebo. Cukup browser, akun Google (untuk Colab), dan laptop dengan Python.

```
gambar  ->  label di Roboflow  ->  versi dataset  ->  ekspor YOLOv8  ->  training di Colab  ->  best.pt  ->  deteksi.py
```

## Isi folder

| Berkas | Fungsi |
|---|---|
| `dataset/gambar/` | 120 gambar buoy VRX **belum dilabel**, untuk dilabel di Roboflow |
| `dataset/README.md` | Daftar kelas dan aturan pelabelan, contoh di `dataset/contoh_label.png` |
| `bahan/video_uji_vrx.mp4` | Video uji 32 detik (160 frame) untuk mencoba model tanpa webcam |
| `model/best_pemateri.pt` | Model YOLOv8n dari pemateri, 4 kelas: `red`, `green`, `orange`, `black` |
| `train_colab_yolov8.ipynb` | Notebook training di Google Colab |
| `deteksi.py` | Menjalankan `best.pt` pada webcam, video, gambar, atau folder |

## 0. Persiapan

```bash
git clone https://github.com/ahnafyura/magang-roboflow-PGR.git
cd magang-roboflow-PGR
```

1. Daftar di https://roboflow.com (gratis, bisa login dengan Google atau GitHub). Buat workspace.
2. Pastikan bisa membuka https://colab.research.google.com.
3. Di laptop:
   ```bash
   pip install "numpy<2" "opencv-python<4.12" "setuptools<80" ultralytics
   python3 deteksi.py --model model/best_pemateri.pt --source bahan/video_uji_vrx.mp4
   ```
   Kalau jendela muncul dengan kotak berwarna di buoy, laptop sudah siap. Tekan `q` untuk keluar.

## 1. Buat project

**Create New Project**, tipe **Object Detection**. Isi nama project dan nama objek (misalnya `buoy`). Daftar kelas ada di `dataset/README.md`.
Nama kelas huruf kecil dan konsisten: `red`, `green`, `orange`, `black`. `Red` dan `red` dianggap dua kelas berbeda.

## 2. Upload gambar

Seret folder `dataset/gambar/` (120 gambar) ke halaman **Upload**. Untuk benda sendiri: foto dengan ponsel dari banyak sudut, jarak, latar, dan cahaya. Sertakan juga beberapa gambar tanpa objek.

## 3. Anotasi

Buka **Annotate**, pilih gambar, gambar kotak dengan alat **Bounding Box**, pilih kelasnya.

- Kotak serapat mungkin dengan objek.
- Labeli **semua** objek di gambar, termasuk yang kecil dan jauh.
- Gambar tanpa objek: tandai sebagai null (kosong), jangan dihapus.
- Setelah sekitar 20 gambar, coba **Label Assist** untuk mempercepat, tetapi tetap periksa hasilnya.

## 4. Masukkan ke dataset

**Add images to Dataset**, bagi train/valid/test sekitar **70/20/10**.

## 5. Buat versi dataset

Menu **Versions** (Generate):

- Preprocessing: **Auto-Orient**, **Resize 640 x 640**.
- Augmentation (hanya diterapkan ke train): flip horizontal, brightness plus minus 20%, sedikit blur atau rotasi plus minus 10 derajat.
- Klik **Generate**.

## 6. Ekspor

**Download Dataset**, format **YOLOv8**, pilih **Show download code** (untuk Colab) atau unduh zip. Isi ekspor:

```
dataset/
  data.yaml          # nc, names, path train/valid/test
  train/images  train/labels
  valid/images  valid/labels
  test/images   test/labels
```

Jangan bagikan API key di repo atau grup.

## 7. Training di Colab

1. Buka `train_colab_yolov8.ipynb` di Colab (**File > Upload notebook**).
2. **Runtime > Change runtime type > T4 GPU**.
3. Isi `WORKSPACE`, `PROJECT`, `VERSION` dari kode unduhan Roboflow, jalankan sel dari atas ke bawah.
4. Catat mAP50, mAP50-95, precision, recall dari sel 7. Simpan `results.png` dan `confusion_matrix.png`.
5. Unduh `best.pt` (sel 9).

Training lokal tanpa Colab juga bisa:
```bash
yolo detect train data=<folder_ekspor>/data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=8
```
Tanpa GPU tambahkan `device=cpu` dan kurangi epoch.

## 8. Uji model di laptop

```bash
python3 deteksi.py --model best.pt --source 0                       # webcam
python3 deteksi.py --model best.pt --source bahan/video_uji_vrx.mp4 # video
python3 deteksi.py --model best.pt --source foto.jpg --conf 0.4     # satu gambar
python3 deteksi.py --model best.pt --source folder/ --no-show --save hasil/
```

`--conf` menaikkan atau menurunkan ambang keyakinan. Di akhir dicetak rata-rata FPS dan jumlah deteksi per kelas.

## 9. Bandingkan dengan best_pemateri.pt

- Kelas sama (dataset buoy): bandingkan mAP lewat sel 10 notebook.
- Kelas berbeda (benda sendiri): bandingkan hasil deteksi pada gambar atau video yang sama.

## Masalah umum

| Gejala | Penyebab dan solusi |
|---|---|
| `images not found` saat training | Path di `data.yaml`. Jalankan sel 4 notebook (menulis ulang menjadi path absolut). |
| Training sangat lambat di Colab | Runtime belum GPU. Cek sel 1 (`nvidia-smi`). |
| `CUDA out of memory` | Turunkan `batch` ke 8 atau 4. |
| Kelas muncul dobel (`Red` dan `red`) | Nama kelas tidak konsisten. Perbaiki di Roboflow (Classes), buat versi baru. |
| mAP tinggi tapi gagal di webcam | Data terlalu seragam. Tambah gambar dengan latar dan cahaya berbeda. |
| `webcam 0 tidak bisa dibuka` | Coba `--source 1`, atau tutup aplikasi lain yang memakai kamera. |
| Galat `check_amp` saat training lokal | Tambahkan `amp=False`. |
