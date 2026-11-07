#!/usr/bin/env python3
"""Jalankan model YOLO (best.pt) pada webcam, video, gambar, atau folder gambar.

Contoh:
    python3 deteksi.py --model model/best_pemateri.pt --source 0              # webcam
    python3 deteksi.py --model best.pt --source bahan/video_uji_vrx.mp4
    python3 deteksi.py --model best.pt --source foto.jpg --conf 0.4
    python3 deteksi.py --model best.pt --source folder_gambar/ --save hasil/

Tekan q untuk berhenti. Tanpa layar (SSH / Colab) pakai --no-show dan --save.
"""
import argparse
import os
import time

import cv2
from ultralytics import YOLO

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def sources(src):
    """Hasilkan (nama, frame) dari webcam, video, satu gambar, atau folder."""
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        if not cap.isOpened():
            raise SystemExit(f'webcam {src} tidak bisa dibuka')
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f'webcam_{i:05d}.jpg', frame
            i += 1
        cap.release()
    elif os.path.isdir(src):
        for name in sorted(os.listdir(src)):
            if name.lower().endswith(IMG_EXT):
                frame = cv2.imread(os.path.join(src, name))
                if frame is not None:
                    yield name, frame
    elif src.lower().endswith(IMG_EXT):
        frame = cv2.imread(src)
        if frame is None:
            raise SystemExit(f'gambar {src} tidak bisa dibaca')
        yield os.path.basename(src), frame
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise SystemExit(f'video {src} tidak bisa dibuka')
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f'frame_{i:05d}.jpg', frame
            i += 1
        cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='path best.pt')
    ap.add_argument('--source', default='0', help='0 = webcam, atau path video / gambar / folder')
    ap.add_argument('--conf', type=float, default=0.25, help='ambang confidence (0..1)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default=None, help='cpu atau 0 (GPU). Kosong = otomatis')
    ap.add_argument('--save', default=None, help='folder untuk menyimpan gambar hasil')
    ap.add_argument('--no-show', action='store_true', help='jangan buka jendela')
    args = ap.parse_args()

    model = YOLO(args.model)
    print('kelas model:', model.names)
    if args.save:
        os.makedirs(args.save, exist_ok=True)

    total = {}
    n = 0
    t0 = time.time()
    for name, frame in sources(args.source):
        t = time.time()
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        fps = 1.0 / max(time.time() - t, 1e-6)

        for (x1, y1, x2, y2), cls, score in zip(result.boxes.xyxy.tolist(),
                                                result.boxes.cls.tolist(),
                                                result.boxes.conf.tolist()):
            label = model.names[int(cls)]
            total[label] = total.get(label, 0) + 1
            if n % 30 == 0 or not args.source.isdigit():
                print(f'{name}: {label} {score:.2f} kotak=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})')

        out = result.plot()
        cv2.putText(out, f'{fps:.1f} FPS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        if args.save:
            cv2.imwrite(os.path.join(args.save, name), out)
        if not args.no_show:
            cv2.imshow('deteksi (q untuk keluar)', out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        n += 1

    cv2.destroyAllWindows()
    dt = time.time() - t0
    print(f'\n{n} frame dalam {dt:.1f} s (rata-rata {n / max(dt, 1e-6):.1f} FPS)')
    print('jumlah deteksi per kelas:', total if total else 'tidak ada')


if __name__ == '__main__':
    main()
