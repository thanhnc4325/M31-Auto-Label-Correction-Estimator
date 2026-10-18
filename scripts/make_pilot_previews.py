"""Tạo ảnh preview pre-label cho 100 ảnh pilot (để tab ③ hiển thị trên Streamlit Cloud).

Nguồn (chỉ đọc): overlay đã bàn giao trong data/real/local/end_corrected.zip
  yolo_overlays/<image_id>_overlay.png  -> data/real/pilot_images/detection/<image_id>.jpg
  sam_overlays/<image_id>_overlay.png   -> data/real/pilot_images/segmentation/<image_id>.jpg
Danh sách ảnh = đúng 100 image_id trong data/real/processed/pilot100_prelabels.csv.

Overlay là ảnh BDD100K đã vẽ box YOLOv8n / mask SAM-B. Chỉ đổi PNG -> JPEG chất lượng cao
(quality 92, không lấy mẫu giảm màu 4:4:4), giữ nguyên độ phân giải 1280×720.
Ảnh BDD100K thuộc giấy phép riêng: xem data/real/pilot_images/LICENSE_BDD100K.md.

Chạy:  .venv\\Scripts\\python.exe scripts\\make_pilot_previews.py [--overwrite]
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data" / "real" / "local" / "end_corrected.zip"
PRELABELS = ROOT / "data" / "real" / "processed" / "pilot100_prelabels.csv"
OUT = ROOT / "data" / "real" / "pilot_images"
SOURCES = {"detection": "yolo_overlays", "segmentation": "sam_overlays"}
QUALITY = 92


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    ids = sorted(pd.read_csv(PRELABELS)["image_id"].unique())
    assert len(ids) == 100, len(ids)
    z = zipfile.ZipFile(ZIP)
    stats = []
    for task, src in SOURCES.items():
        (OUT / task).mkdir(parents=True, exist_ok=True)
        for iid in ids:
            dst = OUT / task / f"{iid}.jpg"
            if dst.exists() and not args.overwrite:
                raise SystemExit(f"Đã tồn tại, không ghi đè: {dst} (dùng --overwrite)")
            img = Image.open(io.BytesIO(z.read(f"{src}/{iid}_overlay.png"))).convert("RGB")
            img.save(dst, "JPEG", quality=QUALITY, subsampling=0, optimize=True)
            back = np.asarray(Image.open(dst).convert("RGB"))
            stats.append({"task": task, "bytes": dst.stat().st_size, "psnr": psnr(np.asarray(img), back),
                          "size": img.size})
    s = pd.DataFrame(stats)
    print(f"Đã ghi {len(s)} ảnh vào {OUT}")
    print(s.groupby("task").agg(n=("bytes", "size"), tong_MB=("bytes", lambda b: b.sum() / 1e6),
                                lon_nhat_KB=("bytes", lambda b: b.max() / 1e3), psnr_min=("psnr", "min"),
                                psnr_tb=("psnr", "mean")).round(2).to_string())
    print("Kích thước ảnh:", sorted(set(s["size"])))


if __name__ == "__main__":
    main()
