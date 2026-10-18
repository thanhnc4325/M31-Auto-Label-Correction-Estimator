"""Chuẩn bị dữ liệu PILOT THẬT (100 ảnh BDD100K, lớp car) cho M31.

Đọc bản sao ZIP local (không giải nén ảnh ra đĩa), ghi CSV đã xử lý vào data/real/processed/.

Nguyên tắc:
- Đặc trưng pre-label chỉ tính từ file dự đoán YOLO/SAM và mask SAM trong ZIP.
- Log sửa (thời gian, add/delete/modify, status, notes) chỉ được chuyển sang file
  correction log riêng để ĐÁNH GIÁ, không trộn vào file đặc trưng.
- Không đọc ground truth. Không tạo thêm nhãn, thời gian sửa hay confidence.
- Không ghi đè file đã xử lý trừ khi truyền --overwrite.

Chạy:  .venv\\Scripts\\python.exe scripts\\prepare_real_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = ROOT / "data" / "real" / "local" / "end_corrected.zip"
DEFAULT_OUT = ROOT / "data" / "real" / "processed"
PRELABELS_NAME = "pilot100_prelabels.csv"
LOG_NAME = "pilot100_correction_log.csv"
MANIFEST_NAME = "pilot100_manifest.json"

# Ngưỡng do M31 chọn (không được bàn giao kèm dữ liệu). Chọn trước khi xem tương quan với thời gian sửa.
LOW_CONF_THRESHOLD = 0.5
SMALL_BOX_AREA_PX = 32 * 32  # định nghĩa "small object" của COCO

MODELS = {"detection": "yolov8n", "segmentation": "sam_b"}  # theo correction log và README_end.md


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(z.read(name)), encoding="utf-8-sig")


def image_sizes(z: zipfile.ZipFile) -> pd.DataFrame:
    """Danh sách ảnh và kích thước, lấy từ overlay pre-label YOLO (một overlay cho mỗi ảnh)."""
    rows = []
    for n in z.namelist():
        if n.startswith("yolo_overlays/") and n.endswith("_overlay.png"):
            w, h = Image.open(io.BytesIO(z.read(n))).size
            rows.append({"image_id": n.split("/")[-1][: -len("_overlay.png")], "image_width": w, "image_height": h})
    return pd.DataFrame(rows)


def box_features(boxes: pd.DataFrame, sizes: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng theo ảnh từ các box: cột x1,y1,x2,y2,confidence."""
    b = boxes.merge(sizes, on="image_id", how="left")
    area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    b = b.assign(
        low=(b["confidence"] < LOW_CONF_THRESHOLD).astype(int),
        small=(area < SMALL_BOX_AREA_PX).astype(int),
        area_ratio=area / (b["image_width"] * b["image_height"]),
    )
    return b.groupby("image_id").agg(
        num_predictions=("confidence", "size"),
        mean_confidence=("confidence", "mean"),
        min_confidence=("confidence", "min"),
        low_confidence_count=("low", "sum"),
        small_box_count=("small", "sum"),
        mean_box_area_ratio=("area_ratio", "mean"),
    ).reset_index()


def mask_features(z: zipfile.ZipFile, sam: pd.DataFrame) -> pd.DataFrame:
    """Tỉ lệ lấp đầy mask trong box prompt và số mask rỗng/thiếu, tính từ PNG mask."""
    names = set(z.namelist())
    fills, empties = [], []
    for r in sam.itertuples():
        box_area = max((r.x2 - r.x1) * (r.y2 - r.y1), 1)
        if not isinstance(r.mask_path, str) or r.mask_path not in names:
            fills.append(np.nan)
            empties.append(1)
            continue
        m = np.asarray(Image.open(io.BytesIO(z.read(r.mask_path)))) > 0
        area = int(m.sum())
        fills.append(area / box_area)
        empties.append(int(area == 0))
    s = sam.assign(fill=fills, empty=empties)
    return s.groupby("image_id").agg(
        mean_mask_fill_ratio=("fill", "mean"), empty_mask_count=("empty", "sum")
    ).reset_index()


def build(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    with zipfile.ZipFile(zip_path) as z:
        sizes = image_sizes(z)
        yolo = read_csv(z, "yolo_predictions_100.csv").rename(
            columns={"x_min": "x1", "y_min": "y1", "x_max": "x2", "y_max": "y2"}
        )
        sam = read_csv(z, "sam_predictions_100.csv")
        logs = pd.concat([read_csv(z, "correction_log_yolo_100.csv"), read_csv(z, "correction_log_sam_100.csv")])
        det = box_features(yolo, sizes)
        seg = box_features(sam, sizes).merge(mask_features(z, sam), on="image_id", how="left")

    frames = []
    for task, feats, conf_src in [("detection", det, "yolov8n"), ("segmentation", seg, "yolo_box_prompt")]:
        f = sizes.merge(feats, on="image_id", how="left")  # ảnh không có dự đoán -> NaN
        for c in ["num_predictions", "low_confidence_count", "small_box_count"] + (
            ["empty_mask_count"] if task == "segmentation" else []
        ):
            f[c] = f[c].fillna(0).astype(int)
        f.insert(1, "task", task)
        f.insert(2, "model", MODELS[task])
        f["confidence_source"] = conf_src
        frames.append(f)
    prelabels = pd.concat(frames, ignore_index=True)
    order = ["image_id", "task", "model", "mean_confidence", "num_predictions", "low_confidence_count",
             "image_width", "image_height", "min_confidence", "small_box_count", "mean_box_area_ratio",
             "mean_mask_fill_ratio", "empty_mask_count", "confidence_source"]
    prelabels = prelabels[order].sort_values(["task", "image_id"]).reset_index(drop=True)
    for c in ["mean_confidence", "min_confidence"]:
        prelabels[c] = prelabels[c].round(4)
    for c in ["mean_box_area_ratio", "mean_mask_fill_ratio"]:
        prelabels[c] = prelabels[c].round(5)

    log = pd.DataFrame({
        "image_id": logs["image_id"],
        "task": logs["task_type"],
        "model": logs["model_name"],
        "correction_time_sec": logs["human_correction_time_sec"],
        "edits_added": logs["action_add"],
        "edits_removed": logs["action_delete"],
        "edits_changed": logs["action_modify"],
        "status": logs["status"],
        "notes": logs["notes"],
    }).sort_values(["task", "image_id"]).reset_index(drop=True)

    # Kiểm tra nhất quán giữa số dự đoán đếm được và số dự đoán khai trong log.
    chk = prelabels.merge(
        logs.rename(columns={"task_type": "task", "num_predictions": "log_num_predictions"})[
            ["image_id", "task", "log_num_predictions"]
        ], on=["image_id", "task"], how="outer",
    )
    mismatch = int((chk["num_predictions"] != chk["log_num_predictions"]).sum())

    manifest = {
        "description": "PILOT THẬT: 100 ảnh BDD100K, lớp car, YOLOv8n + SAM-B (box prompt từ YOLO).",
        "source_zip": zip_path.name,
        "source_sha256": sha256(zip_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "low_confidence_threshold": LOW_CONF_THRESHOLD,
        "small_box_area_px": SMALL_BOX_AREA_PX,
        "n_images": int(sizes["image_id"].nunique()),
        "n_prelabel_rows": len(prelabels),
        "n_log_rows": len(log),
        "num_predictions_mismatch_vs_log": mismatch,
        "notes": [
            "SAM không có confidence riêng: cột confidence của SAM là confidence của box YOLO dùng làm prompt.",
            "File pre-label không chứa cột nào từ correction log hay ground truth.",
            "Correction log chỉ dùng để đánh giá/hiệu chỉnh, không làm đầu vào dự đoán.",
            "Ground truth (image_groundtruth_pairs_100.csv) không được đọc.",
        ],
    }
    return prelabels, log, manifest


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not args.zip.is_file():
        raise SystemExit(f"Không tìm thấy ZIP: {args.zip}")
    targets = [args.out / n for n in (PRELABELS_NAME, LOG_NAME, MANIFEST_NAME)]
    existing = [t for t in targets if t.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"Đã tồn tại, không ghi đè: {[str(t) for t in existing]}. Dùng --overwrite nếu chắc chắn.")

    prelabels, log, manifest = build(args.zip)
    args.out.mkdir(parents=True, exist_ok=True)
    prelabels.to_csv(targets[0], index=False, encoding="utf-8")
    log.to_csv(targets[1], index=False, encoding="utf-8")
    targets[2].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for t in targets:
        print(f"Đã ghi {t}")
    print(f"Lệch số dự đoán so với log: {manifest['num_predictions_mismatch_vs_log']}")


if __name__ == "__main__":
    main()
