"""Bảng đặc trưng mở rộng cho pilot 100 ảnh (mỗi dòng = ảnh × task).

Chỉ dùng thông tin có TRƯỚC khi người sửa bắt đầu: box/confidence YOLO, box prompt + mask PNG của SAM,
kích thước ảnh. Không đọc correction log, ground truth hay overlay pixel.

Nhóm A: đặc trưng định trước (theo yêu cầu: số lượng, confidence, kích thước, chồng lấn, mask).
Nhóm B: giả thuyết gợi ý từ ghi chú người sửa (box chạm đáy ảnh, mask tràn box, mask chồng nhau).
        Vì ý tưởng đến từ chính dữ liệu đánh giá, kết quả nhóm B trên pilot này bị LẠC QUAN.

Chạy:  .venv\\Scripts\\python.exe scripts\\build_features.py [--overwrite]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_real_data import DEFAULT_ZIP, MODELS, read_csv, sha256, image_sizes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "real" / "processed"
FEATURES_NAME = "pilot100_features.csv"
DICT_NAME = "pilot100_features_dictionary.csv"

SMALL_BOX_AREA_PX = 32 * 32
OVERLAP_IOU = 0.3
BOTTOM_MARGIN_PX = 2
CONF_BINS = [(0.0, 0.3, "conf_n_lt030"), (0.3, 0.5, "conf_n_030_050"),
             (0.5, 0.7, "conf_n_050_070"), (0.7, 1.01, "conf_n_ge070")]

YOLO_SRC = "yolo_predictions_100.csv"
SAM_SRC = "sam_predictions_100.csv + sam_masks/*.png"
SIZE_SRC = "kích thước yolo_overlays/*.png"

# name, group, task scope, source, formula
DICTIONARY = [
    ("n_boxes", "A", "cả hai", YOLO_SRC, "Số box car của YOLO (với segmentation = số box prompt = số mask)"),
    ("has_predictions", "A", "cả hai", YOLO_SRC, "1 nếu n_boxes > 0"),
    ("conf_mean", "A", "cả hai", YOLO_SRC, "Trung bình confidence; trống nếu không có box"),
    ("conf_min", "A", "cả hai", YOLO_SRC, "Confidence nhỏ nhất"),
    ("conf_max", "A", "cả hai", YOLO_SRC, "Confidence lớn nhất"),
    ("conf_std", "A", "cả hai", YOLO_SRC, "Độ lệch chuẩn confidence (ddof=0)"),
    ("conf_n_lt030", "A", "cả hai", YOLO_SRC, "Số box confidence < 0.3"),
    ("conf_n_030_050", "A", "cả hai", YOLO_SRC, "Số box 0.3 ≤ confidence < 0.5"),
    ("conf_n_050_070", "A", "cả hai", YOLO_SRC, "Số box 0.5 ≤ confidence < 0.7"),
    ("conf_n_ge070", "A", "cả hai", YOLO_SRC, "Số box confidence ≥ 0.7"),
    ("frac_conf_lt050", "A", "cả hai", YOLO_SRC, "Tỉ lệ box confidence < 0.5"),
    ("box_w_mean", "A", "cả hai", YOLO_SRC, "Chiều rộng box TB (px) = mean(x2 − x1)"),
    ("box_h_mean", "A", "cả hai", YOLO_SRC, "Chiều cao box TB (px) = mean(y2 − y1)"),
    ("box_area_ratio_mean", "A", "cả hai", YOLO_SRC + "; " + SIZE_SRC, "mean((x2−x1)(y2−y1) / (W·H))"),
    ("box_area_ratio_min", "A", "cả hai", YOLO_SRC + "; " + SIZE_SRC, "min diện tích box / diện tích ảnh"),
    ("box_area_ratio_max", "A", "cả hai", YOLO_SRC + "; " + SIZE_SRC, "max diện tích box / diện tích ảnh"),
    ("box_area_ratio_sum", "A", "cả hai", YOLO_SRC + "; " + SIZE_SRC, "tổng diện tích box / diện tích ảnh (có thể > 1 nếu chồng)"),
    ("frac_small_boxes", "A", "cả hai", YOLO_SRC, "Tỉ lệ box có diện tích < 32×32 px"),
    ("max_pair_iou", "A", "cả hai", YOLO_SRC, "IoU lớn nhất giữa hai box bất kỳ trong ảnh (0 nếu < 2 box)"),
    ("n_overlap_pairs", "A", "cả hai", YOLO_SRC, "Số cặp box có IoU > 0.3"),
    ("n_masks", "A", "segmentation", SAM_SRC, "Số dòng SAM (mask) của ảnh"),
    ("frac_empty_masks", "A", "segmentation", SAM_SRC, "Tỉ lệ mask có 0 pixel hoặc thiếu file"),
    ("mask_area_ratio_mean", "A", "segmentation", SAM_SRC, "mean(pixel mask / (W·H))"),
    ("mask_union_coverage", "A", "segmentation", SAM_SRC, "pixel thuộc ít nhất 1 mask / (W·H)"),
    ("mask_fill_in_box_mean", "A", "segmentation", SAM_SRC, "mean(pixel mask nằm trong box prompt / diện tích box)"),
    ("mask_fill_in_box_min", "A", "segmentation", SAM_SRC, "min(pixel mask trong box / diện tích box)"),
    ("n_boxes_touch_bottom", "B", "cả hai", YOLO_SRC + "; " + SIZE_SRC, "Số box có y2 ≥ H − 2 (chạm đáy ảnh)"),
    ("mask_outside_box_frac_mean", "B", "segmentation", SAM_SRC, "mean(pixel mask ngoài box prompt / pixel mask)"),
    ("mask_overlap_frac", "B", "segmentation", SAM_SRC, "pixel thuộc ≥ 2 mask / pixel thuộc ≥ 1 mask"),
]
GROUP_A = [d[0] for d in DICTIONARY if d[1] == "A"]
GROUP_B = [d[0] for d in DICTIONARY if d[1] == "B"]


def _iou_matrix(b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(b[:, None, 0], b[None, :, 0]); y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2]); y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def box_feats(g: pd.DataFrame, W: int, H: int) -> dict:
    n = len(g)
    f = {"n_boxes": n, "has_predictions": int(n > 0)}
    for lo, hi, name in CONF_BINS:
        f[name] = int(((g["confidence"] >= lo) & (g["confidence"] < hi)).sum()) if n else 0
    f["n_overlap_pairs"] = 0
    f["max_pair_iou"] = 0.0
    f["n_boxes_touch_bottom"] = 0
    if n == 0:
        return f
    c = g["confidence"].to_numpy(float)
    b = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    area = w * h
    f.update(
        conf_mean=c.mean(), conf_min=c.min(), conf_max=c.max(), conf_std=c.std(),
        frac_conf_lt050=float((c < 0.5).mean()),
        box_w_mean=w.mean(), box_h_mean=h.mean(),
        box_area_ratio_mean=(area / (W * H)).mean(), box_area_ratio_min=(area / (W * H)).min(),
        box_area_ratio_max=(area / (W * H)).max(), box_area_ratio_sum=area.sum() / (W * H),
        frac_small_boxes=float((area < SMALL_BOX_AREA_PX).mean()),
        n_boxes_touch_bottom=int((b[:, 3] >= H - BOTTOM_MARGIN_PX).sum()),
    )
    if n >= 2:
        iou = _iou_matrix(b)[np.triu_indices(n, 1)]
        f["max_pair_iou"] = float(iou.max())
        f["n_overlap_pairs"] = int((iou > OVERLAP_IOU).sum())
    return f


def mask_feats(z: zipfile.ZipFile, g: pd.DataFrame, W: int, H: int, names: set[str]) -> dict:
    f = {"n_masks": len(g)}
    if len(g) == 0:
        return f
    count = np.zeros((H, W), dtype=np.uint8)
    empty, area_r, fill, outside = [], [], [], []
    for r in g.itertuples():
        if not isinstance(r.mask_path, str) or r.mask_path not in names:
            empty.append(1)
            continue
        m = np.asarray(Image.open(io.BytesIO(z.read(r.mask_path)))) > 0
        total = int(m.sum())
        inside = int(m[r.y1:r.y2, r.x1:r.x2].sum())
        box_area = max((r.x2 - r.x1) * (r.y2 - r.y1), 1)
        empty.append(int(total == 0))
        area_r.append(total / (W * H))
        fill.append(inside / box_area)
        outside.append((total - inside) / total if total else np.nan)
        count += m.astype(np.uint8)
    union = int((count >= 1).sum())
    f.update(
        frac_empty_masks=float(np.mean(empty)),
        mask_area_ratio_mean=float(np.mean(area_r)) if area_r else np.nan,
        mask_union_coverage=union / (W * H),
        mask_fill_in_box_mean=float(np.mean(fill)) if fill else np.nan,
        mask_fill_in_box_min=float(np.min(fill)) if fill else np.nan,
        mask_outside_box_frac_mean=float(np.nanmean(outside)) if outside else np.nan,
        mask_overlap_frac=int((count >= 2).sum()) / union if union else np.nan,
    )
    return f


def build(zip_path: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        sizes = image_sizes(z).set_index("image_id")
        yolo = read_csv(z, "yolo_predictions_100.csv").rename(
            columns={"x_min": "x1", "y_min": "y1", "x_max": "x2", "y_max": "y2"})
        sam = read_csv(z, "sam_predictions_100.csv")
        for image_id, (W, H) in sizes[["image_width", "image_height"]].iterrows():
            yg = yolo[yolo["image_id"] == image_id]
            sg = sam[sam["image_id"] == image_id]
            base = {"image_id": image_id, "image_width": W, "image_height": H}
            rows.append({**base, "task": "detection", "model": MODELS["detection"], **box_feats(yg, W, H)})
            rows.append({**base, "task": "segmentation", "model": MODELS["segmentation"],
                         **box_feats(sg, W, H), **mask_feats(z, sg, W, H, names)})
    df = pd.DataFrame(rows)
    cols = ["image_id", "task", "model", "image_width", "image_height"] + GROUP_A + GROUP_B
    df = df.reindex(columns=cols).sort_values(["task", "image_id"]).reset_index(drop=True)
    return df.round(6)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    targets = [args.out / FEATURES_NAME, args.out / DICT_NAME]
    if any(t.exists() for t in targets) and not args.overwrite:
        raise SystemExit(f"Đã tồn tại, không ghi đè: {[str(t) for t in targets if t.exists()]}")
    df = build(args.zip)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(targets[0], index=False, encoding="utf-8")
    pd.DataFrame(DICTIONARY, columns=["feature", "group", "task", "source", "formula"]).assign(
        missing_detection=lambda d: [int(df.loc[df.task == "detection", c].isna().sum()) for c in d.feature],
        missing_segmentation=lambda d: [int(df.loc[df.task == "segmentation", c].isna().sum()) for c in d.feature],
    ).to_csv(targets[1], index=False, encoding="utf-8")
    print(json.dumps({"rows": len(df), "source_sha256": sha256(args.zip)[:16], "written": [str(t) for t in targets]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
