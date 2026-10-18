"""Pilot so sánh phiên bản model trên CÙNG 100 ảnh BDD100K của M31.

- Detection: YOLO A vs YOLO B (mặc định yolov8n vs yolo11n), lọc lớp `car` như pipeline gốc.
- Segmentation: SAM A vs SAM B (mặc định sam_b vs sam2_b), CÙNG box prompt = 555 box YOLOv8n đã bàn giao
  trong pilot (sam_predictions_100.csv), để khác biệt chỉ đến từ SAM.
- Không so điểm YOLO với điểm SAM.

Đo thời gian: nạp model đo riêng (KHÔNG tính vào thời gian ảnh); chạy `--warmup` ảnh khởi động không tính;
mỗi ảnh đo wall-clock quanh lời gọi model trên ảnh đã giải mã sẵn trong RAM, có torch.cuda.synchronize.
Không tính thời gian đọc/giải mã ảnh và hậu xử lý của script.

Chất lượng (chỉ khi có nhãn tham chiếu phù hợp):
- Segmentation: nhãn BDD100K 10k seg `*_train_id.png` (car = 13, 255 = bỏ qua) cho cả 100 ảnh -> IoU/Dice
  của hợp các mask so với vùng car (theo ảnh, KHÔNG theo đối tượng vì nhãn là semantic).
- Detection: chỉ ảnh có nhãn box `car` trong bdd100k_labels_images_train.json (pilot: 14/100) -> precision/recall
  @IoU 0.5 trên TẬP CON đó. Trên cả 100 ảnh chỉ báo thống kê mô tả và mức trùng khớp giữa hai YOLO.
- Kiểm tra tái lập: so YOLO A / SAM A chạy lại với đầu ra đã bàn giao trong pilot.

Chạy (bằng .venv_models có ultralytics + torch):
  .venv_models\\Scripts\\python.exe scripts\\compare_model_versions.py --archive <BDD100K archive.zip> ^
      --yolo-a <yolov8n.pt> --yolo-b <yolo11n.pt> --sam-a <sam_b.pt> --sam-b <sam2_b.pt>
Ảnh và nhãn được đọc thẳng từ ZIP (chỉ đọc); kết quả ghi vào data/real/model_comparison/.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PILOT_ZIP = ROOT / "data" / "real" / "local" / "end_corrected.zip"
DEFAULT_OUT = ROOT / "data" / "real" / "model_comparison"
DET_LABELS_IN_ARCHIVE = "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
CAR_TRAIN_ID, IGNORE_ID = 13, 255
MATCH_IOU = 0.5


# ------------------------------------------------------------------ dữ liệu
def read_csv_from_zip(z: zipfile.ZipFile, name: str) -> list[dict]:
    return list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8-sig")))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def decode_image(data: bytes) -> np.ndarray:
    import cv2
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def decode_label(data: bytes) -> np.ndarray:
    import cv2
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def extract_det_labels(archive: zipfile.ZipFile, ids: list[str]) -> dict[str, list[list[float]]]:
    """Lấy box `car` cho các image_id có trong nhãn detection (đọc toàn file JSON vào RAM rồi cắt đúng đối tượng)."""
    raw = archive.read(DET_LABELS_IN_ARCHIVE)
    out = {}
    for iid in ids:
        key = f'"{iid}.jpg"'.encode()
        pos = raw.find(key)
        if pos < 0:
            continue
        start = raw.rfind(b"{", 0, pos)
        depth, i, in_str, esc = 0, start, False, False
        while i < len(raw):
            c = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif c == 0x5C:
                    esc = True
                elif c == 0x22:
                    in_str = False
            elif c == 0x22:
                in_str = True
            elif c == 0x7B:
                depth += 1
            elif c == 0x7D:
                depth -= 1
                if depth == 0:
                    break
            i += 1
        obj = json.loads(raw[start:i + 1])
        if obj.get("name") != f"{iid}.jpg":
            continue
        out[iid] = [[l["box2d"][k] for k in ("x1", "y1", "x2", "y2")]
                    for l in obj.get("labels", []) if l.get("category") == "car" and "box2d" in l]
    return out


# ------------------------------------------------------------------ chỉ số
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(pred: np.ndarray, conf: np.ndarray, ref: np.ndarray, thr: float = MATCH_IOU) -> tuple[int, list[float]]:
    """Ghép tham lam theo confidence giảm dần; trả về (số cặp, danh sách IoU của cặp)."""
    m = iou_matrix(pred, ref)
    used, ious = set(), []
    if m.shape[1] == 0:
        return 0, ious
    for i in np.argsort(-conf):
        row = m[i].copy()
        if used:
            row[list(used)] = -1.0
        j = int(np.argmax(row))
        if row[j] >= thr:
            used.add(j)
            ious.append(float(row[j]))
    return len(ious), ious


def mask_scores(pred: np.ndarray, gt_label: np.ndarray, region: np.ndarray | None = None) -> dict:
    valid = gt_label != IGNORE_ID
    if region is not None:
        valid &= region
    g = (gt_label == CAR_TRAIN_ID) & valid
    p = pred & valid
    inter, union = int((p & g).sum()), int((p | g).sum())
    s = int(p.sum()) + int(g.sum())
    return {"inter": inter, "union": union, "pred_px": int(p.sum()), "gt_px": int(g.sum()),
            "iou": inter / union if union else float("nan"), "dice": 2 * inter / s if s else float("nan")}


# ------------------------------------------------------------------ chạy model
def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    t0 = time.perf_counter()
    r = fn()
    sync()
    return r, time.perf_counter() - t0


def run_yolo(weights: str, images: dict, ids: list[str], device: str, warmup: int, imgsz: int, conf: float):
    from ultralytics import YOLO
    model, load_s = timed(lambda: YOLO(weights))
    car = [k for k, v in model.names.items() if v == "car"]
    for iid in ids[:warmup]:
        model(images[iid], device=device, verbose=False, imgsz=imgsz, conf=conf)
    per_image, boxes = [], []
    for iid in ids:
        try:
            res, sec = timed(lambda: model(images[iid], device=device, verbose=False, imgsz=imgsz, conf=conf)[0])
            b = res.boxes
            keep = [i for i in range(len(b)) if int(b.cls[i]) in car]
            xyxy = b.xyxy.cpu().numpy()[keep] if keep else np.zeros((0, 4))
            cf = b.conf.cpu().numpy()[keep] if keep else np.zeros(0)
            per_image.append({"image_id": iid, "status": "ok", "inference_sec": sec,
                              "ultra_inference_ms": res.speed.get("inference"), "n_car_boxes": len(keep),
                              "mean_conf": float(cf.mean()) if len(cf) else ""})
            boxes += [{"image_id": iid, "x1": x[0], "y1": x[1], "x2": x[2], "y2": x[3], "confidence": c}
                      for x, c in zip(xyxy.tolist(), cf.tolist())]
        except Exception as exc:  # ghi nhận lỗi, không dừng cả lượt
            per_image.append({"image_id": iid, "status": f"error: {type(exc).__name__}: {exc}"[:200]})
    return load_s, per_image, boxes


def run_sam(weights: str, images: dict, ids: list[str], prompts: dict, device: str, warmup: int):
    from ultralytics import SAM
    model, load_s = timed(lambda: SAM(weights))
    warm = [i for i in ids if prompts.get(i)][:warmup]
    for iid in warm:
        model(images[iid], bboxes=prompts[iid], device=device, verbose=False)
    per_image, masks = [], {}
    for iid in ids:
        h, w = images[iid].shape[:2]
        if not prompts.get(iid):
            masks[iid] = np.zeros((h, w), bool)
            per_image.append({"image_id": iid, "status": "no_prompt", "n_prompts": 0})
            continue
        try:
            res, sec = timed(lambda: model(images[iid], bboxes=prompts[iid], device=device, verbose=False)[0])
            m = res.masks.data.cpu().numpy() > 0 if res.masks is not None else np.zeros((0, h, w), bool)
            if m.shape[1:] != (h, w):
                import cv2
                m = np.stack([cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for x in m]) \
                    if len(m) else np.zeros((0, h, w), bool)
            masks[iid] = m.any(0) if len(m) else np.zeros((h, w), bool)
            per_image.append({"image_id": iid, "status": "ok", "inference_sec": sec,
                              "ultra_inference_ms": res.speed.get("inference"), "n_prompts": len(prompts[iid]),
                              "n_masks": int(len(m)), "n_empty_masks": int(sum(int(x.sum() == 0) for x in m))})
        except Exception as exc:
            masks[iid] = None
            per_image.append({"image_id": iid, "status": f"error: {type(exc).__name__}: {exc}"[:200]})
    return load_s, per_image, masks


def timing_summary(rows: list[dict]) -> dict:
    t = np.array([r["inference_sec"] for r in rows if r.get("status") == "ok"], float)
    if not len(t):
        return {"n_ok": 0}
    return {"n_ok": int(len(t)), "total_sec": float(t.sum()), "mean_sec": float(t.mean()),
            "median_sec": float(np.median(t)), "p90_sec": float(np.quantile(t, 0.9)), "max_sec": float(t.max())}


# ------------------------------------------------------------------ main
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="BDD100K archive.zip (ảnh 10k seg + nhãn)")
    ap.add_argument("--pilot-zip", default=str(DEFAULT_PILOT_ZIP))
    ap.add_argument("--yolo-a", required=True); ap.add_argument("--yolo-b", required=True)
    ap.add_argument("--sam-a", required=True); ap.add_argument("--sam-b", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu (thử nhanh); 0 = cả 100")
    args = ap.parse_args()
    out = Path(args.out)

    import cv2
    import torch
    import ultralytics

    pilot = zipfile.ZipFile(args.pilot_zip)
    pairs = read_csv_from_zip(pilot, "image_groundtruth_pairs_100.csv")
    ids = [r["image_id"] for r in pairs][: args.limit or None]
    delivered_det = read_csv_from_zip(pilot, "yolo_predictions_100.csv")
    delivered_sam = read_csv_from_zip(pilot, "sam_predictions_100.csv")
    prompts: dict[str, list[list[int]]] = {}
    for r in delivered_sam:
        prompts.setdefault(r["image_id"], []).append([int(r[k]) for k in ("x1", "y1", "x2", "y2")])

    archive = zipfile.ZipFile(args.archive)
    images = {r["image_id"]: decode_image(archive.read(r["image_path"])) for r in pairs if r["image_id"] in ids}
    labels = {r["image_id"]: decode_label(archive.read(r["gt_id_label_path"])) for r in pairs if r["image_id"] in ids}
    det_gt = extract_det_labels(archive, ids)
    print(f"Ảnh: {len(images)} · nhãn seg: {len(labels)} · ảnh có nhãn box car: {len(det_gt)}")

    names = {"yolo_a": Path(args.yolo_a).stem, "yolo_b": Path(args.yolo_b).stem,
             "sam_a": Path(args.sam_a).stem, "sam_b": Path(args.sam_b).stem}
    rows, summary = [], {"detection": {}, "segmentation": {}}

    # ---- Detection
    det_boxes = {}
    for key in ("yolo_a", "yolo_b"):
        load_s, per, boxes = run_yolo(getattr(args, key), images, ids, args.device, args.warmup, args.imgsz, args.conf)
        det_boxes[key] = boxes
        write_csv(out / f"detections_{names[key]}.csv", boxes)
        summary["detection"][names[key]] = {"weights_file": Path(getattr(args, key)).name, "load_sec": load_s,
                                            **timing_summary(per), "n_failed": sum(r["status"].startswith("error") for r in per),
                                            "total_car_boxes": len(boxes)}
        for r in per:
            rows.append({"model": names[key], "task": "detection", **r})
        print(f"{names[key]}: {summary['detection'][names[key]]}")

    def by_img(boxes):
        d = {}
        for b in boxes:
            d.setdefault(b["image_id"], []).append(b)
        arr = lambda L: (np.array([[x["x1"], x["y1"], x["x2"], x["y2"]] for x in L], float).reshape(-1, 4),  # noqa: E731
                         np.array([x["confidence"] for x in L], float))
        return {k: arr(v) for k, v in d.items()}

    A, B = by_img(det_boxes["yolo_a"]), by_img(det_boxes["yolo_b"])
    empty = (np.zeros((0, 4)), np.zeros(0))
    delivered = by_img([{**r, "x1": float(r["x_min"]), "y1": float(r["y_min"]), "x2": float(r["x_max"]),
                         "y2": float(r["y_max"]), "confidence": float(r["confidence"])} for r in delivered_det])
    agree = {"matched": 0, "only_a": 0, "only_b": 0, "repro_matched": 0, "repro_delivered": 0, "repro_rerun": 0}
    for iid in ids:
        a, b = A.get(iid, empty), B.get(iid, empty)
        n, _ = greedy_match(a[0], a[1], b[0])
        agree["matched"] += n; agree["only_a"] += len(a[0]) - n; agree["only_b"] += len(b[0]) - n
        d = delivered.get(iid, empty)
        rn, _ = greedy_match(a[0], a[1], d[0], thr=0.9)
        agree["repro_matched"] += rn; agree["repro_delivered"] += len(d[0]); agree["repro_rerun"] += len(a[0])
    summary["detection"]["agreement_iou0.5"] = {k: agree[k] for k in ("matched", "only_a", "only_b")}
    summary["detection"]["reproducibility_vs_delivered_yolo_a_iou0.9"] = {
        "matched": agree["repro_matched"], "delivered_boxes": agree["repro_delivered"], "rerun_boxes": agree["repro_rerun"]}

    gt_rows = []
    for key, P in (("yolo_a", A), ("yolo_b", B)):
        tp = fp = fn = 0
        ious = []
        for iid, g in det_gt.items():
            p = P.get(iid, empty)
            gb = np.array(g, float).reshape(-1, 4)
            n, mi = greedy_match(p[0], p[1], gb)
            tp += n; fp += len(p[0]) - n; fn += len(gb) - n; ious += mi
            gt_rows.append({"model": names[key], "image_id": iid, "gt_car_boxes": len(gb), "pred_car_boxes": len(p[0]),
                            "tp": n, "fp": len(p[0]) - n, "fn": len(gb) - n})
        summary["detection"][f"gt_subset_{names[key]}"] = {
            "n_images": len(det_gt), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "mean_iou_matched": float(np.mean(ious)) if ious else None}
    write_csv(out / "detection_gt_subset.csv", gt_rows)

    # ---- Segmentation
    delivered_union = {}
    for iid in ids:
        name = f"sam_masks/{iid}_car_sam.png"
        delivered_union[iid] = decode_label(pilot.read(name)) > 0 if name in pilot.namelist() else None
    for key in ("sam_a", "sam_b"):
        load_s, per, masks = run_sam(getattr(args, key), images, ids, prompts, args.device, args.warmup)
        tot = {"all": [0, 0, 0, 0], "box": [0, 0, 0, 0]}
        for r in per:
            iid = r["image_id"]
            if masks.get(iid) is None:
                continue
            h, w = labels[iid].shape
            region = np.zeros((h, w), bool)
            for x1, y1, x2, y2 in prompts.get(iid, []):
                region[y1:y2, x1:x2] = True
            s_all = mask_scores(masks[iid], labels[iid])
            s_box = mask_scores(masks[iid], labels[iid], region)
            r.update({"iou": s_all["iou"], "dice": s_all["dice"], "gt_car_px": s_all["gt_px"], "pred_px": s_all["pred_px"],
                      "iou_in_prompt_boxes": s_box["iou"], "dice_in_prompt_boxes": s_box["dice"]})
            for k, s in (("all", s_all), ("box", s_box)):
                tot[k][0] += s["inter"]; tot[k][1] += s["union"]; tot[k][2] += s["pred_px"]; tot[k][3] += s["gt_px"]
            if key == "sam_a" and delivered_union.get(iid) is not None:
                d, m = delivered_union[iid], masks[iid]
                u = int((d | m).sum())
                r["agreement_iou_vs_delivered"] = int((d & m).sum()) / u if u else 1.0
        ok = [r for r in per if r.get("status") in ("ok", "no_prompt") and "iou" in r]
        iou_vals = [r["iou"] for r in ok if r["iou"] == r["iou"]]
        summary["segmentation"][names[key]] = {
            "weights_file": Path(getattr(args, key)).name, "load_sec": load_s, **timing_summary(per),
            "n_failed": sum(str(r["status"]).startswith("error") for r in per),
            "n_no_prompt": sum(r["status"] == "no_prompt" for r in per),
            "mean_iou_per_image": float(np.mean(iou_vals)) if iou_vals else None,
            "n_images_iou_defined": len(iou_vals),
            "pooled_iou": tot["all"][0] / tot["all"][1] if tot["all"][1] else None,
            "pooled_dice": 2 * tot["all"][0] / (tot["all"][2] + tot["all"][3]) if tot["all"][2] + tot["all"][3] else None,
            "pooled_iou_in_prompt_boxes": tot["box"][0] / tot["box"][1] if tot["box"][1] else None,
            "pooled_dice_in_prompt_boxes": 2 * tot["box"][0] / (tot["box"][2] + tot["box"][3])
            if tot["box"][2] + tot["box"][3] else None,
        }
        if key == "sam_a":
            ag = [r["agreement_iou_vs_delivered"] for r in per if "agreement_iou_vs_delivered" in r]
            summary["segmentation"]["reproducibility_vs_delivered_sam_a_mean_iou"] = float(np.mean(ag)) if ag else None
        for r in per:
            rows.append({"model": names[key], "task": "segmentation", **r})
        print(f"{names[key]}: {summary['segmentation'][names[key]]}")

    config = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "n_images_requested": len(ids),
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "cpu": platform.processor(), "os": platform.platform()},
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__, "opencv": cv2.__version__},
        "settings": {"device": args.device, "imgsz_yolo": args.imgsz, "conf_yolo": args.conf, "warmup_images": args.warmup,
                     "yolo_class_filter": "car (lọc sau khi suy luận, giống pipeline gốc)",
                     "sam_prompts": "555 box YOLOv8n đã bàn giao (sam_predictions_100.csv), giống nhau cho cả hai SAM",
                     "timing": "wall-clock quanh lời gọi model trên ảnh đã giải mã trong RAM, có cuda synchronize; "
                               "không tính nạp model, ảnh khởi động, đọc/giải mã ảnh, tính chỉ số",
                     "seg_reference": "BDD100K 10k seg *_train_id.png, car = 13, 255 bỏ qua; IoU theo ảnh (semantic)",
                     "det_reference": "bdd100k_labels_images_train.json, category car, chỉ ảnh có nhãn; ghép IoU >= 0.5"},
        "weights": {k: Path(getattr(args, k)).name for k in ("yolo_a", "yolo_b", "sam_a", "sam_b")},
    }
    write_csv(out / "per_image_results.csv", rows)
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nĐã ghi kết quả vào {out}")


if __name__ == "__main__":
    os.environ.setdefault("YOLO_OFFLINE", "1")
    main()
