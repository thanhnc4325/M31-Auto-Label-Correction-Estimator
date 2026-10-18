"""So sánh NHIỀU model YOLO (detection) và SAM (segmentation có box prompt) trên cùng 100 ảnh pilot M31.

Dùng lại các hàm đã kiểm thử trong compare_model_versions.py. Mỗi lần chạy ghi vào thư mục riêng
data/real/model_comparison/runs/<YYYYmmdd-HHMMSS>/ (không ghi đè kết quả cũ).

- YOLO: cùng imgsz/conf, lọc lớp `car` sau suy luận; chất lượng chỉ trên ảnh có nhãn box car (pilot: 14 ảnh),
  cộng với mức trùng khớp so với model tham chiếu (mặc định YOLO đầu tiên = yolov8n) trên cả 100 ảnh.
- SAM: CÙNG 555 box prompt YOLOv8n đã bàn giao cho mọi model ("SAM có prompt"); IoU/Dice so với nhãn
  seg BDD100K (car = 13, 255 bỏ qua) cho cả 100 ảnh.
- Thời gian: như compare_model_versions (không tính nạp model, ảnh khởi động, giải mã ảnh, tính chỉ số).
- Khoảng tin cậy 95%: bootstrap theo ảnh (2000 lần, seed 42) cho chỉ số của từng model và chênh lệch so với tham chiếu.
- Model không nạp/chạy được: ghi lý do, chuyển sang model tiếp theo.

Chạy:
  .venv_models\\Scripts\\python.exe scripts\\compare_model_zoo.py --archive <archive.zip> ^
      --yolo <yolov8n.pt> <yolo11n.pt> ... --sam <sam_b.pt> <sam2_b.pt> ...
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_model_versions as cmv  # noqa: E402

N_BOOT, SEED = 2000, 42


def boot(stat, n: int, rng) -> list[float]:
    v = np.array([stat(rng.integers(0, n, n)) for _ in range(N_BOOT)], float)
    v = v[~np.isnan(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [None, None]


def ratio(num: np.ndarray, den: np.ndarray):
    return lambda i: num[i].sum() / den[i].sum() if den[i].sum() > 0 else np.nan


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--pilot-zip", default=str(cmv.DEFAULT_PILOT_ZIP))
    ap.add_argument("--yolo", nargs="*", default=[], help="weights YOLO; model đầu tiên là tham chiếu")
    ap.add_argument("--sam", nargs="*", default=[], help="weights SAM; model đầu tiên là tham chiếu")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out-root", default=str(cmv.DEFAULT_OUT / "runs"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import cv2
    import torch
    import ultralytics

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out_root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    pilot = zipfile.ZipFile(args.pilot_zip)
    pairs = cmv.read_csv_from_zip(pilot, "image_groundtruth_pairs_100.csv")
    ids = [r["image_id"] for r in pairs][: args.limit or None]
    prompts: dict[str, list[list[int]]] = {}
    for r in cmv.read_csv_from_zip(pilot, "sam_predictions_100.csv"):
        prompts.setdefault(r["image_id"], []).append([int(r[k]) for k in ("x1", "y1", "x2", "y2")])
    archive = zipfile.ZipFile(args.archive)
    images = {r["image_id"]: cmv.decode_image(archive.read(r["image_path"])) for r in pairs if r["image_id"] in ids}
    labels = {r["image_id"]: cmv.decode_label(archive.read(r["gt_id_label_path"])) for r in pairs if r["image_id"] in ids}
    det_gt = cmv.extract_det_labels(archive, ids)
    gt_ids = sorted(det_gt)
    print(f"[{run_id}] ảnh {len(images)} · nhãn seg {len(labels)} · ảnh có nhãn box car {len(gt_ids)}")

    per_image, inventory, det_rows, seg_rows = [], [], [], []
    empty = (np.zeros((0, 4)), np.zeros(0))

    # ------------------------------------------------------------------ YOLO
    ref_boxes, ref_name = None, None
    ref_time = None
    for order, w in enumerate(args.yolo):
        name = Path(w).stem
        rec = {"task": "detection", "model": name, "weights_file": Path(w).name, "run_order": order}
        try:
            load_s, per, boxes = cmv.run_yolo(w, images, ids, args.device, args.warmup, args.imgsz, args.conf)
        except Exception as exc:
            rec.update(status="không chạy được", error=f"{type(exc).__name__}: {exc}"[:300], n_images_ok=0)
            inventory.append(rec)
            print(f"  {name}: LỖI {rec['error']}")
            traceback.print_exc()
            continue
        cmv.write_csv(out / f"detections_{name}.csv", boxes)
        n_ok = sum(r["status"] == "ok" for r in per)
        rec.update(status="chạy đủ" if n_ok == len(ids) else f"chạy {n_ok}/{len(ids)}", n_images_ok=n_ok,
                   n_failed=len(ids) - n_ok, load_sec=load_s, **cmv.timing_summary(per))
        inventory.append(rec)
        for r in per:
            per_image.append({"model": name, "task": "detection", **r})
        B = {}
        for b in boxes:
            B.setdefault(b["image_id"], []).append(b)
        B = {k: (np.array([[x["x1"], x["y1"], x["x2"], x["y2"]] for x in v], float), np.array([x["confidence"] for x in v]))
             for k, v in B.items()}
        t = {r["image_id"]: r["inference_sec"] for r in per if r["status"] == "ok"}
        # Chất lượng trên ảnh có nhãn
        tp, fp, fn = (np.zeros(len(gt_ids)) for _ in range(3))
        for k, iid in enumerate(gt_ids):
            p = B.get(iid, empty)
            g = np.array(det_gt[iid], float).reshape(-1, 4)
            n, _ = cmv.greedy_match(p[0], p[1], g)
            tp[k], fp[k], fn[k] = n, len(p[0]) - n, len(g) - n
        row = {"model": name, "n_images_ok": n_ok, "total_car_boxes": len(boxes),
               "mean_sec": rec.get("mean_sec"), "total_sec": rec.get("total_sec"), "load_sec": load_s,
               "gt_images": len(gt_ids), "gt_boxes": int(tp.sum() + fn.sum()), "tp": int(tp.sum()), "fp": int(fp.sum()),
               "fn": int(fn.sum()),
               "precision": float(tp.sum() / (tp.sum() + fp.sum())) if tp.sum() + fp.sum() else None,
               "precision_ci95": boot(ratio(tp, tp + fp), len(gt_ids), rng),
               "recall": float(tp.sum() / (tp.sum() + fn.sum())) if tp.sum() + fn.sum() else None,
               "recall_ci95": boot(ratio(tp, tp + fn), len(gt_ids), rng)}
        if ref_boxes is None:
            ref_boxes, ref_name, ref_time, ref_tp = B, name, t, (tp, fp, fn)
        else:
            m = oa = ob = 0
            for iid in ids:
                a, b = ref_boxes.get(iid, empty), B.get(iid, empty)
                n, _ = cmv.greedy_match(a[0], a[1], b[0])
                m += n; oa += len(a[0]) - n; ob += len(b[0]) - n
            common = [i for i in ids if i in t and i in ref_time]
            d = np.array([t[i] - ref_time[i] for i in common])
            rtp, rfp, rfn = ref_tp
            row.update({
                "reference": ref_name, "boxes_matched_with_ref_iou0.5": m, "only_ref": oa, "only_model": ob,
                "time_diff_vs_ref_ms": float(d.mean() * 1000), "time_diff_ci95_ms": [x * 1000 for x in boot(lambda i: d[i].mean(), len(d), rng)],
                "n_images_faster_than_ref": int((d < 0).sum()),
                "precision_diff_vs_ref_ci95": boot(lambda i: ratio(tp, tp + fp)(i) - ratio(rtp, rtp + rfp)(i), len(gt_ids), rng),
                "recall_diff_vs_ref_ci95": boot(lambda i: ratio(tp, tp + fn)(i) - ratio(rtp, rtp + rfn)(i), len(gt_ids), rng)})
        det_rows.append(row)
        print(f"  {name}: {n_ok} ảnh · {rec.get('mean_sec', 0) * 1000:.1f} ms/ảnh · P={row['precision']:.3f} R={row['recall']:.3f}")
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ SAM
    ref_seg = None
    for order, w in enumerate(args.sam):
        name = Path(w).stem
        rec = {"task": "segmentation", "model": name, "weights_file": Path(w).name, "run_order": order}
        try:
            load_s, per, masks = cmv.run_sam(w, images, ids, prompts, args.device, args.warmup)
        except Exception as exc:
            rec.update(status="không chạy được", error=f"{type(exc).__name__}: {exc}"[:300], n_images_ok=0)
            inventory.append(rec)
            print(f"  {name}: LỖI {rec['error']}")
            traceback.print_exc()
            continue
        n_ok = sum(r["status"] == "ok" for r in per)
        n_prompt = sum(1 for i in ids if prompts.get(i))
        I, U, P, G, I_b, U_b = (np.zeros(len(ids)) for _ in range(6))
        t = {}
        for k, r in enumerate(per):
            iid = r["image_id"]
            if r["status"] == "ok":
                t[iid] = r["inference_sec"]
            if masks.get(iid) is None:
                continue
            region = np.zeros(labels[iid].shape, bool)
            for x1, y1, x2, y2 in prompts.get(iid, []):
                region[y1:y2, x1:x2] = True
            s, sb = cmv.mask_scores(masks[iid], labels[iid]), cmv.mask_scores(masks[iid], labels[iid], region)
            I[k], U[k], P[k], G[k], I_b[k], U_b[k] = s["inter"], s["union"], s["pred_px"], s["gt_px"], sb["inter"], sb["union"]
            r.update({"iou": s["iou"], "dice": s["dice"], "gt_car_px": s["gt_px"], "pred_px": s["pred_px"],
                      "iou_in_prompt_boxes": sb["iou"]})
        masks.clear()
        rec.update(status="chạy đủ" if n_ok == n_prompt else f"chạy {n_ok}/{n_prompt}", n_images_ok=n_ok,
                   n_images_with_prompt=n_prompt, n_failed=sum(str(r["status"]).startswith("error") for r in per),
                   load_sec=load_s, **cmv.timing_summary(per))
        inventory.append(rec)
        for r in per:
            per_image.append({"model": name, "task": "segmentation", **r})
        iou_img = np.where(U > 0, I / np.where(U > 0, U, 1), np.nan)
        row = {"model": name, "n_images_ok": n_ok, "n_images_with_prompt": n_prompt, "mean_sec": rec.get("mean_sec"),
               "total_sec": rec.get("total_sec"), "load_sec": load_s, "n_images_scored": int((U > 0).sum()),
               "pooled_iou": float(I.sum() / U.sum()), "pooled_iou_ci95": boot(ratio(I, U), len(ids), rng),
               "pooled_dice": float(2 * I.sum() / (P.sum() + G.sum())),
               "mean_iou_per_image": float(np.nanmean(iou_img)),
               "pooled_iou_in_prompt_boxes": float(I_b.sum() / U_b.sum()),
               "pooled_iou_in_prompt_boxes_ci95": boot(ratio(I_b, U_b), len(ids), rng)}
        if ref_seg is None:
            ref_seg = {"name": name, "I": I, "U": U, "Ib": I_b, "Ub": U_b, "t": t}
        else:
            R = ref_seg
            common = [i for i in ids if i in t and i in R["t"]]
            d = np.array([t[i] - R["t"][i] for i in common])
            row.update({
                "reference": R["name"],
                "pooled_iou_diff_vs_ref": row["pooled_iou"] - float(R["I"].sum() / R["U"].sum()),
                "pooled_iou_diff_ci95": boot(lambda i: ratio(I, U)(i) - ratio(R["I"], R["U"])(i), len(ids), rng),
                "iou_in_boxes_diff_ci95": boot(lambda i: ratio(I_b, U_b)(i) - ratio(R["Ib"], R["Ub"])(i), len(ids), rng),
                "time_diff_vs_ref_ms": float(d.mean() * 1000),
                "time_diff_ci95_ms": [x * 1000 for x in boot(lambda i: d[i].mean(), len(d), rng)],
                "n_images_faster_than_ref": int((d < 0).sum())})
        seg_rows.append(row)
        print(f"  {name}: {n_ok}/{n_prompt} ảnh · {rec.get('mean_sec', 0):.3f} s/ảnh · IoU gộp {row['pooled_iou']:.3f}")
        torch.cuda.empty_cache()

    config = {
        "run_id": run_id, "n_images": len(ids), "n_images_with_det_labels": len(gt_ids),
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "cpu": platform.processor(), "os": platform.platform()},
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                     "ultralytics": ultralytics.__version__, "opencv": cv2.__version__},
        "settings": {"device": args.device, "imgsz_yolo": args.imgsz, "conf_yolo": args.conf, "warmup_images": args.warmup,
                     "yolo_reference": Path(args.yolo[0]).stem if args.yolo else None,
                     "sam_reference": Path(args.sam[0]).stem if args.sam else None,
                     "sam_prompts": "555 box YOLOv8n đã bàn giao, giống nhau cho mọi SAM (SAM có prompt)",
                     "timing": "wall-clock một lời gọi model trên ảnh đã giải mã, có cuda synchronize; không tính nạp model, "
                               "ảnh khởi động, giải mã ảnh, tính chỉ số; model chạy lần lượt theo run_order",
                     "bootstrap": {"n": N_BOOT, "seed": SEED, "unit": "ảnh"}},
    }
    cmv.write_csv(out / "per_image_results.csv", per_image)
    cmv.write_csv(out / "model_inventory.csv", [{k: v for k, v in r.items()} for r in inventory])
    (out / "detection_summary.json").write_text(json.dumps(det_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "segmentation_summary.json").write_text(json.dumps(seg_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    os.environ.setdefault("YOLO_OFFLINE", "1")
    main()
