"""Đọc kết quả benchmark YOLO/SAM đã chạy (scripts/compare_model_zoo.py + scripts/audit_sam_eval.py).

Chỉ đọc file có sẵn trong data/real/model_comparison/runs/<run_id>/; không tính lại, không tạo số liệu.
Thiếu file -> ghi vào danh sách `problems` để giao diện báo rõ, bỏ qua phần tương ứng của lượt đó.

- Detection: detection_summary.json (precision/recall trên ảnh có nhãn box, CI, chênh lệch so với tham chiếu).
- Segmentation: chỉ số CHÍNH lấy từ sam_eval_recomputed.csv, tập "A_92_prompted" (92 ảnh có box prompt);
  thời gian / số ảnh chạy / IoU trong box lấy từ segmentation_summary.json. Không dùng tập 98 ảnh thay thế.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SAM_MAIN_SET = "A_92_prompted"

_FAMILY_ORDER = ["yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "yolo26"]
_SIZE_ORDER = {s: i for i, s in enumerate(["t", "n", "s", "m", "b", "c", "l", "x", "e"])}


def display_name(model: str) -> str:
    """Tên dễ đọc: yolov8n -> YOLOv8n, yolo11m -> YOLO11m, sam2.1_t -> SAM2.1-T, mobile_sam -> MobileSAM."""
    if model == "mobile_sam":
        return "MobileSAM"
    m = re.fullmatch(r"sam(2(?:\.1)?)?_([a-z])", model)
    if m:
        return f"SAM{m.group(1) or ''}-{m.group(2).upper()}"
    m = re.fullmatch(r"yolo(v?)(\d+)([a-z])", model)
    if m:
        return f"YOLO{m.group(1)}{m.group(2)}{m.group(3)}"
    return model


def yolo_sort_key(model: str) -> tuple:
    m = re.fullmatch(r"(yolov?\d+)([a-z])", model)
    if not m:
        return (99, 99, model)
    fam = _FAMILY_ORDER.index(m.group(1)) if m.group(1) in _FAMILY_ORDER else 98
    return (fam, _SIZE_ORDER.get(m.group(2), 50), model)


@dataclass
class BenchmarkData:
    detection: pd.DataFrame = field(default_factory=pd.DataFrame)
    segmentation: pd.DataFrame = field(default_factory=pd.DataFrame)
    configs: dict = field(default_factory=dict)          # run_id -> run_config.json
    problems: list[str] = field(default_factory=list)    # thông báo thiếu dữ liệu

    @property
    def available(self) -> bool:
        return not self.detection.empty or not self.segmentation.empty


def _ci(v):
    return (v[0], v[1]) if isinstance(v, list) and len(v) == 2 else (None, None)


def _load_detection(run_id: str, path: Path) -> pd.DataFrame:
    rows = []
    for r in json.loads(path.read_text(encoding="utf-8")):
        p_lo, p_hi = _ci(r.get("precision_ci95"))
        r_lo, r_hi = _ci(r.get("recall_ci95"))
        dp_lo, dp_hi = _ci(r.get("precision_diff_vs_ref_ci95"))
        dr_lo, dr_hi = _ci(r.get("recall_diff_vs_ref_ci95"))
        rows.append({
            "run_id": run_id, "model": r["model"], "is_reference": "reference" not in r,
            "reference": r.get("reference", r["model"]),
            "n_images_ok": r.get("n_images_ok"), "mean_ms": r["mean_sec"] * 1000 if r.get("mean_sec") is not None else None,
            "total_car_boxes": r.get("total_car_boxes"),
            "gt_images": r.get("gt_images"), "gt_boxes": r.get("gt_boxes"),
            "tp": r.get("tp"), "fp": r.get("fp"), "fn": r.get("fn"),
            "precision": r.get("precision"), "precision_ci_low": p_lo, "precision_ci_high": p_hi,
            "recall": r.get("recall"), "recall_ci_low": r_lo, "recall_ci_high": r_hi,
            "d_precision_ci_low": dp_lo, "d_precision_ci_high": dp_hi,
            "d_recall_ci_low": dr_lo, "d_recall_ci_high": dr_hi,
            "matched_with_ref": r.get("boxes_matched_with_ref_iou0.5"),
            "only_ref": r.get("only_ref"), "only_model": r.get("only_model"),
        })
    return pd.DataFrame(rows)


def _load_segmentation(run_id: str, summary_path: Path, recomputed_path: Path) -> pd.DataFrame:
    summ = pd.DataFrame(json.loads(summary_path.read_text(encoding="utf-8")))
    rec = pd.read_csv(recomputed_path)
    rec = rec[rec["eval_set"] == SAM_MAIN_SET]
    if rec.empty:
        raise ValueError(f"không có dòng eval_set = {SAM_MAIN_SET}")
    keep = ["model", "n_images_ok", "n_images_with_prompt", "mean_sec", "total_sec", "pooled_iou_in_prompt_boxes"]
    df = rec.drop(columns=["mean_sec"]).merge(summ[[c for c in keep if c in summ]], on="model", how="left")
    ref = df["reference"].iloc[0]
    ref_sec = df.loc[df["model"] == ref, "mean_sec"]
    df["speedup_vs_ref"] = float(ref_sec.iloc[0]) / df["mean_sec"] if len(ref_sec) else None
    df["is_reference"] = df["model"] == ref
    df.insert(0, "run_id", run_id)
    return df


def load_benchmark(runs_dir: str | Path, label: str | None = None) -> BenchmarkData:
    """label: đường dẫn hiển thị trong thông báo (tránh lộ đường dẫn tuyệt đối của máy chủ)."""
    runs_dir = Path(runs_dir)
    label = label or runs_dir.name
    out = BenchmarkData()
    if not runs_dir.is_dir():
        out.problems.append(f"Không tìm thấy thư mục kết quả `{label}`.")
        return out
    det, seg = [], []
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        rid = run.name
        cfg = run / "run_config.json"
        if cfg.is_file():
            out.configs[rid] = json.loads(cfg.read_text(encoding="utf-8"))
        else:
            out.problems.append(f"Lượt `{rid}`: thiếu `run_config.json` (không hiển thị được cấu hình).")
        d = run / "detection_summary.json"
        if d.is_file():
            det.append(_load_detection(rid, d))
        else:
            out.problems.append(f"Lượt `{rid}`: thiếu `detection_summary.json`, bỏ qua phần YOLO của lượt này.")
        s, r = run / "segmentation_summary.json", run / "sam_eval_recomputed.csv"
        if s.is_file() and r.is_file():
            try:
                seg.append(_load_segmentation(rid, s, r))
            except (ValueError, KeyError) as exc:
                out.problems.append(f"Lượt `{rid}`: không đọc được chỉ số SAM ({exc}), bỏ qua phần SAM của lượt này.")
        elif s.is_file():
            out.problems.append(
                f"Lượt `{rid}`: thiếu `sam_eval_recomputed.csv` (IoU chính trên 92 ảnh có prompt), bỏ qua phần SAM. "
                f"Tạo bằng `python scripts/audit_sam_eval.py --run <thư mục lượt>`.")
        else:
            out.problems.append(f"Lượt `{rid}`: thiếu `segmentation_summary.json`, bỏ qua phần SAM của lượt này.")
    if det:
        out.detection = pd.concat(det, ignore_index=True)
    if seg:
        out.segmentation = pd.concat(seg, ignore_index=True)
    if not det and not seg and not out.problems:
        out.problems.append(f"Thư mục `{label}` chưa có lượt chạy nào.")
    return out
