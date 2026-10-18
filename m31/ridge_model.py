"""Ridge đã huấn luyện trên pilot thật: CV theo nhóm image_id, fit cuối, lưu/đọc artifact JSON.

- Đầu vào: chỉ đặc trưng pre-label (FEATURES trong m31.calibration), cùng ma trận thiết kế và ridge
  với CalibratedTimeEstimator (`_design`, `_fit_ridge`, alpha = RIDGE_ALPHA).
- Nhãn: correction_time_sec (chỉ dùng để học/đánh giá, KHÔNG lưu vào artifact).
- Mỗi task (detection / segmentation) một mô hình riêng.
- Fold được chia theo image_id (cả hai task của một ảnh luôn chung fold), phân tầng theo
  "ảnh có cần sửa ở ít nhất một task".
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import FEATURES, RIDGE_ALPHA, CalibratedTimeEstimator, _design, _fit_ridge
from .schema import KEY_COLUMNS

ARTIFACT_VERSION = 1
FEATURE_DEFINITIONS = {
    "num_predictions": "Số box/mask pre-label của ảnh",
    "low_confidence_count": "Số dự đoán có confidence < 0.5",
    "uncertainty": "1 − mean_confidence; ảnh không có dự đoán (mean_confidence trống) được tính là 1.0 "
                   "(theo m31.calibration._design)",
}
EXCLUDED_INPUTS = ["correction_time_sec", "edits_added", "edits_removed", "edits_changed", "status", "notes",
                   "ground truth", "mọi thông tin phát sinh sau khi con người sửa"]


def group_folds(merged: pd.DataFrame, k: int, seed: int) -> list[np.ndarray]:
    """Trả về k mảng image_id; phân tầng theo ảnh có correction_time_sec > 0 ở bất kỳ task nào."""
    needs = merged.groupby("image_id")["correction_time_sec"].max() > 0
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    for flag in (True, False):
        ids = rng.permutation(np.array(sorted(needs.index[needs == flag])))
        for i, iid in enumerate(ids):
            folds[i % k].append(iid)
    return [np.array(f) for f in folds]


def cross_validate(merged: pd.DataFrame, k: int = 5, seed: int = 42, repeats: int = 20):
    """CV k-fold theo nhóm image_id, lặp `repeats` lần (seed, seed+1, ...).

    Trả về (summary per task, bảng fold của lần lặp đầu).
    """
    rows, fold_rows = [], []
    for r in range(repeats):
        folds = group_folds(merged, k, seed + r)
        for task, g in merged.groupby("task"):
            g = g.reset_index(drop=True)
            y = g["correction_time_sec"].to_numpy(float)
            X = _design(g)
            pred = {"ridge": np.zeros(len(g)), "zero": np.zeros(len(g)), "train_median": np.zeros(len(g))}
            for fi, test_ids in enumerate(folds):
                te = g["image_id"].isin(test_ids).to_numpy()
                tr = ~te
                pred["ridge"][te] = np.clip(X[te] @ _fit_ridge(X[tr], y[tr]), 0, None)
                pred["train_median"][te] = np.median(y[tr])
                if r == 0:
                    fold_rows.append({"task": task, "fold": fi, "n_test": int(te.sum()), "n_train": int(tr.sum()),
                                      "n_corrected_test": int((y[te] > 0).sum()),
                                      "n_corrected_train": int((y[tr] > 0).sum())})
            pos = y > 0
            for m, p in pred.items():
                err = np.abs(p - y)
                rows.append({"repeat": r, "task": task, "model": m, "mae": err.mean(),
                             "mae_corrected": err[pos].mean(), "mae_uncorrected": err[~pos].mean()})
    res = pd.DataFrame(rows)
    summary = {}
    for task, g in res.groupby("task"):
        piv = g.pivot(index="repeat", columns="model", values="mae")
        n = int((merged["task"] == task).sum())
        summary[task] = {
            "n_samples": n,
            "n_corrected": int(((merged["task"] == task) & (merged["correction_time_sec"] > 0)).sum()),
            "mae": {m: round(float(piv[m].mean()), 4) for m in piv},
            "mae_std_across_repeats": {m: round(float(piv[m].std()), 4) for m in piv},
            "mae_on_corrected_images": {m: round(float(v), 4)
                                        for m, v in g.groupby("model")["mae_corrected"].mean().items()},
            "mae_on_uncorrected_images": {m: round(float(v), 4)
                                          for m, v in g.groupby("model")["mae_uncorrected"].mean().items()},
            "pct_repeats_ridge_beats_zero": round(float((piv["ridge"] < piv["zero"]).mean()), 4),
            "pct_repeats_ridge_beats_train_median": round(float((piv["ridge"] < piv["train_median"]).mean()), 4),
        }
    return summary, pd.DataFrame(fold_rows)


def fit_final(merged: pd.DataFrame) -> dict[str, np.ndarray]:
    return {task: _fit_ridge(_design(g), g["correction_time_sec"].to_numpy(float))
            for task, g in merged.groupby("task")}


def build_artifact(merged: pd.DataFrame, coefs: dict[str, np.ndarray], cv_summary: dict, cv_settings: dict,
                   sources: list[dict]) -> dict:
    tasks = {}
    for task, c in coefs.items():
        g = merged[merged["task"] == task]
        tasks[task] = {
            "models_in_training_data": sorted(g["model"].unique().tolist()),
            "intercept": round(float(c[0]), 6),
            "coefficients": {f: round(float(v), 6) for f, v in zip(FEATURES, c[1:])},
            "n_samples": int(len(g)),
            "n_corrected": int((g["correction_time_sec"] > 0).sum()),
            "cv": cv_summary[task],
        }
    return {
        "artifact_version": ARTIFACT_VERSION,
        "model_type": "ridge_linear_regression_per_task",
        "target": "correction_time_sec (giây), dự đoán bị cắt về ≥ 0",
        "ridge_alpha": RIDGE_ALPHA,
        "intercept_penalized": False,
        "feature_scaling": "không chuẩn hoá (giống CalibratedTimeEstimator)",
        "features": FEATURES,
        "feature_definitions": FEATURE_DEFINITIONS,
        "excluded_inputs": EXCLUDED_INPUTS,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "training_data": sources,
        "cv_settings": cv_settings,
        "tasks": tasks,
        "warning": "Mô hình thử nghiệm trên pilot 100 ảnh BDD100K (lớp car), mỗi task chỉ 13 ảnh có thời gian sửa > 0. "
                   "Xem 'cv' của từng task: không được coi là đã kiểm chứng hay tốt hơn mốc so sánh nếu CV không cho thấy.",
    }


class TrainedRidgeEstimator(CalibratedTimeEstimator):
    """CalibratedTimeEstimator với hệ số đọc từ artifact JSON."""

    name = "Ridge đã huấn luyện trên pilot — thử nghiệm"

    def __init__(self, coefs: dict[str, np.ndarray], artifact: dict):
        super().__init__(coefs)
        self.artifact = artifact


def load_trained_ridge(path: str | Path) -> TrainedRidgeEstimator:
    art = json.loads(Path(path).read_text(encoding="utf-8"))
    if art.get("features") != FEATURES:
        raise ValueError(f"Artifact dùng features {art.get('features')}, app cần {FEATURES}.")
    coefs = {t: np.array([v["intercept"]] + [v["coefficients"][f] for f in FEATURES], dtype=float)
             for t, v in art["tasks"].items()}
    return TrainedRidgeEstimator(coefs, art)


def merge_for_training(prelabels: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Ghép 1-1 theo khoá; chỉ giữ khoá, đặc trưng đầu vào và nhãn (bỏ edits_*, status, notes)."""
    feats = ["mean_confidence", "num_predictions", "low_confidence_count"]
    m = prelabels[KEY_COLUMNS + feats].merge(
        log[KEY_COLUMNS + ["correction_time_sec"]], on=KEY_COLUMNS, how="inner", validate="one_to_one")
    return m
