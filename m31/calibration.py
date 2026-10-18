"""Hiệu chỉnh và đánh giá ước lượng thời gian sửa từ correction_log.csv.

- Chỉ dùng đặc trưng pre-label làm đầu vào (KHÔNG dùng ground truth).
- Mục tiêu học là correction_time_sec đo thật trong correction log.
- MAE của baseline tính trên toàn bộ dòng khớp (baseline không học từ dữ liệu này).
- MAE của mô hình hiệu chỉnh tính bằng k-fold cross-validation để tránh đánh giá trên chính dữ liệu đã học.

Đây là VỊ TRÍ THAY THẾ: nếu nhóm muốn mô hình khác (vd. gradient boosting),
chỉ cần tạo lớp có `name`, `is_calibrated`, `predict(df) -> Series` giống CalibratedTimeEstimator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baseline import baseline_time, quality_score
from .schema import KEY_COLUMNS

FEATURES = ["num_predictions", "low_confidence_count", "uncertainty"]
MIN_ROWS_FOR_MAE = 5          # tối thiểu để báo MAE baseline
MIN_ROWS_FOR_FIT = 20         # tối thiểu mỗi task để chạy cross-validation
MIN_NONZERO_FOR_FIT = 20      # tối thiểu số ảnh CÓ sửa (thời gian > 0) mỗi task để cho phép dùng mô hình hiệu chỉnh
CV_FOLDS = 5
RIDGE_ALPHA = 1.0


class BaselineTimeEstimator:
    name = "Baseline heuristic (tạm thời, chưa hiệu chỉnh)"
    is_calibrated = False

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return baseline_time(df)


def _design(df: pd.DataFrame) -> np.ndarray:
    X = pd.DataFrame({
        "num_predictions": df["num_predictions"],
        "low_confidence_count": df["low_confidence_count"],
        "uncertainty": 1.0 - df["mean_confidence"].fillna(0.0),
    })[FEATURES].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    penalty = alpha * np.eye(X.shape[1])
    penalty[0, 0] = 0.0  # không phạt hệ số chặn
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


class CalibratedTimeEstimator:
    """Hồi quy tuyến tính (ridge) theo từng task, học từ correction_log thật.

    Task chưa đủ dữ liệu sẽ dùng lại baseline.
    """
    name = "Hồi quy tuyến tính hiệu chỉnh từ correction_log"
    is_calibrated = True

    def __init__(self, coefs: dict[str, np.ndarray]):
        self.coefs = coefs

    @classmethod
    def fit(cls, matched: pd.DataFrame, tasks: list[str] | None = None) -> "CalibratedTimeEstimator":
        coefs = {}
        for task, g in matched.groupby("task"):
            if (tasks is None or task in tasks) and len(g) >= MIN_ROWS_FOR_FIT:
                coefs[task] = _fit_ridge(_design(g), g["correction_time_sec"].to_numpy(dtype=float))
        return cls(coefs)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        out = baseline_time(df).astype(float)
        for task, coef in self.coefs.items():
            mask = df["task"] == task
            if mask.any():
                out.loc[mask] = _design(df[mask]) @ coef
        return out.clip(lower=0).round(1)

    def coef_table(self) -> pd.DataFrame:
        rows = [dict(task=t, intercept=c[0], **dict(zip(FEATURES, c[1:]))) for t, c in self.coefs.items()]
        return pd.DataFrame(rows)


@dataclass
class CalibrationReport:
    n_log: int
    n_matched: int
    n_unmatched_log: int
    baseline_mae: dict[str, float] = field(default_factory=dict)       # task/"tổng" -> MAE
    calibrated_cv_mae: dict[str, float] = field(default_factory=dict)  # task -> MAE (CV)
    constant_cv_mae: dict[str, float] = field(default_factory=dict)    # task -> MAE (CV) khi đoán hằng số = median
    n_nonzero: dict[str, int] = field(default_factory=dict)            # task -> số ảnh có thời gian sửa > 0
    quality_vs_edits_spearman: float | None = None
    quality_vs_time_spearman: dict[str, float] = field(default_factory=dict)
    estimator: CalibratedTimeEstimator | None = None
    matched: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)


def merge_with_log(prelabels: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    return prelabels.merge(log, on=KEY_COLUMNS, how="inner")


def _cv_mae(g: pd.DataFrame, k: int = CV_FOLDS, seed: int = 0) -> tuple[float, float]:
    """MAE k-fold của (hồi quy ridge, mốc hằng số = median của fold huấn luyện)."""
    idx = np.random.default_rng(seed).permutation(len(g))
    folds = np.array_split(idx, k)
    X, y = _design(g), g["correction_time_sec"].to_numpy(dtype=float)
    errs, const_errs = [], []
    for f in folds:
        train = np.setdiff1d(idx, f)
        pred = np.clip(X[f] @ _fit_ridge(X[train], y[train]), 0, None)
        errs.append(np.abs(pred - y[f]))
        const_errs.append(np.abs(np.median(y[train]) - y[f]))
    return float(np.concatenate(errs).mean()), float(np.concatenate(const_errs).mean())


def _spearman(a: pd.Series, b: pd.Series) -> float | None:
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return None
    return float(a.rank().corr(b.rank()))


def evaluate(prelabels: pd.DataFrame, log: pd.DataFrame) -> CalibrationReport:
    matched = merge_with_log(prelabels, log)
    rep = CalibrationReport(
        n_log=len(log), n_matched=len(matched), n_unmatched_log=len(log) - len(matched), matched=matched
    )
    if rep.n_unmatched_log:
        rep.notes.append(
            f"{rep.n_unmatched_log} dòng correction log không khớp (image_id, task, model) với file pre-label."
        )
    if len(matched) < MIN_ROWS_FOR_MAE:
        rep.notes.append(
            f"Chỉ có {len(matched)} dòng khớp — cần ít nhất {MIN_ROWS_FOR_MAE} để báo MAE. Chưa đánh giá."
        )
        return rep

    matched = matched.assign(baseline_pred=baseline_time(matched))
    err = (matched["baseline_pred"] - matched["correction_time_sec"]).abs()
    rep.baseline_mae = {t: float(e.mean()) for t, e in err.groupby(matched["task"])}
    rep.baseline_mae["tổng"] = float(err.mean())

    usable = []
    for task, g in matched.groupby("task"):
        rep.n_nonzero[task] = int((g["correction_time_sec"] > 0).sum())
        if len(g) < MIN_ROWS_FOR_FIT:
            rep.notes.append(
                f"Task '{task}': {len(g)} dòng < {MIN_ROWS_FOR_FIT} — chưa đủ để hiệu chỉnh, vẫn dùng baseline."
            )
            continue
        rep.calibrated_cv_mae[task], rep.constant_cv_mae[task] = _cv_mae(g)
        if rep.n_nonzero[task] >= MIN_NONZERO_FOR_FIT:
            usable.append(task)
        else:
            rep.notes.append(
                f"Task '{task}': chỉ {rep.n_nonzero[task]} ảnh có thời gian sửa > 0 (< {MIN_NONZERO_FOR_FIT}). "
                "MAE hiệu chỉnh chỉ để tham khảo; mô hình hiệu chỉnh KHÔNG được bật, vẫn dùng baseline."
            )
    if usable:
        rep.estimator = CalibratedTimeEstimator.fit(matched, tasks=usable)

    q = quality_score(matched)
    edits = matched[["edits_added", "edits_removed", "edits_changed"]].sum(axis=1)
    rep.quality_vs_edits_spearman = _spearman(q, edits)
    for task in matched["task"].unique():
        m = matched["task"] == task
        rho = _spearman(q[m], matched.loc[m, "correction_time_sec"])
        if rho is not None:
            rep.quality_vs_time_spearman[task] = rho

    rep.matched = matched
    return rep


def priority_vs_correction(estimates: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Đánh giá SAU: mỗi mức ưu tiên có bao nhiêu ảnh thực sự cần sửa (thời gian sửa > 0).

    `estimates` là đầu ra của m31.estimator.estimate (chỉ dựa trên pre-label);
    log chỉ được ghép vào ở bước này để đối chiếu.
    """
    m = estimates[KEY_COLUMNS + ["priority"]].merge(log[KEY_COLUMNS + ["correction_time_sec"]], on=KEY_COLUMNS)
    if m.empty:
        return pd.DataFrame()
    m["needs_correction"] = m["correction_time_sec"] > 0
    out = m.groupby(["task", "priority"]).agg(
        so_anh=("needs_correction", "size"),
        so_anh_can_sua=("needs_correction", "sum"),
        tong_thoi_gian_sua_that=("correction_time_sec", "sum"),
    ).reset_index()
    out["ti_le_can_sua"] = (out["so_anh_can_sua"] / out["so_anh"]).round(3)
    return out
