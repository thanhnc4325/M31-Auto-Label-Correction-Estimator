"""Schema và kiểm tra hợp lệ cho các file CSV bàn giao.

Đây là "hợp đồng dữ liệu" giữa M31 và các thành viên phụ trách YOLO/SAM.
Nếu schema thay đổi, chỉ cần sửa file này.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

KEY_COLUMNS = ["image_id", "task", "model"]

PRELABEL_COLUMNS = KEY_COLUMNS + [
    "mean_confidence",
    "num_predictions",
    "low_confidence_count",
    "image_width",
    "image_height",
]

CORRECTION_COLUMNS = KEY_COLUMNS + [
    "correction_time_sec",
    "edits_added",
    "edits_removed",
    "edits_changed",
]

# Cột đặc trưng pre-label TÙY CHỌN (giữ lại nếu có, vd. từ scripts/prepare_real_data.py).
# Tất cả đều tính được TRƯỚC khi con người sửa.
OPTIONAL_PRELABEL_NUMERIC = [
    "min_confidence", "small_box_count", "mean_box_area_ratio", "mean_mask_fill_ratio", "empty_mask_count",
]
OPTIONAL_PRELABEL_TEXT = ["confidence_source"]
OPTIONAL_PRELABEL_COLUMNS = OPTIONAL_PRELABEL_NUMERIC + OPTIONAL_PRELABEL_TEXT

# Cột chỉ để hiển thị/đối chiếu trong correction log.
OPTIONAL_CORRECTION_COLUMNS = ["status", "notes"]

TASKS = ("detection", "segmentation")
TASK_ALIASES = {
    "detection": "detection", "det": "detection", "detect": "detection", "bbox": "detection",
    "segmentation": "segmentation", "seg": "segmentation", "segment": "segmentation", "mask": "segmentation",
}

# Các token trong tên cột gợi ý đó là ground truth / chỉ số tính từ ground truth.
# Những cột này KHÔNG được dùng làm đầu vào dự đoán -> bị loại bỏ kèm cảnh báo.
SUSPECTED_GT_TOKENS = {"gt", "groundtruth", "ground", "truth", "iou", "map", "ap", "f1", "recall", "precision"}

# Token gợi ý thông tin chỉ có SAU khi con người sửa (thời gian sửa, thao tác sửa, trạng thái review).
# Nếu xuất hiện trong file pre-label -> loại bỏ để tránh rò rỉ dữ liệu.
POST_CORRECTION_TOKENS = {"correction", "corrected", "edits", "action", "human", "status", "notes", "reviewed"}


@dataclass
class ValidationResult:
    df: pd.DataFrame
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _normalize_keys(df: pd.DataFrame, res: ValidationResult) -> pd.DataFrame:
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    raw_task = df["task"].astype(str).str.strip().str.lower()
    df["task"] = raw_task.map(TASK_ALIASES)
    bad = raw_task[df["task"].isna()].unique().tolist()
    if bad:
        res.errors.append(f"Giá trị 'task' không hợp lệ: {bad}. Chỉ chấp nhận: {list(TASKS)}.")
    return df


def _to_numeric(df: pd.DataFrame, cols: list[str], res: ValidationResult) -> pd.DataFrame:
    for c in cols:
        converted = pd.to_numeric(df[c], errors="coerce")
        n_bad = int(converted.isna().sum() - df[c].isna().sum())
        if n_bad > 0:
            res.errors.append(f"Cột '{c}' có {n_bad} giá trị không phải số.")
        df[c] = converted
    return df


def _check_missing(df: pd.DataFrame, required: list[str], res: ValidationResult) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        res.errors.append(f"Thiếu cột bắt buộc: {missing}. Cần đủ: {required}.")
        return False
    return True


def _tokens(col: str) -> set[str]:
    return set(col.replace("-", "_").split("_"))


def _drop_extra(
    df: pd.DataFrame, required: list[str], optional: list[str], res: ValidationResult, check_post_correction: bool
) -> pd.DataFrame:
    keep = required + [c for c in optional if c in df.columns]
    extra = [c for c in df.columns if c not in keep]
    if not extra:
        return df[keep]
    gt_like = [c for c in extra if SUSPECTED_GT_TOKENS & _tokens(c)]
    if gt_like:
        res.warnings.append(
            f"Đã loại các cột có vẻ là ground truth / chỉ số đánh giá: {gt_like}. "
            "Ground truth không được dùng làm đầu vào dự đoán."
        )
    post = [c for c in extra if c not in gt_like and check_post_correction and POST_CORRECTION_TOKENS & _tokens(c)]
    if post:
        res.warnings.append(
            f"Đã loại các cột chỉ có sau khi con người sửa: {post}. "
            "Đưa các cột này vào correction_log.csv, không đưa vào file pre-label (tránh rò rỉ dữ liệu)."
        )
    other = [c for c in extra if c not in gt_like and c not in post]
    if other:
        res.warnings.append(f"Bỏ qua các cột ngoài schema: {other}.")
    return df[keep]


def validate_prelabels(df: pd.DataFrame) -> ValidationResult:
    res = ValidationResult(df=pd.DataFrame(columns=PRELABEL_COLUMNS))
    df = _normalize_columns(df)
    if not _check_missing(df, PRELABEL_COLUMNS, res):
        return res
    df = _drop_extra(df, PRELABEL_COLUMNS, OPTIONAL_PRELABEL_COLUMNS, res, check_post_correction=True)
    df = _normalize_keys(df, res)
    df = _to_numeric(df, PRELABEL_COLUMNS[3:] + [c for c in OPTIONAL_PRELABEL_NUMERIC if c in df.columns], res)

    if ((df["mean_confidence"] < 0) | (df["mean_confidence"] > 1)).any():
        res.errors.append("'mean_confidence' phải nằm trong [0, 1].")
    for c in ["num_predictions", "low_confidence_count"]:
        if (df[c] < 0).any():
            res.errors.append(f"'{c}' không được âm.")
    if (df["low_confidence_count"] > df["num_predictions"]).any():
        res.errors.append("'low_confidence_count' không được lớn hơn 'num_predictions'.")
    if ((df["image_width"] <= 0) | (df["image_height"] <= 0)).any():
        res.errors.append("Kích thước ảnh phải dương.")

    zero_pred_nan_conf = (df["num_predictions"] == 0) & df["mean_confidence"].isna()
    other_nan = df[PRELABEL_COLUMNS].drop(columns=["mean_confidence"]).isna().any(axis=1) | (
        df["mean_confidence"].isna() & ~zero_pred_nan_conf
    )
    if other_nan.any():
        res.errors.append(f"Có {int(other_nan.sum())} dòng thiếu giá trị.")

    dup = df.duplicated(subset=KEY_COLUMNS)
    if dup.any():
        res.errors.append(f"Có {int(dup.sum())} dòng trùng khóa (image_id, task, model).")

    res.df = df.reset_index(drop=True)
    return res


def validate_correction_log(df: pd.DataFrame) -> ValidationResult:
    res = ValidationResult(df=pd.DataFrame(columns=CORRECTION_COLUMNS))
    df = _normalize_columns(df)
    if not _check_missing(df, CORRECTION_COLUMNS, res):
        return res
    df = _drop_extra(df, CORRECTION_COLUMNS, OPTIONAL_CORRECTION_COLUMNS, res, check_post_correction=False)
    df = _normalize_keys(df, res)
    df = _to_numeric(df, CORRECTION_COLUMNS[3:], res)

    if (df[CORRECTION_COLUMNS[3:]] < 0).any().any():
        res.errors.append("Thời gian sửa và số chỉnh sửa không được âm.")
    if df[CORRECTION_COLUMNS[3:]].isna().any(axis=1).any():
        res.errors.append("Có dòng thiếu giá trị trong correction log.")

    dup = df.duplicated(subset=KEY_COLUMNS)
    if dup.any():
        res.warnings.append(
            f"Có {int(dup.sum())} dòng trùng khóa trong correction log; sẽ lấy trung bình theo khóa."
        )
        agg = {c: "mean" for c in CORRECTION_COLUMNS[3:]} | {c: "first" for c in OPTIONAL_CORRECTION_COLUMNS if c in df}
        df = df.groupby(KEY_COLUMNS, as_index=False).agg(agg)

    res.df = df.reset_index(drop=True)
    return res
