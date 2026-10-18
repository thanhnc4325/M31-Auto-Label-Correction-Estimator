"""Baseline heuristic TẠM THỜI — chưa được huấn luyện hay hiệu chỉnh trên dữ liệu thật.

Mọi hằng số dưới đây là GIẢ ĐỊNH ban đầu do nhóm đặt ra, KHÔNG phải số đo.
Chúng tồn tại để ứng dụng chạy được trước khi có correction_log.csv thật.
Khi có dữ liệu sửa thật, thời gian sửa sẽ được thay bằng mô hình hiệu chỉnh
(xem m31/calibration.py).
"""
from __future__ import annotations

import pandas as pd

# Thời gian sửa (giây) = base + review * num_predictions + fix * low_confidence_count
# GIẢ ĐỊNH — cần thay bằng giá trị hiệu chỉnh từ correction_log.csv.
BASELINE_TIME_PARAMS = {
    "detection":    {"base_sec": 10.0, "review_sec_per_pred": 2.0, "fix_sec_per_low_conf": 8.0},
    "segmentation": {"base_sec": 15.0, "review_sec_per_pred": 3.0, "fix_sec_per_low_conf": 20.0},
}

# Điểm chất lượng = 100 * (w_conf * mean_confidence + w_ratio * (1 - tỉ lệ dự đoán độ tin cậy thấp))
# GIẢ ĐỊNH — chưa được đối chiếu với ground truth hay số lần sửa thật.
QUALITY_WEIGHTS = {"mean_confidence": 0.6, "high_conf_ratio": 0.4}

# Ảnh không có dự đoán nào: không suy ra được chất lượng -> gán điểm trung tính
# và để ưu tiên cao để người kiểm tra xem có bỏ sót đối tượng không.
EMPTY_PREDICTION_SCORE = 50.0


def low_conf_ratio(df: pd.DataFrame) -> pd.Series:
    n = df["num_predictions"].where(df["num_predictions"] > 0)
    return (df["low_confidence_count"] / n).fillna(0.0)


def quality_score(df: pd.DataFrame) -> pd.Series:
    """Điểm chất lượng dự đoán 0–100 (ước lượng thử nghiệm)."""
    w = QUALITY_WEIGHTS
    conf = df["mean_confidence"].fillna(0.0)
    score = 100.0 * (w["mean_confidence"] * conf + w["high_conf_ratio"] * (1.0 - low_conf_ratio(df)))
    score = score.where(df["num_predictions"] > 0, EMPTY_PREDICTION_SCORE)
    return score.clip(0, 100).round(1)


def baseline_time(df: pd.DataFrame) -> pd.Series:
    """Thời gian sửa dự kiến (giây) theo công thức giả định."""
    params = pd.DataFrame.from_dict(BASELINE_TIME_PARAMS, orient="index")
    p = params.loc[df["task"]].reset_index(drop=True)
    p.index = df.index
    t = (
        p["base_sec"]
        + p["review_sec_per_pred"] * df["num_predictions"]
        + p["fix_sec_per_low_conf"] * df["low_confidence_count"]
    )
    return t.clip(lower=0).round(1)
