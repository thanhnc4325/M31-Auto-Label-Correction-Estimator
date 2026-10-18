"""Ghép đầu ra cho từng ảnh: điểm chất lượng, thời gian sửa, ưu tiên, lý do."""
from __future__ import annotations

import pandas as pd

from .baseline import low_conf_ratio, quality_score

# Ngưỡng ưu tiên theo điểm chất lượng (có thể chỉnh trong giao diện).
DEFAULT_THRESHOLDS = {"high_below": 50.0, "medium_below": 75.0}
PRIORITY_ORDER = {"cao": 0, "vừa": 1, "thấp": 2}

TASK_VI = {"detection": "phát hiện (detection)", "segmentation": "phân đoạn (segmentation)"}


def assign_priority(score: pd.Series, num_predictions: pd.Series, thresholds: dict) -> pd.Series:
    pr = pd.Series("thấp", index=score.index, dtype=object)
    pr[score < thresholds["medium_below"]] = "vừa"
    pr[score < thresholds["high_below"]] = "cao"
    pr[num_predictions == 0] = "cao"
    return pr


def _reason(row: pd.Series) -> str:
    if row["num_predictions"] == 0:
        return "Không có dự đoán nào — có thể bỏ sót đối tượng, cần kiểm tra thủ công."
    parts = []
    conf = row["mean_confidence"]
    level = "thấp" if conf < 0.5 else ("trung bình" if conf < 0.75 else "cao")
    parts.append(f"độ tin cậy TB {conf:.2f} ({level})")
    n_low, n = int(row["low_confidence_count"]), int(row["num_predictions"])
    if n_low:
        parts.append(f"{n_low}/{n} dự đoán độ tin cậy thấp ({row['low_conf_ratio']:.0%})")
    else:
        parts.append(f"không có dự đoán độ tin cậy thấp trong {n} dự đoán")
    if n >= 30:
        parts.append(f"nhiều đối tượng ({n}) nên tốn thời gian rà soát")
    # Đặc trưng tùy chọn (chỉ mô tả, không đổi điểm số).
    if pd.notna(row.get("small_box_count")) and row["small_box_count"] > 0:
        parts.append(f"{int(row['small_box_count'])} box nhỏ (< 32×32 px)")
    if pd.notna(row.get("mean_mask_fill_ratio")):
        parts.append(f"mask lấp TB {row['mean_mask_fill_ratio']:.0%} diện tích box prompt")
    if pd.notna(row.get("empty_mask_count")) and row["empty_mask_count"] > 0:
        parts.append(f"{int(row['empty_mask_count'])} mask rỗng")
    if row.get("confidence_source") == "yolo_box_prompt":
        parts.append("confidence lấy từ box YOLO (SAM không có điểm riêng)")
    if row["task"] == "segmentation":
        parts.append("baseline giả định mask sửa lâu hơn bbox")
    text = "; ".join(parts)
    return text[0].upper() + text[1:] + "."


def estimate(prelabels: pd.DataFrame, time_estimator, thresholds: dict | None = None) -> pd.DataFrame:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    df = prelabels.copy()
    df["low_conf_ratio"] = low_conf_ratio(df)
    df["predicted_quality_score"] = quality_score(df)
    df["estimated_correction_time_sec"] = time_estimator.predict(df)
    df["priority"] = assign_priority(df["predicted_quality_score"], df["num_predictions"], thresholds)
    df["reason"] = df.apply(_reason, axis=1)
    df["time_source"] = time_estimator.name
    return df
