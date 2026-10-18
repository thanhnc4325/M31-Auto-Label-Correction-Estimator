"""Sinh dữ liệu MINH HỌA (tổng hợp ngẫu nhiên) để chạy thử giao diện.

CẢNH BÁO: Mọi số liệu do script này tạo ra là GIẢ, KHÔNG phải kết quả chạy YOLO/SAM
trên BDD100K và KHÔNG phải thời gian sửa đo thật. Không dùng để báo cáo kết quả.

Chạy:  python scripts/make_sample_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "sample"
N_PER_TASK = 30
N_LOGGED_PER_TASK = 24  # chỉ một phần ảnh có log sửa, giống thực tế


def main(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    pre_rows, log_rows = [], []
    for task, model, secs in [
        ("detection", "yolo_demo", (8, 2.5, 9)),
        ("segmentation", "sam_demo", (12, 3.5, 18)),
    ]:
        for i in range(N_PER_TASK):
            image_id = f"demo_{task[:3]}_{i + 1:03d}"
            n = int(rng.poisson(14)) if rng.random() > 0.07 else 0
            conf = float(np.clip(rng.beta(5, 2), 0.05, 0.99)) if n else np.nan
            n_low = int(rng.binomial(n, np.clip(1.1 - conf, 0, 1))) if n else 0
            pre_rows.append(dict(
                image_id=image_id, task=task, model=model,
                mean_confidence=None if np.isnan(conf) else round(conf, 3),
                num_predictions=n, low_confidence_count=n_low,
                image_width=1280, image_height=720,
            ))
            if i < N_LOGGED_PER_TASK:
                base, review, fix = secs
                t = base + review * n + fix * n_low + rng.normal(0, 8)
                if n == 0:
                    t += rng.uniform(20, 60)
                log_rows.append(dict(
                    image_id=image_id, task=task, model=model,
                    correction_time_sec=round(max(t, 3.0), 1),
                    edits_added=int(rng.poisson(1 + (3 if n == 0 else 0))),
                    edits_removed=int(rng.binomial(n_low, 0.4)),
                    edits_changed=int(rng.binomial(n_low, 0.5)),
                ))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pre_rows).to_csv(OUT / "ILLUSTRATIVE_prelabels.csv", index=False)
    pd.DataFrame(log_rows).to_csv(OUT / "ILLUSTRATIVE_correction_log.csv", index=False)
    print(f"Đã ghi dữ liệu MINH HỌA vào {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
