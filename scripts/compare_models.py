"""So sánh Ridge hiện tại với 2 mô hình cho dữ liệu nhiều số 0 (hurdle, LAD) trên pilot thật.

- Fold theo image_id (hai task của một ảnh luôn chung fold), phân tầng theo ảnh có sửa.
- (1) CV ngoài: MAE từng mô hình cố định + mốc luôn đoán 0 — so sánh KHÁM PHÁ (chọn theo bảng này là chọn trên test).
- (2) CV lồng nhau: vòng trong (chỉ trên tập train của vòng ngoài) chọn mô hình có MAE nhỏ nhất trong
  {ridge, hurdle, lad}; vòng ngoài đánh giá mô hình được chọn -> ước lượng không lệch cho QUY TRÌNH chọn.
- Tiêu chí (đặt trước): đổi mô hình mặc định chỉ khi quy trình lồng nhau có MAE < luôn đoán 0 ở ≥ 80% số lần lặp
  cho CẢ HAI task.

Chạy:  .venv\\Scripts\\python.exe scripts\\compare_models.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from m31.calibration import _design  # noqa: E402
from m31.ridge_model import group_folds, merge_for_training  # noqa: E402
from m31.schema import validate_correction_log, validate_prelabels  # noqa: E402
from m31.simple_models import CANDIDATES  # noqa: E402

PROC = ROOT / "data" / "real" / "processed"
MIN_PCT = 0.80


def inner_select(sub: pd.DataFrame, task: str, k: int, seed: int) -> str:
    """Chọn mô hình bằng CV nội bộ chỉ trên `sub` (tập train của vòng ngoài, gồm cả hai task)."""
    folds = group_folds(sub, k, seed)
    g = sub[sub["task"] == task].reset_index(drop=True)
    X, y = _design(g), g["correction_time_sec"].to_numpy(float)
    err = {m: 0.0 for m in CANDIDATES}
    for ids in folds:
        te = g["image_id"].isin(ids).to_numpy()
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        for m, fn in CANDIDATES.items():
            err[m] += np.abs(fn(X[~te], y[~te], X[te]) - y[te]).sum()
    return min(err, key=lambda m: (err[m], list(CANDIDATES).index(m)))  # hoà -> ưu tiên thứ tự cố định


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--inner-k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    pre = validate_prelabels(pd.read_csv(PROC / "pilot100_prelabels.csv"))
    log = validate_correction_log(pd.read_csv(PROC / "pilot100_correction_log.csv"))
    assert pre.ok and log.ok
    merged = merge_for_training(pre.df, log.df)

    fold_rows, rep_rows, picks = [], [], []
    for r in range(args.repeats):
        seed = args.seed + r
        folds = group_folds(merged, args.k, seed)
        for task, g in merged.groupby("task"):
            g = g.reset_index(drop=True)
            X, y = _design(g), g["correction_time_sec"].to_numpy(float)
            oof = {m: np.zeros(len(g)) for m in ["zero", *CANDIDATES, "nested_selected"]}
            for fi, ids in enumerate(folds):
                te = g["image_id"].isin(ids).to_numpy()
                for m, fn in CANDIDATES.items():
                    oof[m][te] = fn(X[~te], y[~te], X[te])
                pick = inner_select(merged[~merged["image_id"].isin(ids)], task, args.inner_k, seed + 1000)
                oof["nested_selected"][te] = CANDIDATES[pick](X[~te], y[~te], X[te])
                picks.append({"repeat": r, "task": task, "fold": fi, "selected": pick})
                for m in oof:
                    fold_rows.append({"repeat": r, "task": task, "fold": fi, "model": m,
                                      "n_test": int(te.sum()), "n_corrected_test": int((y[te] > 0).sum()),
                                      "mae": float(np.abs(oof[m][te] - y[te]).mean())})
            for m, p in oof.items():
                e = np.abs(p - y)
                rep_rows.append({"repeat": r, "task": task, "model": m, "mae": e.mean(),
                                 "mae_corrected": e[y > 0].mean(), "mae_uncorrected": e[y == 0].mean(),
                                 "n_pred_ge_0_5s": int((p >= 0.5).sum())})
    folds_df, rep = pd.DataFrame(fold_rows), pd.DataFrame(rep_rows)
    zero = rep[rep.model == "zero"].set_index(["task", "repeat"])["mae"]
    summ = []
    for (task, m), g in rep.groupby(["task", "model"], sort=False):
        gi = g.set_index(["task", "repeat"])["mae"]
        f = folds_df[(folds_df.task == task) & (folds_df.model == m)]["mae"]
        summ.append({"task": task, "model": m, "n": int((merged.task == task).sum()),
                     "n_corrected": int(((merged.task == task) & (merged.correction_time_sec > 0)).sum()),
                     "mae_mean": gi.mean(), "mae_std_repeats": gi.std(),
                     "fold_mae_min": f.min(), "fold_mae_max": f.max(), "fold_mae_std": f.std(),
                     "mae_on_corrected": g["mae_corrected"].mean(), "mae_on_uncorrected": g["mae_uncorrected"].mean(),
                     "avg_predictions_ge_0_5s": g["n_pred_ge_0_5s"].mean(),
                     "pct_repeats_beat_zero": float((gi < zero.loc[gi.index]).mean()) if m != "zero" else np.nan,
                     "pct_repeats_tie_zero": float(np.isclose(gi, zero.loc[gi.index]).mean()) if m != "zero" else np.nan})
    summ = pd.DataFrame(summ).round(4)
    pick_df = pd.DataFrame(picks)
    pick_counts = pick_df.groupby(["task", "selected"]).size().unstack(fill_value=0)
    pos_per_fold = folds_df[(folds_df.model == "zero")].groupby("task")["n_corrected_test"].agg(["min", "max"])

    nested = summ[summ.model == "nested_selected"].set_index("task")
    replace_default = bool((nested["pct_repeats_beat_zero"] >= MIN_PCT).all())

    summ.to_csv(PROC / "pilot100_model_comparison.csv", index=False)
    folds_df.round(4).to_csv(PROC / "pilot100_model_comparison_folds.csv", index=False)
    (PROC / "pilot100_model_comparison_summary.json").write_text(json.dumps({
        "settings": vars(args) | {"fold_grouping": "image_id", "stratified_on": "ảnh có thời gian sửa > 0",
                                  "candidates": list(CANDIDATES), "criterion_min_pct_repeats_beat_zero": MIN_PCT},
        "corrected_per_test_fold": pos_per_fold.to_dict(orient="index"),
        "nested_selection_counts": pick_counts.to_dict(orient="index"),
        "replace_default": replace_default,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 220)
    print(summ.to_string(index=False))
    print("\nSố ca sửa trong mỗi fold kiểm tra (mọi lần lặp):\n" + pos_per_fold.to_string())
    print("\nVòng trong chọn mô hình (số fold ngoài × lần lặp):\n" + pick_counts.to_string())
    print(f"\nĐổi mô hình mặc định? {replace_default}")


if __name__ == "__main__":
    main()
