"""Thống kê theo cặp (cùng ảnh) cho kết quả scripts/compare_model_versions.py.

Đọc data/real/model_comparison/{per_image_results,detection_gt_subset}.csv, tính chênh lệch B − A theo từng ảnh
và khoảng tin cậy bootstrap 95% (lấy mẫu lại theo ẢNH, 5000 lần, seed 42). Ghi paired_stats.json.

Chạy:  .venv\\Scripts\\python.exe scripts\\summarize_model_versions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "real" / "model_comparison"
N_BOOT, SEED = 5000, 42


def boot_ci(stat, n: int, rng) -> list[float]:
    vals = [stat(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def paired(df: pd.DataFrame, a: str, b: str, col: str, rng, tie: float) -> dict:
    p = df.pivot(index="image_id", columns="model", values=col)[[a, b]].dropna()
    d = (p[b] - p[a]).to_numpy()
    return {"n_images": int(len(d)), f"mean_{a}": float(p[a].mean()), f"mean_{b}": float(p[b].mean()),
            "mean_diff_b_minus_a": float(d.mean()), "ci95_mean_diff": boot_ci(lambda i: d[i].mean(), len(d), rng),
            "n_b_better": int((d > tie).sum()) if "sec" not in col else int((d < -tie).sum()),
            "n_a_better": int((d < -tie).sum()) if "sec" not in col else int((d > tie).sum()),
            "n_tie": int((np.abs(d) <= tie).sum()), "tie_tolerance": tie}


def pooled_iou(df: pd.DataFrame, a: str, b: str, rng, prefix: str = "") -> dict:
    """IoU gộp pixel; giao/hợp mỗi ảnh suy ra từ dice, pred_px, gt_px (I = dice·(P+G)/2, U = P+G−I)."""
    out = {}
    tabs = {}
    for m in (a, b):
        g = df[(df.model == m)].set_index("image_id")
        ok = g[f"{prefix}iou"].notna() if prefix else g["iou"].notna()
        P, G = g["pred_px"].astype(float), g["gt_car_px"].astype(float)
        dice = g[f"dice{('_' + prefix.rstrip('_')) if prefix else ''}"] if prefix else g["dice"]
        I = (dice * (P + G) / 2).where(ok, 0.0)
        tabs[m] = pd.DataFrame({"I": I, "U": (P + G - I).where(ok, 0.0)})
    ids = tabs[a].index.intersection(tabs[b].index)
    Ia, Ua, Ib, Ub = (tabs[m].loc[ids, c].to_numpy() for m in (a, b) for c in ("I", "U"))
    f = lambda i: Ib[i].sum() / Ub[i].sum() - Ia[i].sum() / Ua[i].sum()  # noqa: E731
    out.update({f"pooled_iou_{a}": float(Ia.sum() / Ua.sum()), f"pooled_iou_{b}": float(Ib.sum() / Ub.sum()),
                "diff_b_minus_a": float(f(np.arange(len(ids)))), "ci95_diff": boot_ci(f, len(ids), rng)})
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    res = pd.read_csv(D / "per_image_results.csv")
    gt = pd.read_csv(D / "detection_gt_subset.csv")
    det, seg = res[res.task == "detection"], res[res.task == "segmentation"]
    ya, yb = list(dict.fromkeys(det.model))
    sa, sb = list(dict.fromkeys(seg.model))
    stats = {"bootstrap": {"n": N_BOOT, "seed": SEED, "unit": "ảnh"}}

    stats["detection_timing_sec"] = paired(det, ya, yb, "inference_sec", rng, tie=0.0)
    stats["detection_car_boxes_per_image"] = paired(det, ya, yb, "n_car_boxes", rng, tie=0.0)
    gsub = {}
    for m in (ya, yb):
        g = gt[gt.model == m].set_index("image_id")[["tp", "fp", "fn"]]
        gsub[m] = g
    ids = gsub[ya].index
    A, B = gsub[ya].loc[ids].to_numpy(float), gsub[yb].loc[ids].to_numpy(float)
    prec = lambda X, i: X[i, 0].sum() / max(X[i, 0].sum() + X[i, 1].sum(), 1)  # noqa: E731
    rec = lambda X, i: X[i, 0].sum() / max(X[i, 0].sum() + X[i, 2].sum(), 1)  # noqa: E731
    all_i = np.arange(len(ids))
    stats["detection_gt_subset"] = {
        "n_images": int(len(ids)), "n_gt_car_boxes": int(A[:, 0].sum() + A[:, 2].sum()),
        f"precision_{ya}": prec(A, all_i), f"precision_{yb}": prec(B, all_i),
        f"recall_{ya}": rec(A, all_i), f"recall_{yb}": rec(B, all_i),
        "precision_diff_ci95": boot_ci(lambda i: prec(B, i) - prec(A, i), len(ids), rng),
        "recall_diff_ci95": boot_ci(lambda i: rec(B, i) - rec(A, i), len(ids), rng),
    }
    sok = seg[seg.status == "ok"]
    stats["segmentation_timing_sec"] = paired(sok, sa, sb, "inference_sec", rng, tie=0.0)
    stats["segmentation_iou_per_image"] = paired(seg, sa, sb, "iou", rng, tie=0.001)
    stats["segmentation_iou_in_prompt_boxes_per_image"] = paired(seg, sa, sb, "iou_in_prompt_boxes", rng, tie=0.001)
    stats["segmentation_pooled_iou"] = pooled_iou(seg, sa, sb, rng)
    (D / "paired_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
