"""Rà soát và tính lại chỉ số SAM của một lượt compare_model_zoo trên tập ảnh thống nhất.

Lý do: IoU trong lượt chạy được tính trên mọi ảnh có hợp (pred ∪ GT) > 0 = 98 ảnh, gồm 92 ảnh SAM thực sự chạy
và 6 ảnh KHÔNG có box prompt (YOLOv8n bỏ sót xe) nhưng có car trong nhãn -> IoU = 0 cho mọi SAM. 6 ảnh này
đo recall của prompt YOLO, không đo SAM. Script tính lại từ per_image_results.csv (không chạy lại model):

- Tập A (chỉ số chính, "SAM có prompt"): 92 ảnh có box prompt, nơi mọi SAM đã chạy.
- Tập B (đầu-cuối, gồm lỗi bỏ sót của prompt): 98 ảnh có IoU xác định = cách tính trong summary của lượt chạy.

Giao/hợp mỗi ảnh suy ra từ dice, pred_px, gt_car_px:  I = dice·(P+G)/2, U = P+G−I  (0 khi P+G = 0).
Kiểm tra: tập B phải tái tạo đúng pooled_iou trong segmentation_summary.json.
Bootstrap 95% theo ảnh trong từng tập, 2000 lần, seed 42; chênh lệch so với tham chiếu tính theo cặp.

Chạy:  .venv\\Scripts\\python.exe scripts\\audit_sam_eval.py [--run data/real/model_comparison/runs/<id>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
N_BOOT, SEED = 2000, 42


def boot(stat, n, rng):
    v = np.array([stat(rng.integers(0, n, n)) for _ in range(N_BOOT)], float)
    v = v[~np.isnan(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "data/real/model_comparison/runs/20261018-222741"))
    args = ap.parse_args()
    run = Path(args.run)
    p = pd.read_csv(run / "per_image_results.csv")
    s = p[p.task == "segmentation"].copy()
    summary = {r["model"]: r for r in json.loads((run / "segmentation_summary.json").read_text(encoding="utf-8"))}
    models = list(summary)
    ref = models[0]

    P, G = s["pred_px"].fillna(0.0), s["gt_car_px"].fillna(0.0)
    s["I"] = (s["dice"].fillna(0.0) * (P + G) / 2)
    s["U"] = P + G - s["I"]

    # Tập ảnh (giống nhau cho mọi model; kiểm tra)
    sets = {m: (frozenset(g.loc[g.status == "ok", "image_id"]), frozenset(g.loc[g.U > 0, "image_id"]))
            for m, g in s.groupby("model")}
    assert len(set(sets.values())) == 1, "Các model không dùng cùng tập ảnh"
    set_a, set_b = next(iter(sets.values()))
    r0 = s[s.model == ref].set_index("image_id")
    img = pd.DataFrame({"image_id": r0.index, "has_prompt": r0.status == "ok",
                        "n_prompts": r0["n_prompts"].fillna(0).astype(int), "gt_car_px": r0["gt_car_px"].astype(int)})
    img["in_set_A_92_prompted"] = img.image_id.isin(set_a)
    img["in_set_B_98_iou_defined"] = img.image_id.isin(set_b)
    img["note"] = np.select(
        [img.has_prompt & (img.gt_car_px == 0), img.has_prompt, img.gt_car_px > 0],
        ["có prompt, nhãn không có car (YOLO bắt nhầm) -> IoU = 0 cho mọi SAM", "có prompt, SAM đã chạy",
         "không có prompt (YOLO bỏ sót), nhãn có car -> IoU = 0 cho mọi SAM, SAM không chạy"],
        "không có prompt, nhãn không có car -> 0/0, không chấm")
    img.to_csv(run / "sam_eval_image_sets.csv", index=False)

    rng = np.random.default_rng(SEED)
    rows, checks = [], {}
    for set_name, ids in (("A_92_prompted", sorted(set_a)), ("B_98_iou_defined", sorted(set_b))):
        piv = {m: s[s.model == m].set_index("image_id").loc[ids] for m in models}
        Ir, Ur = piv[ref]["I"].to_numpy(), piv[ref]["U"].to_numpy()
        for m in models:
            I, U = piv[m]["I"].to_numpy(), piv[m]["U"].to_numpy()
            Pm, Gm = piv[m]["pred_px"].to_numpy(float), piv[m]["gt_car_px"].to_numpy(float)
            Ib, Ub = piv[m]["iou_in_prompt_boxes"].to_numpy(float), None
            iou_img = I / U
            row = {"eval_set": set_name, "n_images": len(ids), "model": m, "reference": ref,
                   "mean_sec": summary[m]["mean_sec"],
                   "pooled_iou": I.sum() / U.sum(),
                   "pooled_iou_ci95": boot(lambda i: I[i].sum() / U[i].sum(), len(ids), rng),
                   "pooled_dice": 2 * I.sum() / (Pm.sum() + Gm.sum()),
                   "mean_iou_per_image": float(iou_img.mean()),
                   "mean_iou_per_image_ci95": boot(lambda i: iou_img[i].mean(), len(ids), rng)}
            if m != ref:
                dimg = iou_img - Ir / Ur
                row.update({
                    "pooled_iou_diff_vs_ref": row["pooled_iou"] - Ir.sum() / Ur.sum(),
                    "pooled_iou_diff_ci95": boot(lambda i: I[i].sum() / U[i].sum() - Ir[i].sum() / Ur[i].sum(), len(ids), rng),
                    "mean_iou_diff_vs_ref": float(dimg.mean()),
                    "mean_iou_diff_ci95": boot(lambda i: dimg[i].mean(), len(ids), rng),
                    "n_images_better": int((dimg > 1e-3).sum()), "n_images_worse": int((dimg < -1e-3).sum()),
                    "n_images_tie": int((np.abs(dimg) <= 1e-3).sum())})
            rows.append(row)
            if set_name.startswith("B"):
                checks[m] = {"recomputed_pooled_iou": row["pooled_iou"], "summary_pooled_iou": summary[m]["pooled_iou"],
                             "abs_diff": abs(row["pooled_iou"] - summary[m]["pooled_iou"]),
                             "recomputed_mean_iou": row["mean_iou_per_image"],
                             "summary_mean_iou": summary[m]["mean_iou_per_image"]}
    out = pd.DataFrame(rows)
    flat = out.copy()
    for c in [c for c in flat.columns if c.endswith("ci95")]:
        flat[c + "_low"] = flat[c].map(lambda v: v[0] if isinstance(v, list) else None)
        flat[c + "_high"] = flat[c].map(lambda v: v[1] if isinstance(v, list) else None)
        flat = flat.drop(columns=c)
    flat.round(5).to_csv(run / "sam_eval_recomputed.csv", index=False)
    audit = {
        "set_A_92_prompted": len(set_a), "set_B_98_iou_defined": len(set_b),
        "set_B_minus_A": sorted(set_b - set_a), "not_scored": sorted(set(img.image_id) - set_b),
        "prompted_but_no_gt_car": img.loc[img.has_prompt & (img.gt_car_px == 0), "image_id"].tolist(),
        "consistency_check_set_B_vs_summary": checks,
        "max_abs_diff_pooled_iou": max(c["abs_diff"] for c in checks.values()),
        "bootstrap": {"n": N_BOOT, "seed": SEED, "unit": "ảnh trong từng tập"},
    }
    (run / "sam_eval_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    show = flat[["eval_set", "n_images", "model", "pooled_iou", "pooled_iou_ci95_low", "pooled_iou_ci95_high",
                 "mean_iou_per_image", "pooled_iou_diff_vs_ref", "pooled_iou_diff_ci95_low", "pooled_iou_diff_ci95_high",
                 "mean_iou_diff_ci95_low", "mean_iou_diff_ci95_high", "n_images_better", "n_images_worse"]]
    print(show.round(4).to_string(index=False))
    print(json.dumps({k: audit[k] for k in ("set_A_92_prompted", "set_B_98_iou_defined", "set_B_minus_A",
                                            "not_scored", "prompted_but_no_gt_car", "max_abs_diff_pooled_iou")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
