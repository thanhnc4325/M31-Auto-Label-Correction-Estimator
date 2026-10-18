"""Train Ridge dự đoán correction_time_sec trên pilot thật, đánh giá CV theo nhóm image_id, lưu artifact JSON.

Chạy:  .venv\\Scripts\\python.exe scripts\\train_ridge.py
Kết quả:
  models/ridge_pilot100.json                      (hệ số, features, số mẫu, CV, nguồn dữ liệu — không chứa log)
  data/real/processed/pilot100_ridge_cv_folds.csv (số mẫu / số ảnh có sửa trong từng fold, lần lặp đầu)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from m31.ridge_model import build_artifact, cross_validate, fit_final, merge_for_training  # noqa: E402
from m31.schema import validate_correction_log, validate_prelabels  # noqa: E402

PROC = ROOT / "data" / "real" / "processed"
PRELABELS = PROC / "pilot100_prelabels.csv"
LOG = PROC / "pilot100_correction_log.csv"
OUT = ROOT / "models" / "ridge_pilot100.json"
FOLDS_OUT = PROC / "pilot100_ridge_cv_folds.csv"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    pre = validate_prelabels(pd.read_csv(PRELABELS))
    log = validate_correction_log(pd.read_csv(LOG))
    for name, res in [("pre-label", pre), ("correction log", log)]:
        if not res.ok:
            raise SystemExit(f"{name} không hợp lệ: {res.errors}")
    merged = merge_for_training(pre.df, log.df)
    counts = merged.assign(c=merged.correction_time_sec > 0).groupby("task").agg(n=("c", "size"), corrected=("c", "sum"))
    print("Số mẫu thực tế:\n" + counts.to_string())

    summary, folds = cross_validate(merged, k=args.k, seed=args.seed, repeats=args.repeats)
    coefs = fit_final(merged)
    sources = [{"file": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "rows": int(len(pd.read_csv(p)))}
               for p in (PRELABELS, LOG)]
    settings = {"method": f"{args.k}-fold theo nhóm image_id, phân tầng theo ảnh có sửa; lặp {args.repeats} lần",
                "k": args.k, "seed": args.seed, "repeats": args.repeats,
                "baselines": ["zero (luôn đoán 0 giây)", "train_median (median correction_time_sec của fold train)"]}
    art = build_artifact(merged, coefs, summary, settings, sources)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, ensure_ascii=False, indent=2), encoding="utf-8")
    folds.to_csv(FOLDS_OUT, index=False)
    print(f"\nĐã ghi {OUT}\nĐã ghi {FOLDS_OUT}")
    print("\nFold (lần lặp đầu):\n" + folds.to_string(index=False))
    for task, s in summary.items():
        print(f"\n[{task}] n={s['n_samples']}, có sửa={s['n_corrected']}")
        print(f"  CV MAE: ridge={s['mae']['ridge']:.3f}  zero={s['mae']['zero']:.3f}  "
              f"train_median={s['mae']['train_median']:.3f}  (độ lệch chuẩn giữa lần lặp ridge="
              f"{s['mae_std_across_repeats']['ridge']:.3f})")
        print(f"  Ridge thắng zero ở {s['pct_repeats_ridge_beats_zero']:.0%} lần lặp; "
              f"thắng train_median ở {s['pct_repeats_ridge_beats_train_median']:.0%}")
        print(f"  Hệ số: intercept={art['tasks'][task]['intercept']:.3f}, {art['tasks'][task]['coefficients']}")


if __name__ == "__main__":
    main()
