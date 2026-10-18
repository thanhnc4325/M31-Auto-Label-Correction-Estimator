"""Phân tích đặc trưng pilot: dự đoán thời gian sửa / ảnh cần sửa so với các baseline đơn giản.

- Đầu vào mô hình: data/real/processed/pilot100_features.csv (chỉ thông tin trước khi sửa).
- Nhãn (chỉ để đánh giá): correction_time_sec từ pilot100_correction_log.csv.
- CV: 5-fold PHÂN TẦNG theo (thời gian > 0), lặp R lần với seed khác nhau. Chuẩn hoá/điền thiếu
  được tính trong từng fold huấn luyện.
- Ghi kết quả vào data/real/processed/pilot100_cv_*.csv và pilot100_analysis_summary.json.

Tiêu chí tích hợp (đặt TRƯỚC khi chạy, chỉ áp dụng cho nhóm A):
  (1) ridge_A hoặc hurdle_A có MAE < luôn-đoán-0 trong ≥ 80% số lần lặp, HOẶC
  (2) logistic_A có AUC CV trung bình ≥ 0.70 và p hoán vị < 0.05.

Chạy:  .venv\\Scripts\\python.exe scripts\\analyze_features.py [--repeats 50 --perms 200]
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_features import GROUP_A, GROUP_B  # noqa: E402
from m31.baseline import baseline_time, quality_score  # noqa: E402

PROC = ROOT / "data" / "real" / "processed"
K = 5
RIDGE_ALPHA = 10.0
LOGIT_L2 = 1.0
MIN_PCT_BEAT_ZERO = 0.80
MIN_AUC = 0.70
MAX_PERM_P = 0.05


# ------------------------------------------------------------------ tiện ích
def stratified_folds(y_pos: np.ndarray, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    folds = [[] for _ in range(k)]
    for cls in (True, False):
        idx = rng.permutation(np.flatnonzero(y_pos == cls))
        for i, j in enumerate(idx):
            folds[i % k].append(j)
    return [np.array(sorted(f)) for f in folds]


def auc(score: np.ndarray, pos: np.ndarray) -> float:
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def prep(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Điền thiếu bằng median fold huấn luyện, chuẩn hoá theo fold huấn luyện, thêm hệ số chặn."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    one = lambda X: np.column_stack([np.ones(len(X)), (X - mu) / sd])  # noqa: E731
    return one(Xtr), one(Xte)


def ridge(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    P = alpha * np.eye(X.shape[1]); P[0, 0] = 0
    return np.linalg.solve(X.T @ X + P, X.T @ y)


def logistic(X: np.ndarray, y: np.ndarray, lam: float = LOGIT_L2, iters: int = 50) -> np.ndarray:
    w = np.zeros(X.shape[1])
    P = lam * np.eye(X.shape[1]); P[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + P @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + P
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


# ------------------------------------------------------------------ một task
def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    usable = lambda cols: [c for c in cols if df[c].notna().any() and df[c].nunique(dropna=True) > 1]  # noqa: E731
    return {"A": usable(GROUP_A), "AB": usable(GROUP_A + GROUP_B), "n_boxes": ["n_boxes"]}


def logistic_oof(df: pd.DataFrame, seed: int, cols: list[str]) -> np.ndarray:
    """Chỉ tính xác suất OOF của logistic (dùng cho kiểm định hoán vị)."""
    pos = df["correction_time_sec"].to_numpy(float) > 0
    X = df[cols].to_numpy(float)
    out = np.zeros(len(pos))
    for te in stratified_folds(pos, K, np.random.default_rng(seed)):
        tr = np.setdiff1d(np.arange(len(pos)), te)
        Xtr, Xte = prep(X[tr], X[te])
        out[te] = sigmoid(Xte @ logistic(Xtr, pos[tr].astype(float)))
    return out


def run_cv(df: pd.DataFrame, seed: int, fsets: dict[str, list[str]]):
    y = df["correction_time_sec"].to_numpy(float)
    pos = y > 0
    folds = stratified_folds(pos, K, np.random.default_rng(seed))
    oof_reg = {m: np.zeros(len(y)) for m in
               ["zero", "train_mean", "m31_baseline", "ridge_n_boxes", "ridge_A", "ridge_AB", "hurdle_A", "hurdle_AB"]}
    oof_cls = {m: np.zeros(len(y)) for m in ["logistic_A", "logistic_AB"]}
    fold_rows, fold_mae = [], []
    base_pred = df["m31_baseline_pred"].to_numpy(float)
    X = {k: df[v].to_numpy(float) for k, v in fsets.items()}
    for fi, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(y)), te)
        fold_rows.append({"fold": fi, "n_test": len(te), "n_pos_test": int(pos[te].sum()),
                          "n_pos_train": int(pos[tr].sum())})
        oof_reg["zero"][te] = 0.0
        oof_reg["train_mean"][te] = y[tr].mean()
        oof_reg["m31_baseline"][te] = base_pred[te]
        mean_pos = y[tr][pos[tr]].mean()
        for name, key in [("ridge_n_boxes", "n_boxes"), ("ridge_A", "A"), ("ridge_AB", "AB")]:
            Xtr, Xte = prep(X[key][tr], X[key][te])
            oof_reg[name][te] = np.clip(Xte @ ridge(Xtr, y[tr]), 0, None)
        for key in ["A", "AB"]:
            Xtr, Xte = prep(X[key][tr], X[key][te])
            p = sigmoid(Xte @ logistic(Xtr, pos[tr].astype(float)))
            oof_cls[f"logistic_{key}"][te] = p
            oof_reg[f"hurdle_{key}"][te] = p * mean_pos
        for m, pred in oof_reg.items():
            fold_mae.append({"fold": fi, "model": m, "mae": float(np.abs(pred[te] - y[te]).mean())})
    return oof_reg, oof_cls, fold_rows, fold_mae


def perm_test(df: pd.DataFrame, fsets, key: str, observed: float, perms: int, repeats: int, seed0: int) -> float:
    rng = np.random.default_rng(seed0)
    hits = 0
    for _ in range(perms):
        d = df.copy()
        d["correction_time_sec"] = rng.permutation(d["correction_time_sec"].to_numpy())
        pos = d["correction_time_sec"].to_numpy() > 0
        aucs = [auc(logistic_oof(d, r, fsets[key]), pos) for r in range(repeats)]
        hits += np.mean(aucs) >= observed
    return (hits + 1) / (perms + 1)


def analyze_task(task: str, df: pd.DataFrame, repeats: int, perms: int, perm_repeats: int):
    fsets = feature_sets(df)
    y = df["correction_time_sec"].to_numpy(float)
    pos = y > 0
    reg_rows, cls_rows, fold_rows, foldmae_rows = [], [], [], []
    for r in range(repeats):
        oof_reg, oof_cls, frows, fmae = run_cv(df, r, fsets)
        for m, pred in oof_reg.items():
            reg_rows.append({"repeat": r, "model": m, "mae": np.abs(pred - y).mean(),
                             "mae_pos": np.abs(pred - y)[pos].mean(), "mae_neg": np.abs(pred - y)[~pos].mean()})
        for m, p in oof_cls.items():
            cls_rows.append({"repeat": r, "model": m, "auc": auc(p, pos)})
        fold_rows += [{"task": task, "repeat": r, **f} for f in frows]
        foldmae_rows += [{"repeat": r, **f} for f in fmae]
    reg = pd.DataFrame(reg_rows)
    fm = pd.DataFrame(foldmae_rows)
    zero = reg[reg.model == "zero"].set_index("repeat")["mae"]
    zero_f = fm[fm.model == "zero"].set_index(["repeat", "fold"])["mae"]
    reg_sum = []
    for m, g in reg.groupby("model", sort=False):
        gi = g.set_index("repeat")["mae"]
        f = fm[fm.model == m].set_index(["repeat", "fold"])["mae"]
        reg_sum.append({
            "task": task, "model": m, "mae_mean": gi.mean(), "mae_std": gi.std(),
            "mae_p05": gi.quantile(0.05), "mae_p95": gi.quantile(0.95),
            "fold_mae_min": f.min(), "fold_mae_max": f.max(),
            "mae_on_corrected_images": g["mae_pos"].mean(), "mae_on_uncorrected_images": g["mae_neg"].mean(),
            "pct_repeats_beat_zero": float((gi < zero).mean()) if m != "zero" else np.nan,
            "pct_folds_beat_zero": float((f < zero_f).mean()) if m != "zero" else np.nan,
        })
    cls = pd.DataFrame(cls_rows)
    cls_sum = []
    fixed = {
        "m31_quality_score (thấp → cần sửa)": -df["m31_quality"].to_numpy(float),
        "n_boxes (nhiều → cần sửa)": df["n_boxes"].to_numpy(float),
        "conf_min (thấp → cần sửa)": -df["conf_min"].fillna(1.0).to_numpy(float),
    }
    for name, s in fixed.items():
        cls_sum.append({"task": task, "model": name, "fitted": False, "auc_mean": auc(s, pos),
                        "auc_std": 0.0, "auc_p05": np.nan, "auc_p95": np.nan, "perm_p": np.nan})
    for m, g in cls.groupby("model", sort=False):
        key = m.split("_")[1]
        obs = g["auc"].iloc[:perm_repeats].mean()
        p = perm_test(df, fsets, key, obs, perms, perm_repeats, seed0=zlib.crc32(f"{task}/{key}".encode())) if perms else np.nan
        cls_sum.append({"task": task, "model": m, "fitted": True, "auc_mean": g["auc"].mean(),
                        "auc_std": g["auc"].std(), "auc_p05": g["auc"].quantile(0.05),
                        "auc_p95": g["auc"].quantile(0.95), "perm_p": p})
    # Đơn biến (khám phá): AUC theo chiều "giá trị cao → cần sửa", CI bootstrap.
    rng = np.random.default_rng(0)
    uni = []
    for c in sorted(set(fsets["AB"])):
        ok = df[c].notna().to_numpy()
        s, p_ = df.loc[ok, c].to_numpy(float), pos[ok]
        a = auc(s, p_)
        boots = []
        for _ in range(1000):
            i = rng.integers(0, len(s), len(s))
            if 0 < p_[i].sum() < len(i):
                boots.append(auc(s[i], p_[i]))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        uni.append({"task": task, "feature": c, "group": "B" if c in GROUP_B else "A", "n": int(ok.sum()),
                    "n_corrected": int(p_.sum()), "auc_high_means_correction": a, "ci95_low": lo, "ci95_high": hi,
                    "ci_excludes_0.5": bool(lo > 0.5 or hi < 0.5)})
    return pd.DataFrame(reg_sum), pd.DataFrame(cls_sum), pd.DataFrame(fold_rows), pd.DataFrame(uni), fsets


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--perm-repeats", type=int, default=5)
    args = ap.parse_args()

    feats = pd.read_csv(PROC / "pilot100_features.csv")
    log = pd.read_csv(PROC / "pilot100_correction_log.csv")[["image_id", "task", "model", "correction_time_sec"]]
    # Nhãn chỉ được ghép ở đây, sau khi đặc trưng đã tính xong.
    df = feats.merge(log, on=["image_id", "task", "model"], how="inner", validate="one_to_one")
    assert len(df) == len(feats) == 200, "Đặc trưng và log không khớp 1-1"
    m31_in = df.assign(num_predictions=df["n_boxes"], low_confidence_count=df["conf_n_lt030"] + df["conf_n_030_050"],
                       mean_confidence=df["conf_mean"])
    df["m31_baseline_pred"] = baseline_time(m31_in)
    df["m31_quality"] = quality_score(m31_in)

    regs, clss, folds, unis, sets = [], [], [], [], {}
    for task, g in df.groupby("task"):
        g = g.reset_index(drop=True)
        r, c, f, u, fs = analyze_task(task, g, args.repeats, args.perms, args.perm_repeats)
        regs.append(r); clss.append(c); folds.append(f); unis.append(u)
        sets[task] = {k: v for k, v in fs.items()}
    reg, cls, fold, uni = (pd.concat(x, ignore_index=True) for x in (regs, clss, folds, unis))

    decision = {}
    for task in reg["task"].unique():
        rt = reg[reg.task == task].set_index("model")
        ct = cls[cls.task == task].set_index("model")
        c1 = {m: bool(rt.loc[m, "pct_repeats_beat_zero"] >= MIN_PCT_BEAT_ZERO) for m in ["ridge_A", "hurdle_A"]}
        c2 = bool(ct.loc["logistic_A", "auc_mean"] >= MIN_AUC and ct.loc["logistic_A", "perm_p"] < MAX_PERM_P)
        decision[task] = {"criterion1_beats_zero": c1, "criterion2_ranking": c2,
                          "integrate": bool(any(c1.values()) or c2)}

    fold_summary = (fold.groupby(["task"]).agg(pos_test_min=("n_pos_test", "min"), pos_test_max=("n_pos_test", "max"),
                                               pos_train_min=("n_pos_train", "min"),
                                               pos_train_max=("n_pos_train", "max")).reset_index())
    reg.round(4).to_csv(PROC / "pilot100_cv_regression.csv", index=False)
    cls.round(4).to_csv(PROC / "pilot100_cv_ranking.csv", index=False)
    fold[fold.repeat == 0].to_csv(PROC / "pilot100_cv_folds_repeat0.csv", index=False)
    uni.round(4).to_csv(PROC / "pilot100_univariate_auc.csv", index=False)
    summary = {
        "settings": {"k": K, "repeats": args.repeats, "stratified_on": "correction_time_sec > 0",
                     "ridge_alpha": RIDGE_ALPHA, "logistic_l2": LOGIT_L2, "perms": args.perms,
                     "perm_repeats": args.perm_repeats},
        "n_rows_per_task": int(len(df) // 2),
        "n_corrected_per_task": df.assign(p=df.correction_time_sec > 0).groupby("task")["p"].sum().astype(int).to_dict(),
        "fold_positive_counts": fold_summary.to_dict(orient="records"),
        "feature_sets": sets,
        "criteria": {"min_pct_repeats_beat_zero": MIN_PCT_BEAT_ZERO, "min_auc": MIN_AUC, "max_perm_p": MAX_PERM_P,
                     "applies_to": "nhóm A"},
        "decision": decision,
    }
    (PROC / "pilot100_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
    pd.set_option("display.width", 220)
    print(reg.round(3).to_string()); print(cls.round(3).to_string())
    print(uni[uni["ci_excludes_0.5"]].round(3).to_string())
    print(json.dumps({"folds": summary["fold_positive_counts"], "decision": decision}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
