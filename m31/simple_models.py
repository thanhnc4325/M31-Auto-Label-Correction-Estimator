"""Hai mô hình đơn giản cho correction_time_sec nhiều số 0 (so sánh với Ridge hiện tại).

Cả hai dùng đúng 3 đặc trưng pre-label như Ridge (m31.calibration._design), không thêm đặc trưng.

- hurdle: logistic L2 dự đoán P(thời gian > 0); dự đoán = median thời gian của các ca có sửa trong tập train
  nếu P > 0.5, ngược lại 0. Ngưỡng 0.5 là quy tắc tối ưu MAE (median của phân phối có điều kiện), KHÔNG tinh chỉnh.
- lad: hồi quy tuyến tính tối thiểu sai số tuyệt đối (hồi quy trung vị), giải bằng IRLS.
"""
from __future__ import annotations

import numpy as np

from .calibration import _fit_ridge

LOGIT_L2 = 1.0
HURDLE_THRESHOLD = 0.5


def _standardize(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chuẩn hoá các cột (trừ hệ số chặn ở cột 0) theo thống kê của tập train."""
    mu, sd = Xtr[:, 1:].mean(0), Xtr[:, 1:].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    f = lambda X: np.column_stack([X[:, 0], (X[:, 1:] - mu) / sd])  # noqa: E731
    return f(Xtr), f(Xte)


def _logistic(X: np.ndarray, y: np.ndarray, lam: float = LOGIT_L2, iters: int = 100) -> np.ndarray:
    w = np.zeros(X.shape[1])
    P = lam * np.eye(X.shape[1]); P[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        step = np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + P, X.T @ (p - y) + P @ w)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def fit_predict_hurdle(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    pos = ytr > 0
    if pos.sum() == 0:
        return np.zeros(len(Xte))
    Str, Ste = _standardize(Xtr, Xte)
    w = _logistic(Str, pos.astype(float))
    p = 1 / (1 + np.exp(-np.clip(Ste @ w, -30, 30)))
    return np.where(p > HURDLE_THRESHOLD, float(np.median(ytr[pos])), 0.0)


def fit_predict_lad(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, iters: int = 200,
                    delta: float = 1e-4) -> np.ndarray:
    """Hồi quy trung vị (L1) bằng IRLS; khởi tạo từ OLS, ridge rất nhỏ chỉ để ổn định số học."""
    reg = 1e-6 * np.eye(Xtr.shape[1]); reg[0, 0] = 0
    w = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ ytr)
    for _ in range(iters):
        wt = 1.0 / np.maximum(np.abs(ytr - Xtr @ w), delta)
        w_new = np.linalg.solve((Xtr * wt[:, None]).T @ Xtr + reg, (Xtr * wt[:, None]).T @ ytr)
        if np.abs(w_new - w).max() < 1e-9:
            w = w_new
            break
        w = w_new
    return np.clip(Xte @ w, 0, None)


def fit_predict_ridge(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    return np.clip(Xte @ _fit_ridge(Xtr, ytr), 0, None)


CANDIDATES = {"ridge": fit_predict_ridge, "hurdle": fit_predict_hurdle, "lad": fit_predict_lad}
