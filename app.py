"""M31: Auto-Label Correction Estimator — giao diện Streamlit (MVP + pilot thật).

Chạy:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from m31.calibration import (
    MIN_NONZERO_FOR_FIT,
    MIN_ROWS_FOR_FIT,
    MIN_ROWS_FOR_MAE,
    BaselineTimeEstimator,
    evaluate,
    priority_vs_correction,
)
from m31.estimator import DEFAULT_THRESHOLDS, PRIORITY_ORDER, TASK_VI, estimate
from m31.schema import (
    CORRECTION_COLUMNS,
    OPTIONAL_PRELABEL_COLUMNS,
    PRELABEL_COLUMNS,
    validate_correction_log,
    validate_prelabels,
)

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "data" / "sample"
SAMPLE_PRELABELS = SAMPLE_DIR / "ILLUSTRATIVE_prelabels.csv"
SAMPLE_LOG = SAMPLE_DIR / "ILLUSTRATIVE_correction_log.csv"
PILOT_DIR = ROOT / "data" / "real" / "processed"
PILOT_PRELABELS = PILOT_DIR / "pilot100_prelabels.csv"
PILOT_LOG = PILOT_DIR / "pilot100_correction_log.csv"
PILOT_MANIFEST = PILOT_DIR / "pilot100_manifest.json"
PILOT_ZIP = ROOT / "data" / "real" / "local" / "end_corrected.zip"
OVERLAY_DIR_IN_ZIP = {"detection": "yolo_overlays", "segmentation": "sam_overlays"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

SRC_PILOT = "Pilot thật · 100 ảnh BDD100K (local)"
SRC_UPLOAD = "Tải lên file thật"
SRC_DEMO = "Dữ liệu MINH HỌA (không phải số đo thật)"

OUTPUT_COLUMNS = [
    "image_id", "task", "model", "predicted_quality_score",
    "estimated_correction_time_sec", "priority", "reason",
]

st.set_page_config(page_title="M31 – Ước lượng công sức sửa nhãn", layout="wide")


@st.cache_data
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))


@st.cache_resource
def open_zip(path: str) -> zipfile.ZipFile | None:
    try:
        return zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        return None


def show_validation(res, label: str) -> None:
    for w in res.warnings:
        st.warning(f"{label}: {w}")
    for e in res.errors:
        st.error(f"{label}: {e}")


def find_image(folder: str, image_id: str) -> Path | None:
    if not folder:
        return None
    base = Path(folder)
    for ext in IMAGE_EXTS:
        p = base / f"{image_id}{ext}"
        if p.is_file():
            return p
    return None


def overlay_from_zip(image_id: str, task: str) -> bytes | None:
    z = open_zip(str(PILOT_ZIP)) if PILOT_ZIP.is_file() else None
    if z is None:
        return None
    name = f"{OVERLAY_DIR_IN_ZIP[task]}/{image_id}_overlay.png"
    try:
        return z.read(name)
    except KeyError:
        return None


# ---------------------------------------------------------------- Sidebar: nguồn dữ liệu
# Pilot chỉ cần file pre-label; correction log là tùy chọn (có thể chưa được nhóm đồng ý công khai).
pilot_available = PILOT_PRELABELS.is_file()
pilot_log_available = PILOT_LOG.is_file()
st.sidebar.header("① Chọn dữ liệu")
source = st.sidebar.radio(
    "Nguồn dữ liệu", [SRC_PILOT, SRC_UPLOAD, SRC_DEMO], index=0 if pilot_available else 1,
)
kind = {SRC_PILOT: "pilot", SRC_UPLOAD: "upload", SRC_DEMO: "demo"}[source]

pre_raw, log_raw, manifest = None, None, None
if kind == "pilot" and pilot_available:
    pre_raw = pd.read_csv(PILOT_PRELABELS)
    if pilot_log_available:
        if st.sidebar.checkbox("Kèm correction log thật của pilot", value=True):
            log_raw = pd.read_csv(PILOT_LOG)
    else:
        st.sidebar.caption("Không có `pilot100_correction_log.csv` trên máy này — tab ④ sẽ không có đối chiếu.")
    if PILOT_MANIFEST.is_file():
        manifest = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
elif kind == "upload":
    up_pre = st.sidebar.file_uploader("CSV pre-label (YOLO/SAM)", type="csv")
    up_log = st.sidebar.file_uploader("correction_log.csv (tùy chọn)", type="csv")
    if up_pre is not None:
        pre_raw = read_csv_bytes(up_pre.getvalue())
    if up_log is not None:
        log_raw = read_csv_bytes(up_log.getvalue())
elif kind == "demo":
    pre_raw = pd.read_csv(SAMPLE_PRELABELS)
    if st.sidebar.checkbox("Kèm correction log MINH HỌA", value=True):
        log_raw = pd.read_csv(SAMPLE_LOG)

st.sidebar.header("Ngưỡng ưu tiên")
high_below = st.sidebar.slider("Ưu tiên CAO nếu điểm chất lượng <", 0, 100, int(DEFAULT_THRESHOLDS["high_below"]))
medium_below = st.sidebar.slider(
    "Ưu tiên VỪA nếu điểm chất lượng <", high_below, 100, max(high_below, int(DEFAULT_THRESHOLDS["medium_below"]))
)
thresholds = {"high_below": float(high_below), "medium_below": float(medium_below)}

st.sidebar.header("Ảnh")
if kind == "pilot":
    st.sidebar.caption(
        "Overlay YOLO/SAM được đọc trực tiếp từ `data/real/local/end_corrected.zip` (nếu có)."
        if PILOT_ZIP.is_file() else "Không thấy ZIP local — sẽ không hiển thị overlay."
    )
image_dir = st.sidebar.text_input(
    "Thư mục ảnh (tùy chọn)", placeholder=r"vd. ...\bdd100k\images\100k\train",
    help="Ảnh được tìm theo tên <image_id>.jpg/.png.",
)

# ---------------------------------------------------------------- Header + nhãn nguồn dữ liệu
st.title("M31 · Ước lượng công sức sửa nhãn tự động")
st.caption("BDD100K · Pre-label: YOLO (detection) / SAM (segmentation) · MVP")

if kind == "pilot":
    st.info(
        "**DỮ LIỆU PILOT THẬT** — 100 ảnh BDD100K, chỉ lớp *car*, YOLOv8n + SAM-B (box prompt từ YOLO). "
        "Thời gian sửa do nhóm ghi nhận. Đây là pilot nhỏ, **không** phải đánh giá trên toàn bộ BDD100K."
    )
    if manifest:
        with st.expander("Nguồn gốc dữ liệu pilot (manifest)"):
            st.json(manifest)
    if pilot_available and not pilot_log_available:
        st.warning(
            "Bản này **không kèm correction log thật** (`data/real/processed/pilot100_correction_log.csv`): "
            "file chỉ được đưa lên khi nhóm đồng ý công khai. Danh sách, ưu tiên và đặc trưng pre-label vẫn "
            "hiển thị; phần so sánh với thời gian sửa thật (tab ④) sẽ trống. Kết quả đối chiếu đã tính sẵn nằm "
            "trong `docs/01` và `docs/07`."
        )
elif kind == "upload":
    st.info("**DỮ LIỆU DO NGƯỜI DÙNG TẢI LÊN** — nguồn và độ tin cậy do người bàn giao chịu trách nhiệm.")
else:
    st.error(
        "**ĐANG DÙNG DỮ LIỆU MINH HỌA.** Các file trong `data/sample/` được sinh ngẫu nhiên bởi "
        "`scripts/make_sample_data.py`. Mọi điểm số, thời gian, MAE bên dưới **không phải** kết quả đo thật."
    )

if kind == "pilot" and not pilot_available:
    st.warning("Không tìm thấy `data/real/processed/pilot100_prelabels.csv` trên máy chủ này.")
    st.markdown(
        "**Cách khắc phục:**\n"
        "- Kiểm tra repository đã có thư mục `data/real/processed/` (file này được phép commit), hoặc\n"
        "- Trên máy có gói pilot: sao chép `end_corrected.zip` vào `data/real/local/` rồi chạy "
        "`python scripts/prepare_real_data.py`, sau đó tải lại trang.\n\n"
        "Hoặc chọn **Tải lên file thật** / **Dữ liệu MINH HỌA** ở thanh bên trái."
    )
    st.stop()

if pre_raw is None:
    st.info(
        "Hãy tải lên CSV pre-label ở thanh bên trái. Cột bắt buộc:\n\n"
        f"`{','.join(PRELABEL_COLUMNS)}`\n\n"
        f"Cột đặc trưng tùy chọn: `{','.join(OPTIONAL_PRELABEL_COLUMNS)}`\n\n"
        f"correction_log.csv (tùy chọn): `{','.join(CORRECTION_COLUMNS)}` (+ `status`, `notes`)"
    )
    st.stop()

pre_res = validate_prelabels(pre_raw)
show_validation(pre_res, "Pre-label")
if not pre_res.ok:
    st.stop()
prelabels = pre_res.df

report, log_df = None, None
if log_raw is not None:
    log_res = validate_correction_log(log_raw)
    show_validation(log_res, "Correction log")
    if log_res.ok:
        log_df = log_res.df
        report = evaluate(prelabels, log_df)

# ---------------------------------------------------------------- Mô hình thời gian
time_estimator = BaselineTimeEstimator()
use_calibrated = False
if report is not None and report.estimator is not None:
    use_calibrated = st.sidebar.toggle(
        "Dùng mô hình đã hiệu chỉnh từ correction log", value=False,
        help=f"Chỉ khả dụng cho task có ≥{MIN_NONZERO_FOR_FIT} ảnh có thời gian sửa > 0.",
    )
    if use_calibrated:
        time_estimator = report.estimator

if use_calibrated:
    st.success(f"Thời gian sửa: **{time_estimator.name}** cho task {list(report.estimator.coefs)}.")
else:
    st.warning(
        "Thời gian sửa dùng **baseline heuristic TẠM THỜI** (hằng số giả định trong `m31/baseline.py`), "
        "**chưa được huấn luyện**. "
        + ("Xem tab ④ để thấy sai số so với thời gian sửa thật." if report is not None else
           "Tải correction log để đối chiếu.")
    )
st.caption(
    "`predicted_quality_score` (0–100) là **ước lượng thử nghiệm** chỉ dựa trên đặc trưng pre-label. "
    "Không dùng ground truth hay thông tin sau khi sửa làm đầu vào."
)

result = estimate(prelabels, time_estimator, thresholds)
extra_cols = [c for c in OPTIONAL_PRELABEL_COLUMNS if c in result.columns]

tab_list, tab_detail, tab_compare, tab_limits = st.tabs([
    "② Danh sách & ưu tiên review", "③ Chi tiết ảnh & pre-label", "④ So sánh với thời gian sửa thật",
    "⑤ Giới hạn & kiểm tra rò rỉ",
])

# ---------------------------------------------------------------- Danh sách
with tab_list:
    c1, c2, c3 = st.columns(3)
    tasks = sorted(result["task"].unique())
    models = sorted(result["model"].unique())
    f_task = c1.multiselect("Tác vụ", tasks, default=tasks)
    f_model = c2.multiselect("Model", models, default=models)
    f_pri = c3.multiselect("Ưu tiên", list(PRIORITY_ORDER), default=list(PRIORITY_ORDER))
    view = result[result["task"].isin(f_task) & result["model"].isin(f_model) & result["priority"].isin(f_pri)]
    view = view.sort_values(
        ["priority", "estimated_correction_time_sec"],
        key=lambda s: s.map(PRIORITY_ORDER) if s.name == "priority" else -s,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số dòng (ảnh × tác vụ)", len(view))
    m2.metric("Ưu tiên cao", int((view["priority"] == "cao").sum()))
    m3.metric("Tổng thời gian sửa ước lượng (baseline)" if not use_calibrated else "Tổng thời gian sửa ước lượng",
              f"{view['estimated_correction_time_sec'].sum() / 60:.1f} phút")
    m4.metric("Điểm chất lượng TB (thử nghiệm)", f"{view['predicted_quality_score'].mean():.1f}" if len(view) else "–")

    st.dataframe(
        view[OUTPUT_COLUMNS + ["mean_confidence", "num_predictions", "low_confidence_count"]],
        width="stretch", hide_index=True,
        column_config={
            "predicted_quality_score": st.column_config.ProgressColumn(
                "Điểm chất lượng (thử nghiệm)", min_value=0, max_value=100, format="%.1f"
            ),
            "estimated_correction_time_sec": st.column_config.NumberColumn("Thời gian sửa ước lượng (s)", format="%.1f"),
            "priority": "Ưu tiên", "reason": "Lý do", "task": "Tác vụ",
        },
    )
    out = result[OUTPUT_COLUMNS + ["time_source"]].assign(data_source=kind)
    st.download_button(
        "Tải kết quả (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
        file_name={"demo": "m31_estimates_ILLUSTRATIVE.csv", "pilot": "m31_estimates_pilot100.csv"}.get(
            kind, "m31_estimates.csv"),
        mime="text/csv",
    )

# ---------------------------------------------------------------- Chi tiết
with tab_detail:
    keys = result["image_id"] + " | " + result["task"] + " | " + result["model"]
    choice = st.selectbox("Chọn ảnh", keys.tolist())
    row = result[keys == choice].iloc[0]

    left, right = st.columns([3, 2])
    with left:
        img = find_image(image_dir, row["image_id"])
        overlay = overlay_from_zip(row["image_id"], row["task"]) if kind == "pilot" else None
        if overlay is not None:
            st.image(overlay, caption=f"Overlay pre-label {row['model']} (từ ZIP local) · {row['image_id']}")
        elif img is not None:
            st.image(str(img), caption=row["image_id"])
        else:
            st.info("Chưa có ảnh để hiển thị (không có ZIP local / thư mục ảnh, hoặc không tìm thấy file).")
    with right:
        st.subheader(row["image_id"])
        st.write(f"**Tác vụ:** {TASK_VI[row['task']]}  \n**Model:** {row['model']}")
        a, b, c = st.columns(3)
        a.metric("Điểm chất lượng*", f"{row['predicted_quality_score']:.1f}")
        b.metric("Thời gian ước lượng", f"{row['estimated_correction_time_sec']:.0f} s")
        c.metric("Ưu tiên", row["priority"].upper())
        st.write(f"**Lý do:** {row['reason']}")
        st.caption(f"*Ước lượng thử nghiệm. Nguồn thời gian: {row['time_source']}.")
        st.markdown("**Đặc trưng pre-label (đầu vào duy nhất của dự đoán)**")
        feats = row[PRELABEL_COLUMNS[3:] + extra_cols].dropna()
        st.table(feats.astype(str).to_frame("giá trị"))
        if log_df is not None:
            m = log_df[(log_df["image_id"] == row["image_id"]) & (log_df["task"] == row["task"])
                       & (log_df["model"] == row["model"])]
            if len(m):
                label = {"pilot": "pilot thật", "demo": "MINH HỌA", "upload": "tải lên"}[kind]
                st.markdown(f"**Log sửa ({label}) — chỉ để đối chiếu, không dùng để dự đoán**")
                st.table(m[[c for c in m.columns if c not in ("image_id", "task", "model")]].T.astype(str)
                         .rename(columns=lambda _: "giá trị"))

# ---------------------------------------------------------------- So sánh với thời gian thật
with tab_compare:
    if report is None:
        st.info("Chưa có correction log nên **chưa đối chiếu và chưa đánh giá** được ước lượng.")
    else:
        if kind == "demo":
            st.error("Các con số dưới đây tính trên dữ liệu MINH HỌA — chỉ để kiểm tra cơ chế, không phải kết quả.")
        elif kind == "pilot":
            st.info("Các con số dưới đây tính trên **pilot thật 100 ảnh (lớp car)** — mẫu nhỏ, chỉ mang tính tham khảo.")
        a, b, c = st.columns(3)
        a.metric("Dòng trong correction log", report.n_log)
        b.metric("Dòng khớp với pre-label", report.n_matched)
        c.metric("Ảnh có thời gian sửa > 0", sum(report.n_nonzero.values()) if report.n_nonzero else "–")
        for n in report.notes:
            st.info(n)

        if report.baseline_mae:
            st.subheader("Sai số tuyệt đối trung bình (MAE, giây)")
            rows = []
            for task, mae in report.baseline_mae.items():
                rows.append({
                    "task": task,
                    "số dòng": int((report.matched["task"] == task).sum()) if task != "tổng" else report.n_matched,
                    "ảnh có sửa": report.n_nonzero.get(task, sum(report.n_nonzero.values())),
                    "MAE baseline": round(mae, 2),
                    "MAE ridge (5-fold CV)": round(report.calibrated_cv_mae[task], 2)
                    if task in report.calibrated_cv_mae else None,
                    "MAE mốc hằng số (median, CV)": round(report.constant_cv_mae[task], 2)
                    if task in report.constant_cv_mae else None,
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(
                f"MAE baseline: trên mọi dòng khớp (≥{MIN_ROWS_FOR_MAE}). MAE ridge và mốc hằng số: cross-validation, "
                f"khi task có ≥{MIN_ROWS_FOR_FIT} dòng. Một mô hình chỉ có ích nếu MAE thấp hơn **mốc hằng số**."
            )

        if report.quality_vs_time_spearman:
            st.markdown("**Tương quan Spearman: điểm chất lượng ↔ thời gian sửa thật** (kỳ vọng âm)")
            st.dataframe(
                pd.DataFrame([{"task": t, "spearman": round(v, 3)} for t, v in report.quality_vs_time_spearman.items()]),
                hide_index=True,
            )

        pv = priority_vs_correction(result, log_df)
        if not pv.empty:
            st.markdown("**Mức ưu tiên so với việc ảnh có thực sự cần sửa** (thời gian sửa thật > 0)")
            st.dataframe(pv, hide_index=True, width="stretch")
            st.caption("Nếu mức ưu tiên có ích, tỉ lệ cần sửa ở mức *cao* phải lớn hơn rõ rệt so với mức *thấp*.")

        if report.estimator is not None:
            st.markdown("**Hệ số mô hình hiệu chỉnh** (giây; `uncertainty = 1 - mean_confidence`)")
            st.dataframe(report.estimator.coef_table().round(2), hide_index=True)

        if report.matched is not None and "baseline_pred" in report.matched:
            st.markdown("**Thời gian sửa thật và ước lượng của baseline**")
            st.scatter_chart(report.matched, x="correction_time_sec", y="baseline_pred", color="task")
            comp = result[["image_id", "task", "model", "predicted_quality_score", "estimated_correction_time_sec",
                           "priority"]].merge(log_df, on=["image_id", "task", "model"])
            comp = comp.sort_values("correction_time_sec", ascending=False)
            st.markdown("**Từng ảnh: ước lượng và thực tế** (sắp theo thời gian sửa thật)")
            st.dataframe(comp, hide_index=True, width="stretch")

# ---------------------------------------------------------------- Giới hạn
with tab_limits:
    st.markdown("**Đầu vào được dùng để dự đoán** (đều có trước khi con người sửa):")
    st.code(", ".join(PRELABEL_COLUMNS[3:] + extra_cols))
    st.markdown(
        "**Không bao giờ dùng làm đầu vào:** `correction_time_sec`, `edits_*`, `status`, `notes`, "
        "ground truth và mọi chỉ số tính từ ground truth. Nếu các cột này xuất hiện trong file pre-label, "
        "ứng dụng tự loại bỏ và cảnh báo. Correction log chỉ được ghép vào ở tab ④ để đối chiếu."
    )
    st.markdown(
        "**Giới hạn hiện tại**\n"
        "- Baseline thời gian sửa dùng hằng số giả định, chưa được học từ dữ liệu.\n"
        "- Điểm chất lượng là công thức cố định đặt trước khi có dữ liệu thật; không được tinh chỉnh trên pilot.\n"
        f"- Mô hình hiệu chỉnh chỉ được bật khi mỗi task có ≥{MIN_NONZERO_FOR_FIT} ảnh có thời gian sửa > 0.\n"
        "- Pilot chỉ có 100 ảnh, 1 lớp (car); kết quả không đại diện cho toàn bộ BDD100K.\n"
        "- Với SAM, confidence là của box YOLO dùng làm prompt, không phải điểm riêng của mask.\n"
        "- Thời gian sửa do nhóm ghi nhận; không xác minh độc lập được từ file.\n"
        "- Đã thử 29 đặc trưng box/mask mở rộng (docs/07_phan_tich_dac_trung.md): không mô hình nào thắng "
        "mốc 'luôn đoán 0 giây' trong cross-validation, AUC xếp hạng ≈ 0,56 (không khác ngẫu nhiên) — "
        "nên chưa dùng chúng để dự đoán."
    )
