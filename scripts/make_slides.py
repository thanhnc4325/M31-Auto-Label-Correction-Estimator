"""Tạo docs/M31_slides_team.pptx (8 slide) từ kết quả pilot đã lưu trong data/real/processed/.

Cần python-pptx (KHÔNG có trong requirements.txt vì app không cần):  pip install python-pptx
Chạy:  python scripts/make_slides.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "real" / "processed"
OUT = ROOT / "docs" / "M31_slides_team.pptx"

ASPHALT = RGBColor(0x22, 0x27, 0x2E)
LANE = RGBColor(0xF2, 0xB7, 0x05)
SLATE = RGBColor(0x4A, 0x55, 0x60)
MUTED = RGBColor(0x6B, 0x75, 0x80)
CARD = RGBColor(0xF1, 0xF3, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
STEEL = RGBColor(0x6C, 0x8E, 0xA4)
HEAD, BODY = "Cambria", "Calibri"
W, H = Inches(13.333), Inches(7.5)


def text(slide, x, y, w, h, paras, size=16, color=ASPHALT, font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None):
    """paras: str | list[str | list[(text, {opts})]]."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if isinstance(paras, str):
        paras = [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(6)
        runs = p if isinstance(p, list) else [(p, {})]
        for t, o in runs:
            r = para.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.color.rgb = o.get("color", color)
    return tb


def box(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    if name:
        s.name = name
    return s


def badge(slide, x, y, label, d=Inches(0.55), fill=LANE, color=ASPHALT):
    c = box(slide, x, y, d, d, fill, MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = BODY, Pt(16), True, color
    return c


def background(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def title(slide, t, color=ASPHALT):
    text(slide, Inches(0.6), Inches(0.45), Inches(12.1), Inches(0.9), t, size=34, font=HEAD, bold=True,
         color=color, name="Title")


def build() -> None:
    reg = pd.read_csv(PROC / "pilot100_cv_regression.csv").set_index(["task", "model"])
    rank = pd.read_csv(PROC / "pilot100_cv_ranking.csv").set_index(["task", "model"])
    summ = json.loads((PROC / "pilot100_analysis_summary.json").read_text(encoding="utf-8"))
    mae = lambda t, m: reg.loc[(t, m), "mae_mean"]  # noqa: E731
    vi = lambda v, d=2: f"{v:.{d}f}".replace(".", ",")  # noqa: E731

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]

    # 1. Tiêu đề ---------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, ASPHALT)
    for i in range(6):  # vạch kẻ đường làm hoạ tiết
        box(s, Inches(0.6 + i * 2.15), Inches(6.35), Inches(1.2), Inches(0.12), LANE, MSO_SHAPE.RECTANGLE)
    text(s, Inches(0.6), Inches(1.5), Inches(12), Inches(1.2), "M31 · Auto-Label Correction Estimator",
         size=44, font=HEAD, bold=True, color=WHITE, name="Title")
    text(s, Inches(0.6), Inches(2.8), Inches(10.5), Inches(1.2),
         "Ước lượng ảnh nào cần review trước và tốn bao lâu để sửa nhãn do YOLO / SAM gán tự động",
         size=22, color=RGBColor(0xD5, 0xDA, 0xE0))
    text(s, Inches(0.6), Inches(4.3), Inches(11), Inches(0.5),
         [[("Pilot thật: ", {"bold": True, "color": LANE}),
           ("100 ảnh BDD100K · lớp car · YOLOv8n + SAM-B — không phải đánh giá trên toàn bộ BDD100K",
            {"color": WHITE})]], size=16)
    text(s, Inches(0.6), Inches(5.05), Inches(12.1), Inches(0.3), "NHÓM THỰC HIỆN · 4 THÀNH VIÊN",
         size=13.5, color=LANE, bold=True)
    text(s, Inches(0.6), Inches(5.38), Inches(5.6), Inches(0.72),
         ["Nguyễn Công Thành · 2A202602219 (Trưởng nhóm)",
          "Lưu Quang Hùng · 2A202602142 (TV2)"], size=12.5, color=RGBColor(0xD5, 0xDA, 0xE0))
    text(s, Inches(6.65), Inches(5.38), Inches(6.0), Inches(0.72),
         ["Kim Nguyên Khôi · 2A202602116 (TV1)",
          "Nguyễn Văn Trọng · 2A202602276 (TV3)"], size=12.5, color=RGBColor(0xD5, 0xDA, 0xE0))
    s.notes_slide.notes_text_frame.text = (
        "Giới thiệu M31: công cụ ước lượng công sức sửa nhãn tự động. Nói rõ ngay từ đầu: đây là pilot 100 ảnh, "
        "chỉ lớp car, không phải đánh giá trên toàn bộ BDD100K.")

    # 2. Bài toán --------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Pre-label nhanh, nhưng con người vẫn phải sửa")
    steps = [("1", "Pre-label", "YOLO vẽ box, SAM tạo mask cho từng ảnh"),
             ("2", "M31 ước lượng", "Điểm chất lượng, thời gian sửa, mức ưu tiên, lý do"),
             ("3", "Người review & sửa", "Thêm / xoá / chỉnh box và mask"),
             ("4", "Đối chiếu", "So ước lượng với thời gian sửa thật")]
    for i, (n, h, d) in enumerate(steps):
        x = Inches(0.6 + i * 3.1)
        box(s, x, Inches(1.8), Inches(2.85), Inches(2.4), CARD, name=f"Step{n}")
        badge(s, x + Inches(0.3), Inches(2.05), n)
        text(s, x + Inches(0.3), Inches(2.8), Inches(2.35), Inches(0.5), h, size=20, bold=True)
        text(s, x + Inches(0.3), Inches(3.3), Inches(2.35), Inches(0.9), d, size=15, color=SLATE)
    box(s, Inches(0.6), Inches(4.75), Inches(12.1), Inches(1.75), ASPHALT, name="Constraint")
    text(s, Inches(0.95), Inches(4.95), Inches(11.4), Inches(1.4),
         [[("Ràng buộc chống rò rỉ: ", {"bold": True, "color": LANE}),
           ("chỉ dùng thông tin có TRƯỚC khi người sửa bắt đầu (confidence, box, mask).", {"color": WHITE})],
          [("Thời gian sửa, thao tác sửa, status, notes và ground truth chỉ dùng để đánh giá.",
            {"color": RGBColor(0xD5, 0xDA, 0xE0)})]], size=17, anchor=MSO_ANCHOR.MIDDLE)
    s.notes_slide.notes_text_frame.text = (
        "Bài toán: biết ảnh nào nên review trước và dự trù công sửa. Bước 2 là M31. Nhấn mạnh ràng buộc: "
        "không dùng thông tin chỉ có sau khi sửa làm đầu vào.")

    # 3. Giải pháp & kiến trúc -------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Pipeline tách riêng đặc trưng và nhãn đánh giá")
    nodes = [(0.6, 2.0, "ZIP pilot (local)", "Dự đoán, mask, overlay, log — không commit"),
             (4.0, 1.4, "CSV đặc trưng", "29 đặc trưng box/mask, chỉ trước khi sửa"),
             (4.0, 3.75, "CSV correction log", "Tách riêng, chỉ để đối chiếu"),
             (7.4, 2.0, "M31 (Python)", "Kiểm tra schema, loại cột GT / sau-sửa, ước lượng, CV"),
             (10.4, 2.0, "App Streamlit", "5 tab: dữ liệu, ưu tiên, chi tiết, so sánh, giới hạn")]
    for x, y, h, d in nodes:
        dark = h.startswith("App")
        box(s, Inches(x), Inches(y), Inches(2.6 if x < 10 else 2.3), Inches(1.9),
            ASPHALT if dark else CARD, name=h)
        text(s, Inches(x + 0.25), Inches(y + 0.25), Inches(2.1 if x < 10 else 1.85), Inches(0.5), h, size=17,
             bold=True, color=WHITE if dark else ASPHALT)
        text(s, Inches(x + 0.25), Inches(y + 0.8), Inches(2.1 if x < 10 else 1.85), Inches(1.0), d, size=13,
             color=RGBColor(0xD5, 0xDA, 0xE0) if dark else SLATE)
    for (x1, y1, x2, y2) in [(3.2, 2.6, 4.0, 2.35), (3.2, 3.3, 4.0, 4.7), (6.6, 2.35, 7.4, 2.95),
                             (6.6, 4.7, 7.4, 3.3), (10.0, 2.95, 10.4, 2.95)]:
        c = s.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        c.line.color.rgb = MUTED
        c.line.width = Pt(2)
        c.line._get_or_add_ln().append(_arrow_end())
    text(s, Inches(0.6), Inches(6.1), Inches(12.1), Inches(0.9),
         [[("Kiểm thử: ", {"bold": True}),
           ("20 unit test, gồm test chạy script đặc trưng trên ZIP không có log/GT và test "
            "\"ghép log vào không làm đổi dự đoán\".", {"color": SLATE})]], size=15)
    s.notes_slide.notes_text_frame.text = (
        "Kiến trúc: dữ liệu local không lên GitHub. Đặc trưng và log nằm ở hai file riêng; log chỉ đi vào "
        "phần đánh giá. App có 5 tab theo luồng demo.")

    # 4. Dữ liệu pilot ---------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Pilot nhỏ: chỉ 13/100 ảnh mỗi task cần sửa")
    n_corr = summ["n_corrected_per_task"]
    stats = [("100", "ảnh BDD100K, lớp car"), ("555", "box YOLO = 555 mask SAM"),
             (f"{n_corr['detection']}/100", "ảnh cần sửa mỗi task (cùng 13 ảnh)"),
             ("1,17 s", "thời gian sửa TB / ảnh — YOLO"), ("1,69 s", "thời gian sửa TB / ảnh — SAM")]
    for i, (v, l) in enumerate(stats):
        x = Inches(0.6 + (i % 3) * 4.1)
        y = Inches(1.7 if i < 3 else 3.95)
        box(s, x, y, Inches(3.8), Inches(2.0), CARD)
        text(s, x + Inches(0.3), y + Inches(0.25), Inches(3.3), Inches(0.9), v, size=44, font=HEAD, bold=True)
        text(s, x + Inches(0.3), y + Inches(1.25), Inches(3.3), Inches(0.6), l, size=15, color=SLATE)
    x, y = Inches(0.6 + 2 * 4.1), Inches(3.95)
    box(s, x, y, Inches(3.8), Inches(2.0), ASPHALT)
    text(s, x + Inches(0.3), y + Inches(0.3), Inches(3.3), Inches(1.5),
         [[("Lưu ý", {"bold": True, "color": LANE})],
          [("Confidence của SAM là confidence box YOLO dùng làm prompt. Không có thao tác \"thêm\" nào trong log.",
            {"color": WHITE})]], size=14)
    text(s, Inches(0.6), Inches(6.3), Inches(12.1), Inches(0.6),
         "Nguồn: end_corrected.zip do nhóm bàn giao; thời gian sửa do nhóm ghi nhận, không kiểm chứng độc lập.",
         size=12, color=MUTED)
    s.notes_slide.notes_text_frame.text = (
        "Dữ liệu: 100 ảnh, 555 box. 87 ảnh mỗi task có thời gian sửa bằng 0. Hai task có cùng 13 ảnh cần sửa "
        "và dùng chung box, nên không phải hai bằng chứng độc lập.")

    # 5. Kết quả MAE -----------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Chưa mô hình nào thắng \"luôn đoán 0 giây\"")
    models = [("zero", "Luôn đoán 0 s"), ("ridge_n_boxes", "Ridge: số box"),
              ("ridge_A", "Ridge: nhóm A"), ("hurdle_A", "Hai bước: logistic")]
    cd = CategoryChartData()
    cd.categories = [m[1] for m in models]
    cd.add_series("Detection (YOLO)", [round(mae("detection", m[0]), 3) for m in models])
    cd.add_series("Segmentation (SAM)", [round(mae("segmentation", m[0]), 3) for m in models])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.5), Inches(7.9), Inches(5.0), cd)
    ch = gf.chart
    ch.has_title = True
    ch.chart_title.text_frame.text = "MAE (giây), CV 5-fold phân tầng × 50 lần — thấp hơn là tốt hơn"
    tp = ch.chart_title.text_frame.paragraphs[0].runs[0].font
    tp.size, tp.name, tp.bold, tp.color.rgb = Pt(13), BODY, False, SLATE
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size, ch.legend.font.name = Pt(12), BODY
    for ser, col in zip(ch.series, [LANE, STEEL]):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = col
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.has_data_labels = True
    pl.data_labels.number_format = "0.00"
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    pl.data_labels.font.size, pl.data_labels.font.name = Pt(12), BODY
    ch.value_axis.has_major_gridlines = True
    ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE3, 0xE6, 0xEA)
    ch.value_axis.tick_labels.font.size = ch.category_axis.tick_labels.font.size = Pt(12)
    ch.value_axis.tick_labels.font.name = ch.category_axis.tick_labels.font.name = BODY
    ch.value_axis.tick_labels.font.color.rgb = ch.category_axis.tick_labels.font.color.rgb = SLATE
    ch.value_axis.format.line.fill.background()
    pct = max(reg.loc[(t, m), "pct_repeats_beat_zero"] for t in ["detection", "segmentation"]
              for m in ["ridge_n_boxes", "ridge_A", "ridge_AB", "hurdle_A", "hurdle_AB"])
    box(s, Inches(8.9), Inches(1.5), Inches(3.8), Inches(5.0), CARD)
    text(s, Inches(9.2), Inches(1.75), Inches(3.2), Inches(4.6),
         [[(f"{pct:.0%}", {"size": 44, "font": HEAD, "bold": True})],
          [("số lần lặp CV mà một mô hình bất kỳ có MAE thấp hơn \"đoán 0\"", {"color": SLATE})],
          [(" ", {"size": 8})],
          [("Baseline M31 hiện tại: ", {"bold": True}),
           (f"{vi(mae('detection', 'm31_baseline'), 1)} s / {vi(mae('segmentation', 'm31_baseline'), 1)} s — "
            "ước lượng cao gấp 30–40 lần thực tế.", {"color": SLATE})],
          [("Ridge nhóm A dùng 20 (detection) / 25 (segmentation) đặc trưng; hai bước = xác suất logistic × "
            "thời gian sửa TB.", {"color": SLATE})],
          [("MAE của chính \"đoán 0\" dao động 0,74–3,93 s giữa các fold (2–3 ảnh sửa mỗi fold).",
            {"color": SLATE})]], size=15)
    s.notes_slide.notes_text_frame.text = (
        "Kết quả chính: không mô hình nào có MAE thấp hơn việc luôn đoán 0 giây, ở bất kỳ lần lặp nào. "
        "Baseline M31 sai thang đo nên không đưa vào biểu đồ. Kết quả theo fold không ổn định vì mỗi fold chỉ "
        "có 2–3 ảnh cần sửa.")

    # 6. Xếp hạng ---------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Xếp hạng ảnh cần sửa chưa khác ngẫu nhiên")
    cards = [
        (vi(rank.loc[("detection", "logistic_A"), "auc_mean"]), "AUC logistic — detection",
         f"p hoán vị = {vi(rank.loc[('detection', 'logistic_A'), 'perm_p'])}"),
        (vi(rank.loc[("segmentation", "logistic_A"), "auc_mean"]), "AUC logistic — segmentation",
         f"p hoán vị = {vi(rank.loc[('segmentation', 'logistic_A'), 'perm_p'])}"),
        (vi(rank.loc[("detection", "m31_quality_score (thấp → cần sửa)"), "auc_mean"]),
         "AUC điểm chất lượng M31", "công thức cố định, không học"),
    ]
    for i, (v, l, sub) in enumerate(cards):
        x = Inches(0.6 + i * 4.1)
        box(s, x, Inches(1.8), Inches(3.8), Inches(2.7), CARD)
        text(s, x + Inches(0.3), Inches(2.1), Inches(3.3), Inches(1.0), v, size=54, font=HEAD, bold=True)
        text(s, x + Inches(0.3), Inches(3.3), Inches(3.3), Inches(0.45), l, size=15, bold=True)
        text(s, x + Inches(0.3), Inches(3.8), Inches(3.3), Inches(0.45), sub, size=14, color=SLATE)
    box(s, Inches(0.6), Inches(4.95), Inches(12.1), Inches(1.75), CARD)
    badge(s, Inches(0.9), Inches(5.25), "?")
    text(s, Inches(1.7), Inches(5.2), Inches(10.7), Inches(1.4),
         [[("Gợi ý cần kiểm tra trên dữ liệu mới (khám phá, chưa phải kết quả): ", {"bold": True}),
           ("box chồng nhau nhiều (AUC 0,70) và mask lấp box ít (≈ 0,67). Đã thử 21–28 đặc trưng trên cùng "
            "13 ca, nên 1–2 tín hiệu có thể chỉ do ngẫu nhiên.", {"color": SLATE})],
          [("AUC 0,5 = ngẫu nhiên; tiêu chí tích hợp đặt trước: AUC ≥ 0,70 và p < 0,05.", {"color": MUTED,
                                                                                        "size": 13})]],
         size=16)
    s.notes_slide.notes_text_frame.text = (
        "Về xếp hạng: AUC khoảng 0,56, kiểm định hoán vị không có ý nghĩa. Hai tín hiệu đơn biến chỉ là giả "
        "thuyết vì đã thử nhiều đặc trưng trên cùng 13 ca.")

    # 6b. So sánh model -----------------------------------------------------------
    model_comparison_slide(prs, blank)

    # 7. Giới hạn ---------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Giới hạn cần nói rõ")
    lims = [("Mẫu nhỏ", "100 ảnh, 1 lớp, 13 ca sửa mỗi task — không đại diện BDD100K"),
            ("Baseline giả định", "Hằng số thời gian sửa đặt tay; điểm chất lượng chưa kiểm chứng"),
            ("SAM", "Không có confidence riêng; dùng confidence box YOLO"),
            ("Đo lường", "Thời gian sửa do nhóm ghi; không có đánh giá với ground truth"),
            ("Ngưỡng do M31 chọn", "Confidence thấp < 0,5; box nhỏ < 32×32 px"),
            ("Quyết định", "Không bật mô hình hiệu chỉnh; không tuyên bố độ chính xác")]
    for i, (h, d) in enumerate(lims):
        x = Inches(0.6 + (i % 2) * 6.15)
        y = Inches(1.65 + (i // 2) * 1.6)
        box(s, x, y, Inches(5.95), Inches(1.4), CARD)
        badge(s, x + Inches(0.3), y + Inches(0.42), str(i + 1))
        text(s, x + Inches(1.1), y + Inches(0.2), Inches(4.6), Inches(0.45), h, size=18, bold=True)
        text(s, x + Inches(1.1), y + Inches(0.65), Inches(4.6), Inches(0.7), d, size=14, color=SLATE)
    s.notes_slide.notes_text_frame.text = "Đọc nhanh 6 giới hạn; nhấn mạnh quyết định không tuyên bố độ chính xác."

    # 8. Bước tiếp theo ---------------------------------------------------------
    s = prs.slides.add_slide(blank)
    background(s, ASPHALT)
    title(s, "Bước tiếp theo", color=WHITE)
    nxt = [("Thêm dữ liệu sửa", "≥ 50 ca cần sửa mỗi task để fold kiểm tra không chỉ có 2–3 ca"),
           ("Đăng ký trước mô hình nhỏ", "Logistic trên chồng lấn box, độ lấp mask, số box — đánh giá trên ảnh mới"),
           ("Đặc trưng ảnh", "Nếu có ảnh gốc: ban đêm, mưa, độ sáng — vẫn chỉ dùng thông tin trước khi sửa")]
    for i, (h, d) in enumerate(nxt):
        y = Inches(1.8 + i * 1.45)
        badge(s, Inches(0.6), y, str(i + 1))
        text(s, Inches(1.45), y - Inches(0.05), Inches(11), Inches(0.5), h, size=20, bold=True, color=WHITE)
        text(s, Inches(1.45), y + Inches(0.45), Inches(11), Inches(0.6), d, size=15,
             color=RGBColor(0xD5, 0xDA, 0xE0))
    for i in range(6):
        box(s, Inches(0.6 + i * 2.15), Inches(6.35), Inches(1.2), Inches(0.12), LANE, MSO_SHAPE.RECTANGLE)
    s.notes_slide.notes_text_frame.text = (
        "Kết: M31 hiện là công cụ minh bạch và một khung đánh giá trung thực; muốn có mô hình hữu ích cần thêm "
        "dữ liệu sửa và đánh giá trên ảnh mới.")

    # Ghi đè metadata mặc định của template python-pptx (tác giả thư viện, ngày 2013).
    from datetime import datetime
    cp = prs.core_properties
    cp.title = "M31 · Auto-Label Correction Estimator"
    cp.subject = "Pilot 100 ảnh BDD100K"
    cp.author = cp.last_modified_by = "Nhóm M31"
    cp.comments = ""
    cp.created = cp.modified = datetime.now()
    cp.revision = 1
    prs.save(OUT)
    print(f"Đã ghi {OUT}")


MODEL_COMPARISON_NOTES = (
    "So sánh model (pilot 100 ảnh BDD100K, lớp car, RTX 2060): đã thử 22 phiên bản YOLO và 11 phiên bản SAM trên cùng "
    "tập pilot, với cấu hình thống nhất trong từng nhóm model. Với cùng box prompt YOLOv8n, SAM2-T/S chạy nhanh hơn "
    "SAM-B khoảng 2,2–2,5 lần; IoU trên 92 ảnh có prompt cao hơn nhẹ (+0,012…+0,016), nhưng kết quả chỉ mang tính "
    "khám phá. Chất lượng detection chưa xếp hạng được một cách đáng tin cậy vì chỉ 14 ảnh (126 box) có nhãn. Nhóm giữ "
    "YOLOv8n + SAM-B làm mặc định; các ứng viên cần được kiểm chứng trên tập ảnh mới có nhãn.\n\n"
    "Chi tiết phương pháp: YOLO dùng imgsz 640, conf 0,25, NMS theo lớp IoU 0,7, lọc lớp car; precision/recall chấm trên "
    "14 ảnh có nhãn box (126 box), 86 ảnh còn lại không chấm. SAM được đánh giá với cùng 555 box prompt từ YOLOv8n; "
    "IoU so với nhãn segmentation BDD100K (car), chỉ số chính trên 92 ảnh có prompt. Khoảng tin cậy bootstrap theo ảnh. "
    "Có nhiều phép so sánh trên mẫu nhỏ, chưa hiệu chỉnh, nên không có model nào được coi là tốt nhất; kết quả không "
    "đại diện cho toàn bộ BDD100K. Báo cáo: docs/11, docs/12, docs/13."
)


def model_comparison_slide(prs, layout):
    """Slide tóm tắt so sánh model (số liệu từ docs/12–13; kết quả khám phá)."""
    s = prs.slides.add_slide(layout)
    background(s, WHITE)
    title(s, "So sánh model trên pilot: giữ mặc định")
    text(s, Inches(0.6), Inches(1.3), Inches(12.1), Inches(0.5),
         "Pilot 100 ảnh BDD100K · lớp car · RTX 2060 · 22 phiên bản YOLO, 11 phiên bản SAM, cấu hình thống nhất trong từng nhóm",
         size=15, color=SLATE)
    cards = [
        ("2,2–2,5×", "SAM2-T/S nhanh hơn SAM-B",
         "Cùng box prompt YOLOv8n cho mọi SAM"),
        ("+0,012…+0,016", "IoU cao hơn nhẹ so với SAM-B",
         "Trên 92 ảnh có prompt · chỉ mang tính khám phá"),
        ("14 ảnh", "Detection chưa xếp hạng được",
         "Chỉ 14 ảnh (126 box) có nhãn"),
    ]
    for i, (v, l, sub) in enumerate(cards):
        x = Inches(0.6 + i * 4.1)
        box(s, x, Inches(2.0), Inches(3.8), Inches(2.55), CARD, name=f"ModelCard{i + 1}")
        text(s, x + Inches(0.3), Inches(2.3), Inches(3.3), Inches(0.8), v, size=28, font=HEAD, bold=True)
        text(s, x + Inches(0.3), Inches(3.25), Inches(3.3), Inches(0.5), l, size=16, bold=True)
        text(s, x + Inches(0.3), Inches(3.75), Inches(3.3), Inches(0.7), sub, size=13, color=SLATE)
    box(s, Inches(0.6), Inches(4.95), Inches(12.1), Inches(1.35), ASPHALT, name="ModelDecision")
    text(s, Inches(0.95), Inches(5.1), Inches(11.4), Inches(1.05),
         [[("Giữ YOLOv8n + SAM-B làm mặc định. ", {"bold": True, "color": LANE}),
           ("Các ứng viên cần được kiểm chứng trên tập ảnh mới có nhãn.", {"color": WHITE})],
          [("Không phải kết quả đại diện cho toàn bộ BDD100K; không có model nào được coi là tốt nhất.",
            {"color": RGBColor(0xD5, 0xDA, 0xE0), "size": 14})]],
         size=17, anchor=MSO_ANCHOR.MIDDLE)
    s.notes_slide.notes_text_frame.text = MODEL_COMPARISON_NOTES
    return s


def _arrow_end():
    from pptx.oxml.ns import qn
    from lxml import etree
    el = etree.SubElement(etree.Element(qn("a:dummy")), qn("a:tailEnd"))
    el.set("type", "triangle")
    return el


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    build()
