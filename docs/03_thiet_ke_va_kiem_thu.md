# Thiết kế và kiểm thử

## Quyết định thiết kế

| Quyết định | Lý do |
|---|---|
| Tách lõi `m31/` khỏi giao diện | Kiểm thử không cần Streamlit; thay mô hình không phải sửa UI |
| Baseline minh bạch thay vì mô hình học | Khi xây MVP chưa có dữ liệu sửa thật; mọi hằng số được ghi là giả định |
| Công thức điểm chất lượng cố định, không chỉnh trên pilot | Pilot là dữ liệu đánh giá duy nhất; chỉnh trên nó sẽ cho kết quả lạc quan giả |
| Báo MAE của **mốc hằng số** bên cạnh model | Pilot có 87% ảnh với thời gian sửa bằng 0; MAE thấp chưa chắc là tốt |
| Mô hình hiệu chỉnh chỉ bật khi ≥ 20 ca sửa mỗi task | 13 ca/task không đủ để học ổn định; ngưỡng này được đặt theo hướng thận trọng |
| Loại tự động cột GT và cột sau-sửa khỏi file pre-label | Chống rò rỉ ngay cả khi người bàn giao gửi nhầm file |
| Đọc overlay trực tiếp từ ZIP | Không giải nén 855 PNG; ảnh/mask nằm local và bị `.gitignore` |
| Ngưỡng confidence thấp 0.5 ghi trong manifest | Nhóm không bàn giao ngưỡng; ghi rõ để người khác tái lập |
| `prepare_real_data.py` không ghi đè nếu thiếu `--overwrite` | Tránh mất dữ liệu đã xử lý |

## Kiểm tra rò rỉ dữ liệu

| Thông tin | Có trước khi sửa? | Dùng làm đầu vào? |
|---|---|---|
| confidence, số dự đoán, toạ độ box, mask SAM, kích thước ảnh | Có | **Có** |
| `inference_time_sec` | Có | Không (là tốc độ máy, không phải chất lượng nhãn) |
| `human_correction_time_sec`, `action_add/delete/modify` | Không | **Không**, chỉ để đánh giá |
| `status`, `notes` | Không | **Không**, chỉ hiển thị khi đối chiếu |
| `image_groundtruth_pairs_100.csv` (đường dẫn GT) | — | **Không đọc** |

Cơ chế bảo vệ:
1. `prepare_real_data.py` ghi đặc trưng và log ra **hai file riêng**.
2. `validate_prelabels()` loại các cột có token `gt`, `iou`, `map`, … hoặc `correction`, `action`, `human`, `status`, `notes`, … kèm cảnh báo.
3. `estimate()` chỉ nhận bảng pre-label; log chỉ được ghép trong `calibration`.
4. Các test ở mục dưới khẳng định 3 điểm trên.

Lưu ý: ghi chú của người sửa có nhắc đến ground truth (vd. "GT nhãn 255"); cột `notes` chỉ được hiển thị ở phần đối chiếu.

## Kiểm thử đã thực hiện

### Unit test (`tests/test_core.py`)

Chạy: `.venv\Scripts\python.exe -m unittest discover tests`. Kết quả lần chạy gần nhất (2026-10-18): **20/20 pass**.

| Nhóm | Test | Kiểm tra |
|---|---|---|
| Schema | `test_valid`, `test_missing_column`, `test_task_alias_and_bad_task`, `test_low_conf_exceeds_predictions`, `test_zero_predictions_allow_empty_confidence` | Hợp lệ hoá đầu vào |
| Schema | `test_ground_truth_columns_are_dropped` | Cột GT bị loại |
| Rò rỉ | `test_post_correction_columns_dropped_from_prelabels` | Cột sau-sửa bị loại khỏi pre-label |
| Rò rỉ | `test_estimates_do_not_depend_on_correction_log` | Ghép log (thời gian 999 s) vào không làm đổi dự đoán |
| Schema | `test_optional_feature_columns_kept` | Cột đặc trưng tùy chọn được giữ |
| Ước lượng | `test_outputs`, `test_zero_predictions_high_priority` | Miền giá trị, thứ tự điểm, ưu tiên cho ảnh không có dự đoán |
| Hiệu chỉnh | `test_too_few_rows_no_mae`, `test_unmatched_rows_reported`, `test_calibration_recovers_linear_relation` | MAE, ghép khoá, CV trên dữ liệu tổng hợp tuyến tính |
| Hiệu chỉnh | `test_calibrated_model_disabled_when_few_nonzero` | Không bật mô hình khi < 20 ca sửa |
| Hiệu chỉnh | `test_priority_vs_correction_counts` | Bảng ưu tiên × cần sửa đếm đúng |
| Dữ liệu thật | `test_build_from_tiny_zip` | Script xử lý trên ZIP tự tạo: đặc trưng, mask fill, ảnh không có dự đoán, không lọt cột log |
| Đặc trưng | `test_build_features_without_logs_or_ground_truth` | `build_features.py` chạy được trên ZIP không có log/GT; IoU, fill, tràn box, chồng mask tính đúng |
| Đặc trưng | `test_stratified_folds_spread_corrections`, `test_auc` | Fold phân tầng chia đều 13 ca sửa (2–3 mỗi fold); AUC đúng ở các trường hợp biên |

### Kiểm tra trên dữ liệu pilot

- `prepare_real_data.py`: 200 dòng pre-label, 200 dòng log; **0** ảnh lệch giữa số dự đoán đếm được và `num_predictions` trong log.
- Bản sao ZIP có SHA-256 trùng bản gốc (`db90d054…29bb`); bản gốc giữ nguyên kích thước và thời gian sửa đổi.
- Kết quả đánh giá: xem [01_bai_toan_giai_phap_gioi_han.md](01_bai_toan_giai_phap_gioi_han.md#4-kết-quả-trên-pilot-đo-ngày-2026-10-18).

### Kiểm tra giao diện (Streamlit `AppTest`, chạy không cần trình duyệt)

- Ba nguồn dữ liệu (pilot, tải lên, minh họa): không có exception.
- Tab chi tiết với pilot: hiển thị overlay đọc từ ZIP.
- Chạy `app.py` trong thư mục không có dữ liệu local: tự chuyển sang "Tải lên" và hiện hướng dẫn, không lỗi.
- Chưa có kiểm thử giao diện tự động trong `tests/`; các bước trên được chạy thủ công.

### Phân tích đặc trưng mở rộng

Xem [07_phan_tich_dac_trung.md](07_phan_tich_dac_trung.md): CV phân tầng lặp 50 lần, kiểm định hoán vị 200 lần,
`mask_union_coverage` được đối chiếu với mask tổng hợp trong ZIP (lệch ≤ 5·10⁻⁷).

### Lỗi đã gặp và cách sửa

- `Series.corr(method="spearman")` cần scipy, mà `.venv` không có. Đã thay bằng tương quan Pearson trên thứ hạng (`m31/calibration.py`).
