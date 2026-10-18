# Dữ liệu thật

## Pilot 100 ảnh (đã tích hợp)

| Thư mục | Nội dung | Commit? |
|---|---|---|
| `local/end_corrected.zip` | Bản sao gói pilot (YOLO/SAM, mask, overlay, correction log). SHA-256 ghi trong manifest. | **Không** (.gitignore) |
| `processed/pilot100_prelabels.csv` | 200 dòng (100 ảnh × 2 tác vụ), chỉ đặc trưng pre-label | Có (xem .gitignore để tắt) |
| `processed/pilot100_correction_log.csv` | 200 dòng log sửa, chỉ để đối chiếu; nhóm đã đồng ý công khai ngày 18/10/2026 | Có |
| `processed/pilot100_manifest.json` | Nguồn, hash, ngưỡng, số dòng | Có |
| `processed/pilot100_features.csv` + `_features_dictionary.csv` | 29 đặc trưng mở rộng (chỉ trước khi sửa) + công thức/nguồn/số thiếu | Có |
| `processed/pilot100_cv_*.csv`, `pilot100_univariate_auc.csv`, `pilot100_analysis_summary.json` | Kết quả phân tích (docs/07) | Có |

Tạo lại: `.venv\Scripts\python.exe scripts\prepare_real_data.py` (thêm `--overwrite` để ghi đè bản đã xử lý).
Gói gốc bên ngoài project không bị sửa.

### Ánh xạ cột

| M31 | Nguồn trong ZIP |
|---|---|
| `num_predictions`, `mean_confidence`, `min_confidence` | `yolo_predictions_100.csv` / `sam_predictions_100.csv` (`confidence`) |
| `low_confidence_count` | số dự đoán có confidence < **0.5** (ngưỡng do M31 chọn, không được bàn giao) |
| `small_box_count`, `mean_box_area_ratio` | toạ độ box; "nhỏ" = diện tích < 32×32 px |
| `mean_mask_fill_ratio`, `empty_mask_count` | PNG trong `sam_masks/` so với box prompt |
| `image_width`, `image_height` | kích thước overlay `yolo_overlays/*.png` |
| `correction_time_sec`, `edits_added/removed/changed`, `status`, `notes` | `correction_log_{yolo,sam}_100.csv` (`human_correction_time_sec`, `action_*`) |

Không đọc: `image_groundtruth_pairs_100.csv` (ground truth), `inference_time_sec` (thời gian chạy model, không phải đặc trưng chất lượng).

## Bàn giao dữ liệu mới

- `task`: `detection` hoặc `segmentation`; `image_id`: tên ảnh BDD100K không có đuôi, giống nhau ở mọi file.
- `mean_confidence` trong [0, 1]; ảnh không có dự đoán: `num_predictions = 0`, để trống `mean_confidence`.
- Ghi rõ ngưỡng dùng để đếm `low_confidence_count`.
- **Không** đưa cột ground truth hay cột sau khi sửa (thời gian sửa, action, status, notes) vào file pre-label;
  ứng dụng sẽ tự loại bỏ kèm cảnh báo.
