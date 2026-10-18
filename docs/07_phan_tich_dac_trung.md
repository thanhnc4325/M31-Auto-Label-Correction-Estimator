# Phân tích đặc trưng pilot (2026-10-18)

**Kết luận ngắn:** các đặc trưng mới (29 đặc trưng, tính từ box/mask trước khi sửa) **chưa cho thấy lợi ích**.
Không mô hình nào có MAE thấp hơn "luôn đoán 0 giây" ở bất kỳ lần lặp nào. Khả năng xếp hạng ảnh cần sửa
(AUC ≈ 0,56) không khác ngẫu nhiên (p hoán vị 0,32–0,41). Theo tiêu chí đặt trước, đặc trưng mới
**không được tích hợp** vào phần dự đoán của ứng dụng.

Tái lập:
```powershell
.\.venv\Scripts\python.exe scripts\build_features.py      # bảng đặc trưng + data dictionary
.\.venv\Scripts\python.exe scripts\analyze_features.py    # CV, kiểm định hoán vị (~vài phút)
```

## 1. Kiểm kê nguồn dữ liệu (bản sao `data/real/local/end_corrected.zip`)

| File | Dòng | Ảnh | Cột | Giá trị thiếu | Vai trò |
|---|---:|---:|---|---|---|
| `yolo_predictions_100.csv` | 555 | 92 | image_id, class_name (chỉ `car`), confidence, x_min, y_min, x_max, y_max | 0 | **Đầu vào** |
| `sam_predictions_100.csv` | 555 | 92 | image_id, task, model (`sam_b`), class_name, prompt_source (`yolo_box`), x1, y1, x2, y2, confidence, mask_path | 0 | **Đầu vào** |
| `sam_masks/*_box_NN.png` | 555 | 92 | mask nhị phân L 1280×720, giá trị {0, 255} | 0 file thiếu | **Đầu vào** |
| `sam_masks/*_car_sam.png` | 100 | 100 | mask hợp của ảnh | — | Chỉ để đối chiếu (khớp `mask_union_coverage`, lệch ≤ 5·10⁻⁷) |
| `yolo_overlays/`, `sam_overlays/` | 100 + 100 | 100 | ảnh đã vẽ box/mask | — | Lấy kích thước ảnh; hiển thị. **Không** lấy đặc trưng pixel (đã bị vẽ đè) |
| `correction_log_{yolo,sam}_100.csv` | 100 + 100 | 100 | num_predictions, inference_time_sec, human_correction_time_sec, action_add/delete/modify, status, notes | 0 | **Chỉ làm nhãn đánh giá** |
| `pilot_comparison_yolo_sam_100.csv` | 200 | 100 | như log | 0 | Trùng với hai log, không dùng |
| `image_groundtruth_pairs_100.csv` | 100 | 100 | đường dẫn ảnh/GT | 0 | **Không đọc** |

Nhận xét:
- 8/100 ảnh không có box nào, nên các đặc trưng confidence/kích thước của chúng để trống (cột "thiếu" trong dictionary).
- Confidence và box của SAM **trùng hoàn toàn** với YOLO. SAM chỉ đóng góp thông tin mới qua mask.
- `action_add` = 0 ở mọi dòng. Không có mask rỗng (`frac_empty_masks` luôn = 0, bị loại khỏi mô hình vì hằng số).
- Ảnh gốc BDD100K không có trong gói, nên không tính được đặc trưng ảnh (độ sáng, ban đêm, …).

## 2. Data dictionary

File máy đọc được: `data/real/processed/pilot100_features_dictionary.csv`. Bảng đặc trưng: `data/real/processed/pilot100_features.csv` (200 dòng = 100 ảnh × 2 task).

**Nhóm A** = đặc trưng định trước theo yêu cầu. **Nhóm B** = giả thuyết lấy từ ghi chú người sửa ("bắt nhầm nắp ca-pô ở đáy ảnh",
"viền mask lan sang nền/xe kề"); vì ý tưởng đến từ chính dữ liệu đánh giá, kết quả nhóm B trên pilot này bị **lạc quan** và không được dùng để quyết định.
W, H = kích thước ảnh (1280×720 cho cả 100 ảnh).

| Đặc trưng | Nhóm | Task | Công thức | Nguồn | Thiếu (det) | Thiếu (seg) |
|---|---|---|---|---|---:|---:|
| `n_boxes` | A | cả hai | Số box car của YOLO (với segmentation = số box prompt = số mask) | yolo_predictions_100.csv | 0 | 0 |
| `has_predictions` | A | cả hai | 1 nếu n_boxes > 0 | yolo_predictions_100.csv | 0 | 0 |
| `conf_mean` | A | cả hai | Trung bình confidence | yolo_predictions_100.csv | 8 | 8 |
| `conf_min` | A | cả hai | Confidence nhỏ nhất | yolo_predictions_100.csv | 8 | 8 |
| `conf_max` | A | cả hai | Confidence lớn nhất | yolo_predictions_100.csv | 8 | 8 |
| `conf_std` | A | cả hai | Độ lệch chuẩn confidence (ddof=0) | yolo_predictions_100.csv | 8 | 8 |
| `conf_n_lt030` | A | cả hai | Số box confidence < 0.3 | yolo_predictions_100.csv | 0 | 0 |
| `conf_n_030_050` | A | cả hai | Số box 0.3 ≤ confidence < 0.5 | yolo_predictions_100.csv | 0 | 0 |
| `conf_n_050_070` | A | cả hai | Số box 0.5 ≤ confidence < 0.7 | yolo_predictions_100.csv | 0 | 0 |
| `conf_n_ge070` | A | cả hai | Số box confidence ≥ 0.7 | yolo_predictions_100.csv | 0 | 0 |
| `frac_conf_lt050` | A | cả hai | Tỉ lệ box confidence < 0.5 | yolo_predictions_100.csv | 8 | 8 |
| `box_w_mean` | A | cả hai | mean(x2 − x1), px | yolo_predictions_100.csv | 8 | 8 |
| `box_h_mean` | A | cả hai | mean(y2 − y1), px | yolo_predictions_100.csv | 8 | 8 |
| `box_area_ratio_mean` | A | cả hai | mean((x2−x1)(y2−y1) / (W·H)) | yolo_predictions + kích thước overlay | 8 | 8 |
| `box_area_ratio_min` | A | cả hai | min diện tích box / (W·H) | như trên | 8 | 8 |
| `box_area_ratio_max` | A | cả hai | max diện tích box / (W·H) | như trên | 8 | 8 |
| `box_area_ratio_sum` | A | cả hai | Σ diện tích box / (W·H) | như trên | 8 | 8 |
| `frac_small_boxes` | A | cả hai | Tỉ lệ box có diện tích < 32×32 px | yolo_predictions_100.csv | 8 | 8 |
| `max_pair_iou` | A | cả hai | IoU lớn nhất giữa hai box (0 nếu < 2 box) | yolo_predictions_100.csv | 0 | 0 |
| `n_overlap_pairs` | A | cả hai | Số cặp box có IoU > 0.3 | yolo_predictions_100.csv | 0 | 0 |
| `n_masks` | A | seg | Số mask của ảnh | sam_predictions + sam_masks | — | 0 |
| `frac_empty_masks` | A | seg | Tỉ lệ mask 0 pixel hoặc thiếu file | sam_predictions + sam_masks | — | 8 |
| `mask_area_ratio_mean` | A | seg | mean(pixel mask / (W·H)) | sam_masks | — | 8 |
| `mask_union_coverage` | A | seg | pixel thuộc ≥ 1 mask / (W·H) | sam_masks | — | 8 |
| `mask_fill_in_box_mean` | A | seg | mean(pixel mask trong box prompt / diện tích box) | sam_predictions + sam_masks | — | 8 |
| `mask_fill_in_box_min` | A | seg | min(pixel mask trong box / diện tích box) | sam_predictions + sam_masks | — | 8 |
| `n_boxes_touch_bottom` | B | cả hai | Số box có y2 ≥ H − 2 | yolo_predictions + kích thước overlay | 0 | 0 |
| `mask_outside_box_frac_mean` | B | seg | mean(pixel mask ngoài box / pixel mask) | sam_predictions + sam_masks | — | 8 |
| `mask_overlap_frac` | B | seg | pixel thuộc ≥ 2 mask / pixel thuộc ≥ 1 mask | sam_masks | — | 8 |

"—" = không áp dụng (dòng detection để trống các cột mask).

## 3. Chống rò rỉ

- `build_features.py` chỉ đọc file dự đoán, mask và kích thước overlay. Test `test_build_features_without_logs_or_ground_truth`
  chạy script trên một ZIP **không có** log hay GT; nếu script đọc các file đó thì test sẽ lỗi.
- Nhãn (`correction_time_sec`) chỉ được ghép trong `analyze_features.py`, sau khi đặc trưng đã tính xong (ghép 1-1, có kiểm tra).
- Điền giá trị thiếu (median) và chuẩn hoá được tính **trong từng fold huấn luyện**.
- Không dùng: thời gian sửa, `action_*`, `status`, `notes`, `inference_time_sec`, ground truth.

## 4. Thiết lập đánh giá

- Mỗi task: 100 ảnh, **13 ảnh có thời gian sửa > 0**. Cả 13 ảnh này **trùng nhau** giữa detection và segmentation,
  và box/confidence giống nhau, nên hai task **không** phải hai bằng chứng độc lập.
- 5-fold **phân tầng** theo (thời gian > 0), lặp 50 lần với seed 0–49. Mỗi fold kiểm tra có **2–3 ảnh cần sửa**;
  mỗi fold huấn luyện có 10–11 ảnh (chi tiết lần lặp 0: `pilot100_cv_folds_repeat0.csv`).
- Mốc so sánh: luôn đoán 0; trung bình fold huấn luyện; baseline M31 hiện tại (không học); ridge chỉ dùng `n_boxes`.
- Mô hình: ridge (α = 10, đặc trưng chuẩn hoá) trên nhóm A và A+B; mô hình hai bước = xác suất logistic (L2 = 1) ×
  thời gian sửa TB của các ảnh có sửa trong fold huấn luyện.
- Xếp hạng ảnh cần sửa: AUC; với logistic là AUC out-of-fold. Kiểm định hoán vị: 200 lần xáo nhãn × 5 lần lặp CV.
- Tiêu chí tích hợp (đặt trước khi chạy, chỉ nhóm A): (1) ridge_A hoặc hurdle_A thắng "đoán 0" ở ≥ 80% số lần lặp,
  **hoặc** (2) logistic_A có AUC ≥ 0,70 và p < 0,05.

## 5. Kết quả

### 5.1 Thời gian sửa: MAE (giây), 50 lần lặp

| Task | Mô hình | MAE TB | Độ lệch chuẩn | MAE theo fold (min–max) | MAE trên ảnh có sửa | MAE trên ảnh không sửa | % lần lặp thắng "đoán 0" |
|---|---|---:|---:|---|---:|---:|---:|
| detection | **luôn đoán 0** | **1,17** | 0 | 0,74–1,78 | 9,00 | 0,00 | — |
| detection | TB fold huấn luyện | 2,04 | 0,00 | 1,74–2,49 | 7,84 | 1,17 | 0% |
| detection | baseline M31 | 36,49 | 0 | 25,2–47,7 | 39,62 | 36,02 | 0% |
| detection | ridge `n_boxes` | 1,98 | 0,02 | 1,41–2,59 | 7,62 | 1,13 | 0% |
| detection | ridge A | 2,07 | 0,08 | 0,96–3,44 | 7,56 | 1,25 | 0% |
| detection | ridge A+B | 2,10 | 0,08 | 0,96–3,44 | 7,58 | 1,28 | 0% |
| detection | hai bước A | 2,11 | 0,09 | 1,14–3,39 | 7,79 | 1,26 | 0% |
| segmentation | **luôn đoán 0** | **1,69** | 0 | 0,82–3,93 | 12,96 | 0,00 | — |
| segmentation | TB fold huấn luyện | 2,93 | 0,00 | 2,31–4,71 | 11,29 | 1,68 | 0% |
| segmentation | baseline M31 | 71,37 | 0 | 46,6–96,9 | 83,96 | 69,48 | 0% |
| segmentation | ridge `n_boxes` | 2,85 | 0,03 | 1,76–4,82 | 10,94 | 1,64 | 0% |
| segmentation | ridge A | 2,98 | 0,16 | 0,61–5,79 | 10,79 | 1,82 | 0% |
| segmentation | ridge A+B | 3,19 | 0,17 | 1,16–5,82 | 10,82 | 2,05 | 0% |
| segmentation | hai bước A | 3,01 | 0,14 | 1,15–5,05 | 11,20 | 1,78 | 0% |

Đầy đủ: `data/real/processed/pilot100_cv_regression.csv`.

- Không mô hình nào thắng "đoán 0" ở bất kỳ lần lặp nào. Xét theo từng fold, ridge A chỉ thắng 0,4% số fold (segmentation).
- MAE của chính mốc "đoán 0" dao động 0,74–1,78 s (detection) và 0,82–3,93 s (segmentation) giữa các fold,
  tùy fold rơi vào 2 hay 3 ảnh cần sửa. Với cỡ mẫu này, kết quả theo từng fold **không ổn định**.
- Các mô hình giảm được sai số trên ảnh có sửa (khoảng 7,6 so với 9,0 s), nhưng phải đổi lại bằng sai số trên 87 ảnh không sửa,
  nên tổng MAE luôn tệ hơn.

### 5.2 Xếp hạng ảnh cần sửa (AUC; 0,5 = ngẫu nhiên)

| Task | Cách xếp hạng | Có học? | AUC | Khoảng 5–95% giữa các lần lặp | p hoán vị |
|---|---|---|---:|---|---:|
| cả hai | điểm chất lượng M31 (thấp → cần sửa) | không | 0,52 | — | — |
| cả hai | `n_boxes` (nhiều → cần sửa) | không | 0,62 | — | — |
| cả hai | `conf_min` (thấp → cần sửa) | không | 0,52 | — | — |
| detection | logistic A | có | 0,56 | 0,47–0,65 | 0,41 |
| detection | logistic A+B | có | 0,54 | 0,46–0,62 | 0,44 |
| segmentation | logistic A | có | 0,56 | 0,48–0,64 | 0,32 |
| segmentation | logistic A+B | có | 0,50 | 0,44–0,58 | 0,42 |

Logistic trên khoảng 20–28 đặc trưng với chỉ 10–11 ảnh có sửa mỗi fold huấn luyện bị quá khớp, và không tốt hơn ngẫu nhiên một cách có ý nghĩa.

### 5.3 Đơn biến (KHÁM PHÁ, không phải kết quả)

AUC với chiều "giá trị cao → cần sửa", khoảng tin cậy bootstrap 95% (1000 lần). Các đặc trưng có khoảng tin cậy không chứa 0,5:

| Đặc trưng | Nhóm | AUC | CI 95% | Ghi chú |
|---|---|---:|---|---|
| `max_pair_iou` | A | 0,70 | 0,53–0,85 | box chồng nhau nhiều → hay cần sửa |
| `n_overlap_pairs` | A | 0,66 | 0,50–0,81 | cùng hướng; tương quan mạnh với `max_pair_iou` |
| `mask_fill_in_box_mean` | A | 0,33 | 0,19–0,48 | mask lấp box ít → hay cần sửa (AUC đảo chiều ≈ 0,67) |
| `has_predictions` | A | 0,55 | 0,52–0,58 | hiển nhiên: 8 ảnh không có box đều không cần sửa |

Đã thử 21 đặc trưng (detection) và 28 đặc trưng (segmentation) trên **cùng 13 ảnh cần sửa**. Ở mức 5%, có thể có 1–2 kết quả
"có ý nghĩa" chỉ do ngẫu nhiên, và chưa hiệu chỉnh cho việc so sánh nhiều lần. Các tín hiệu trên chỉ là **giả thuyết**
cần kiểm tra trên dữ liệu mới. Nhóm B không có đặc trưng nào nổi bật.

## 6. Quyết định

| Task | Tiêu chí 1 (thắng "đoán 0" ≥ 80%) | Tiêu chí 2 (AUC ≥ 0,70 và p < 0,05) | Tích hợp? |
|---|---|---|---|
| detection | Không (0%) | Không (0,56; p = 0,41) | **Không** |
| segmentation | Không (0%) | Không (0,56; p = 0,32) | **Không** |

Ứng dụng giữ nguyên cách ước lượng hiện tại và các cảnh báo giới hạn. Bảng đặc trưng được lưu lại để dùng khi có thêm dữ liệu.

## 7. Đề xuất cho vòng dữ liệu tiếp theo

1. **Đăng ký trước** một mô hình nhỏ (vd. logistic trên `max_pair_iou`, `mask_fill_in_box_mean`, `n_boxes`) và chỉ đánh giá nó
   trên ảnh **mới**, không chọn lại đặc trưng sau khi xem kết quả.
2. Cần thêm nhiều ảnh cần sửa (gợi ý: ≥ 50 ca mỗi task) để fold kiểm tra không chỉ có 2–3 ca.
3. Nếu có ảnh gốc BDD100K, có thể thêm đặc trưng ảnh (ban đêm, mưa, độ sáng), vẫn bảo đảm tính được trước khi sửa.
