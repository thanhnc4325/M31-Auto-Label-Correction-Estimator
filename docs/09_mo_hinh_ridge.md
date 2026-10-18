# Mô hình Ridge đã huấn luyện (thử nghiệm)

**Kết luận:** đã train một mô hình có giám sát thật (Ridge, mỗi task một mô hình) trên pilot 100 ảnh.
Theo cross-validation, Ridge **có MAE cao hơn** việc luôn đoán 0 giây ở cả hai task, và thua ở **100%** số lần lặp.
Mô hình được giữ lại và tích hợp vào app dưới dạng **lựa chọn thử nghiệm**. Không tuyên bố nó cải thiện độ chính xác hay tiết kiệm thời gian.

## Dữ liệu

| | Detection (yolov8n) | Segmentation (sam_b) |
|---|---:|---:|
| Số mẫu (ảnh) | 100 | 100 |
| Ảnh có `correction_time_sec` > 0 | 13 | 13 |

- Nguồn: `data/real/processed/pilot100_prelabels.csv` (đặc trưng) và `pilot100_correction_log.csv` (nhãn). Hai file ghép 1-1 theo `(image_id, task, model)`.
  Correction log được nhóm đồng ý công khai ngày 2026-10-18.
- **Giữ toàn bộ** 87 ảnh có thời gian sửa = 0 mỗi task trong tập train.
- 13 ảnh cần sửa là **cùng 13 image_id** ở hai task.
- Không dùng dữ liệu minh họa (`data/sample/`).

## Đầu vào và mô hình

| Feature | Định nghĩa |
|---|---|
| `num_predictions` | Số box/mask pre-label |
| `low_confidence_count` | Số dự đoán có confidence < 0.5 |
| `uncertainty` | 1 − `mean_confidence`; ảnh không có dự đoán được tính là 1.0 (theo `m31.calibration._design` có sẵn) |

- Ridge tuyến tính: `t = b0 + b1·num_predictions + b2·low_confidence_count + b3·uncertainty`, α = 1.0, không phạt hệ số chặn,
  không chuẩn hoá đặc trưng; dự đoán bị cắt về ≥ 0. Dùng đúng hàm `_design` / `_fit_ridge` có sẵn trong `m31/calibration.py`.
- **Không dùng làm đầu vào:** `correction_time_sec`, `edits_*`, `status`, `notes`, ground truth, hay bất kỳ thông tin nào phát sinh sau khi con người sửa.
  `merge_for_training()` chỉ giữ khoá, 3 cột pre-label và nhãn; có test kiểm tra.

## Đánh giá

- 5-fold theo **nhóm image_id**: hai task của cùng một ảnh luôn nằm chung fold, tránh rò rỉ giữa detection và segmentation.
  Fold được phân tầng theo "ảnh có sửa".
- Lặp 20 lần với seed 42–61. Mỗi fold kiểm tra có **2–3** ảnh có sửa; mỗi fold train có 10–11 ảnh có sửa
  (chi tiết lần lặp đầu: `data/real/processed/pilot100_ridge_cv_folds.csv`).
- Mốc so sánh: **luôn đoán 0 giây**, và **median `correction_time_sec` của fold train**. Median luôn bằng 0, nên hai mốc trùng nhau.

### Kết quả (chạy ngày 2026-10-18)

| Task | MAE Ridge (CV) | MAE luôn đoán 0 | MAE median train | Độ lệch chuẩn Ridge giữa lần lặp | % lần lặp Ridge thắng 0 giây |
|---|---:|---:|---:|---:|---:|
| detection | 2,007 s | **1,170 s** | 1,170 s | 0,038 | 0% |
| segmentation | 2,885 s | **1,685 s** | 1,685 s | 0,110 | 0% |

Đầy đủ (gồm MAE riêng trên ảnh có sửa / không sửa) trong `models/ridge_pilot100.json`, trường `tasks.<task>.cv`.

**Diễn giải:** Ridge giảm được sai số trên 13 ảnh có sửa, nhưng dự đoán vài giây cho 87 ảnh không cần sửa,
nên tổng MAE luôn tệ hơn việc đoán 0. Kết quả này khớp với phân tích 29 đặc trưng trong
[07_phan_tich_dac_trung.md](07_phan_tich_dac_trung.md). Với 13 ca sửa mỗi task, không thể kết luận mô hình có ích.

## Mô hình cuối (artifact)

`models/ridge_pilot100.json` (khoảng 4 KB) được fit trên **toàn bộ** 100 ảnh mỗi task, sau khi đánh giá. File gồm:
hệ số và tên features, định nghĩa features, danh sách đầu vào bị loại, số mẫu, số ảnh có sửa, kết quả CV,
cấu hình CV, ngày train, và đường dẫn + SHA-256 + số dòng của hai file nguồn.
**Không** chứa dòng nào của correction log (không image_id, không thời gian từng ảnh, không ghi chú).

| Task | b0 | num_predictions | low_confidence_count | uncertainty |
|---|---:|---:|---:|---:|
| detection | −0,099 | 0,154 | 0,133 | 0,309 |
| segmentation | 0,370 | −0,035 | 0,739 | −0,044 |

## Train lại

```powershell
.\.venv\Scripts\python.exe scripts\train_ridge.py                 # mặc định k=5, seed=42, repeats=20
.\.venv\Scripts\python.exe scripts\train_ridge.py --repeats 50    # tuỳ chọn
```

Script in số mẫu, bảng fold, CV MAE, hệ số; rồi ghi đè `models/ridge_pilot100.json` và `pilot100_ridge_cv_folds.csv`.

## Trong app

Thanh bên → **Mô hình thời gian sửa** → chọn **"Ridge đã huấn luyện — thử nghiệm"** (mặc định là "Baseline").

- Đầu trang hiện cảnh báo kèm CV MAE của Ridge và các mốc; vì Ridge không thắng, app ghi rõ
  "Ridge KHÔNG tốt hơn mốc luôn đoán 0 giây".
- Tab ④ có bảng CV từ artifact. Bảng "Từng ảnh" dùng mô hình đã fit trên chính pilot, nên app ghi chú rằng đó **không** phải kết quả kiểm chứng.
- Điểm chất lượng và mức ưu tiên không đổi; chỉ cột thời gian sửa ước lượng đổi theo mô hình.

## So sánh với mô hình khác

Hurdle và hồi quy trung vị (LAD) có MAE thấp hơn Ridge, nhưng chỉ vì gần như luôn dự đoán 0; không mô hình nào thắng mốc luôn đoán 0 giây trong CV lồng nhau. Ridge vẫn là lựa chọn thử nghiệm; mặc định giữ Baseline. Xem [10_so_sanh_mo_hinh.md](10_so_sanh_mo_hinh.md).

## Giới hạn

- 100 ảnh, 1 lớp (car), 13 ca sửa mỗi task. Mỗi fold kiểm tra chỉ có 2–3 ca, nên MAE theo fold dao động mạnh.
- Thời gian sửa do nhóm ghi nhận, không kiểm chứng độc lập.
- Chỉ 3 đặc trưng; SAM dùng confidence của box YOLO. Cách mã hoá `uncertainty = 1` cho ảnh không có dự đoán là quy ước có sẵn, chưa được tối ưu.
- Áp dụng mô hình lên dữ liệu khác (dữ liệu tải lên, minh họa) chỉ mang tính minh hoạ.
