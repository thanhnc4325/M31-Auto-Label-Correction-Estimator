# So sánh Ridge với hai mô hình cho dữ liệu nhiều số 0

**Kết luận:** không có mô hình nào dự đoán `correction_time_sec` tốt hơn mốc **luôn đoán 0 giây**.
Hai phương án mới có MAE thấp hơn Ridge, nhưng chỉ vì chúng **gần như luôn dự đoán 0**: LAD suy biến hẳn thành 0,
còn hurdle hiếm khi dự đoán khác 0. Theo tiêu chí đặt trước, **không đổi mô hình mặc định**:
app giữ **Baseline** làm mặc định và **Ridge** là lựa chọn thử nghiệm. Hurdle và LAD không được tích hợp vào app.

Tái lập: `.\.venv\Scripts\python.exe scripts\compare_models.py` (khoảng 5 giây).
Kết quả: `data/real/processed/pilot100_model_comparison*.{csv,json}`.

## Thiết lập (chốt trước khi chạy)

- **Dữ liệu:** pilot thật, 100 ảnh mỗi task, 13 ảnh có thời gian sửa > 0 (cùng 13 image_id ở hai task). Giữ mọi dòng, kể cả 87 dòng = 0.
- **Đầu vào:** đúng 3 đặc trưng của Ridge (`num_predictions`, `low_confidence_count`, `uncertainty`). Không dùng `edits_*`, `status`, `notes`, ground truth.
- **Ứng viên:** tối đa 2 mô hình mới, cả hai hợp với MAE và dữ liệu nhiều số 0, không tinh chỉnh siêu tham số theo kết quả:
  1. **Hurdle:** logistic L2 (λ = 1, đặc trưng chuẩn hoá theo fold train) dự đoán P(sửa). Dự đoán = median thời gian của
     ca có sửa trong fold train nếu P > 0,5, ngược lại 0. Ngưỡng 0,5 là quy tắc tối ưu MAE (median có điều kiện).
  2. **LAD:** hồi quy tuyến tính tối thiểu sai số tuyệt đối (hồi quy trung vị), giải bằng IRLS.
- **Fold:** 5-fold theo **image_id** (hai task của một ảnh luôn chung fold), phân tầng theo ảnh có sửa, lặp 20 lần (seed 42–61).
  Mỗi fold kiểm tra có **2–3** ảnh có sửa ở mọi lần lặp.
- **Tránh chọn trên test:** **CV lồng nhau**. Trong mỗi fold ngoài, CV 4-fold (cũng theo image_id) **chỉ trên tập train** chọn
  1 trong {ridge, hurdle, lad} có MAE nhỏ nhất; mô hình được chọn rồi được fit lại trên tập train và đánh giá trên fold ngoài.
  MAE của dòng `nested_selected` là ước lượng cho **quy trình chọn mô hình**. Các dòng từng mô hình riêng lẻ chỉ là so sánh **khám phá**.
- **Tiêu chí đổi mặc định:** `nested_selected` có MAE < luôn đoán 0 ở ≥ 80% số lần lặp cho **cả hai** task.

## Kết quả

MAE (giây), trung bình 20 lần lặp. "≥ 0,5 s" là số ảnh (trên 100) được dự đoán từ 0,5 giây trở lên.

| Task | Mô hình | MAE | Độ lệch chuẩn giữa lần lặp | MAE fold (min–max) | Độ lệch chuẩn giữa fold | MAE ảnh có sửa | MAE ảnh không sửa | ≥ 0,5 s | % lần lặp thắng "đoán 0" |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| detection | **luôn đoán 0** | **1,170** | 0 | 0,76–1,83 | 0,26 | 9,00 | 0,00 | 0 | — |
| detection | Ridge (hiện tại) | 2,007 | 0,038 | 1,32–2,73 | 0,28 | 7,74 | 1,15 | 80,3 | 0% |
| detection | hurdle | 1,225 | 0,062 | 0,76–1,98 | 0,32 | 8,97 | 0,07 | 0,8 | 0% |
| detection | LAD | 1,170 | 0 | 0,76–1,83 | 0,26 | 9,00 | 0,00 | 0 | 0% |
| detection | **chọn lồng nhau** | 1,213 | 0,064 | 0,76–1,98 | 0,31 | 8,97 | 0,05 | 0,65 | **0%** |
| segmentation | **luôn đoán 0** | **1,685** | 0 | 0,82–3,83 | 0,77 | 12,96 | 0,00 | 0 | — |
| segmentation | Ridge (hiện tại) | 2,885 | 0,110 | 1,70–4,83 | 0,75 | 11,07 | 1,66 | 77,9 | 0% |
| segmentation | hurdle | 1,747 | 0,072 | 0,82–3,83 | 0,80 | 12,93 | 0,08 | 0,8 | 0% |
| segmentation | LAD | 1,685 | 0 | 0,82–3,83 | 0,77 | 12,96 | 0,00 | 0 | 0% |
| segmentation | **chọn lồng nhau** | 1,738 | 0,074 | 0,82–3,83 | 0,80 | 12,93 | 0,07 | 0,70 | **0%** |

Vòng trong chọn hurdle ở 49 và LAD ở 51 trên 100 fold ngoài, cho mỗi task; **không lần nào chọn Ridge**.

## Diễn giải

- **LAD = đoán 0:** khi 87% nhãn bằng 0, hồi quy trung vị tối ưu là đường gần 0. Dự đoán lớn nhất khi fit trên toàn pilot là khoảng 5·10⁻⁵ giây.
  MAE "bằng" mốc 0 vì nó chính là mốc 0, không phải vì nó học được gì.
- **Hurdle:** logistic hiếm khi vượt 0,5, trung bình 0,8 ảnh/100 được dự đoán có sửa. Những lần dự đoán đó phần lớn sai, nên MAE nhỉnh hơn mốc 0.
  Fit trên toàn pilot, hurdle dự đoán 0 cho mọi ảnh.
- **Ridge** giảm sai số trên 13 ảnh có sửa (7,7 so với 9,0 s ở detection) nhưng dự đoán khoảng 1–3 s cho gần như mọi ảnh,
  nên sai số trên 87 ảnh không sửa đẩy MAE lên cao nhất.
- Kết quả theo fold **dao động mạnh** (vd. segmentation 0,82–3,83 s ngay cả với mốc 0), vì mỗi fold chỉ có 2–3 ca sửa.
  So sánh này chỉ có giá trị khám phá; pilot quá nhỏ để có tập kiểm tra độc lập.
- Với 3 đặc trưng hiện có, dữ liệu không chứa tín hiệu đủ mạnh để phân biệt ảnh cần sửa. Điều này khớp với
  [07_phan_tich_dac_trung.md](07_phan_tich_dac_trung.md) (AUC ≈ 0,56) và [09_mo_hinh_ridge.md](09_mo_hinh_ridge.md).

## Quyết định

| Tiêu chí | detection | segmentation | Đạt? |
|---|---|---|---|
| `nested_selected` thắng "đoán 0" ở ≥ 80% số lần lặp | 0% | 0% | **Không** |

- App **không đổi**: Baseline là mặc định, "Ridge đã huấn luyện — thử nghiệm" vẫn là lựa chọn kèm cảnh báo.
- Không tích hợp hurdle hay LAD vào app: chúng chỉ tái tạo mốc "đoán 0".
- **Chưa cải thiện** được dự đoán thời gian sửa. Muốn tiến xa hơn cần thêm ca sửa (gợi ý ≥ 50 mỗi task) hoặc đặc trưng mới, đánh giá trên ảnh chưa dùng.
