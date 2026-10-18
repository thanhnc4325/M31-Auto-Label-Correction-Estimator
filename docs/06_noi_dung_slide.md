# Slide trình bày

File: [`M31_slides_team.pptx`](M31_slides_team.pptx) (9 slide, 16:9, có ghi chú người thuyết trình). Tạo lại bằng
`python scripts/make_slides.py` (cần `python-pptx`; số liệu đọc từ `data/real/processed/`).

| # | Tiêu đề | Nội dung |
|---|---|---|
| 1 | M31 · Auto-Label Correction Estimator | Mục tiêu; pilot 100 ảnh BDD100K, lớp car; danh sách thành viên; không phải đánh giá toàn bộ BDD100K |
| 2 | Pre-label nhanh, nhưng con người vẫn phải sửa | 4 bước: pre-label → M31 ước lượng → người sửa → đối chiếu; ràng buộc chống rò rỉ |
| 3 | Pipeline tách riêng đặc trưng và nhãn đánh giá | ZIP local → CSV đặc trưng / CSV log → M31 → app 5 tab; 20 unit test |
| 4 | Pilot nhỏ: chỉ 13/100 ảnh mỗi task cần sửa | 100 ảnh, 555 box/mask, 13/100 cần sửa (cùng 13 ảnh), TB 1,17 s / 1,69 s |
| 5 | Chưa mô hình nào thắng "luôn đoán 0 giây" | Biểu đồ MAE (CV 5-fold × 50): đoán 0 = 1,17 / 1,69 s; ridge số box 1,98 / 2,85; ridge nhóm A 2,07 / 2,98; hai bước 2,11 / 3,01; 0% lần lặp thắng; baseline M31 36,5 / 71,4 s |
| 6 | Xếp hạng ảnh cần sửa chưa khác ngẫu nhiên | AUC logistic 0,56 (p 0,41 / 0,32); điểm chất lượng 0,52; tín hiệu khám phá: chồng lấn box 0,70, độ lấp mask ≈ 0,67 |
| 7 | So sánh model trên pilot: giữ mặc định | 22 YOLO, 11 SAM trên cùng pilot (100 ảnh, lớp car, RTX 2060); SAM2-T/S nhanh hơn SAM-B 2,2–2,5× với cùng box prompt YOLOv8n; IoU +0,012…+0,016 trên 92 ảnh có prompt (khám phá); detection chỉ 14 ảnh (126 box) có nhãn, chưa xếp hạng; giữ YOLOv8n + SAM-B. Chi tiết phương pháp trong ghi chú slide và docs/01, docs/11–13 |
| 8 | Giới hạn cần nói rõ | Mẫu nhỏ; baseline giả định; SAM không có confidence riêng; đo lường; ngưỡng do M31 chọn; không tuyên bố độ chính xác |
| 9 | Bước tiếp theo | Thêm ≥ 50 ca sửa/task; đăng ký trước mô hình nhỏ; đặc trưng ảnh nếu có ảnh gốc |

Mọi con số lấy từ `docs/01_bai_toan_giai_phap_gioi_han.md` và `docs/07_phan_tich_dac_trung.md`.
Thumbnail dự án: [`thumbnail.png`](thumbnail.png) (1280×720, không dùng ảnh BDD100K), tạo lại bằng `scripts/make_thumbnail.py`.
