# Kịch bản video demo (≤ 3 phút)

**Thông điệp chính:** M31 là công cụ minh bạch để ước lượng công sức sửa nhãn và tự kiểm tra mình.
Trên pilot 100 ảnh, mô hình **chưa tốt hơn** việc luôn đoán 0 giây; chúng tôi nói rõ điều đó và không tuyên bố độ chính xác.

## Chuẩn bị trước khi quay

- [ ] `streamlit run app.py` đang chạy; trình duyệt mở `http://localhost:8501`, phóng to 110–125% để chữ dễ đọc.
- [ ] Thanh bên: nguồn **Pilot thật · 100 ảnh BDD100K (local)**, tick "Kèm correction log thật của pilot".
- [ ] Có `data\real\local\end_corrected.zip` để overlay hiển thị ở tab ③. Nếu không có, nói "overlay chỉ có trên máy local".
- [ ] Mở sẵn `docs/M31_slides.pptx`, slide 5, để chèn 10 giây cuối (tùy chọn).
- [ ] Tắt thông báo; không để lộ đường dẫn cá nhân, email hoặc tab trình duyệt khác.

## Kịch bản

| Thời điểm | Màn hình | Lời thoại |
|---|---|---|
| 0:00–0:15 | Đầu trang app, banner xanh "DỮ LIỆU PILOT THẬT" | "Đây là M31: ước lượng ảnh nào cần review trước và tốn bao lâu để sửa nhãn YOLO và SAM gán tự động. Dữ liệu là bộ BDD100K, chỉ lớp car. Nhóm chạy pilot trên 100 ảnh được chọn từ bộ dữ liệu này, dùng YOLOv8n và SAM-B để tạo pre-label, sau đó ghi nhận công sức sửa nhãn. Nhóm không huấn luyện YOLO/SAM từ đầu trên 100 ảnh này. Thử nghiệm M31 đánh giá khả năng ước lượng công sửa trên tập pilot đó." |
| 0:15–0:35 | Thanh bên ①: bấm qua 3 nguồn, dừng ở "MINH HỌA" để thấy banner đỏ, quay lại "Pilot thật" | "App có ba nguồn: pilot thật, file tải lên, và dữ liệu minh họa. Dữ liệu minh họa luôn có banner đỏ và không dùng để báo cáo." |
| 0:35–1:00 | Tab ② Danh sách & ưu tiên: lọc "cao", chỉ cột Lý do | "Mỗi ảnh có điểm chất lượng thử nghiệm, thời gian sửa ước lượng, mức ưu tiên và lý do. Đầu vào chỉ là thông tin có trước khi sửa: confidence, số box, box nhỏ, độ lấp mask." |
| 1:00–1:30 | Tab ③ Chi tiết: chọn `00054602-3bf57337 \| detection`; chỉ overlay, bảng đặc trưng, bảng log sửa | "Overlay YOLO đọc thẳng từ ZIP local. Ảnh này người sửa xoá một box bắt nhầm nắp ca-pô, mất 8,5 giây. Log sửa chỉ hiện để đối chiếu, không dùng để dự đoán." |
| 1:30–2:15 | Tab ④ So sánh: bảng MAE, rồi bảng ưu tiên × cần sửa | "Đối chiếu với thời gian sửa thật: chỉ 13 trên 100 ảnh mỗi task cần sửa. Baseline ước lượng cao gấp 30 đến 40 lần, và luôn đoán 0 giây cho sai số thấp nhất. Tỉ lệ ảnh cần sửa gần như như nhau ở mọi mức ưu tiên." |
| 2:15–2:40 | Tab ⑤ Giới hạn & kiểm tra rò rỉ | "Chúng tôi còn thử 29 đặc trưng box và mask với cross-validation lặp 50 lần: không mô hình nào thắng việc đoán 0 giây. Nên app không bật mô hình hiệu chỉnh. Thời gian sửa, thao tác sửa, status, notes và ground truth không bao giờ là đầu vào." |
| 2:40–3:00 | (Tùy chọn) slide 8 "Bước tiếp theo" | "Bước tiếp theo: thêm dữ liệu sửa, đăng ký trước một mô hình nhỏ dựa trên độ chồng lấn box và độ lấp mask, rồi đánh giá trên ảnh mới. Cảm ơn." |

## Không được nói

- "M31 dự đoán chính xác …" / "mô hình đạt độ chính xác X%".
- "Đã kiểm chứng trên BDD100K".
- Bất kỳ con số nào không hiển thị trên màn hình hoặc không có trong `docs/01` và `docs/07`.

## Số liệu dùng trong video (nguồn)

| Số liệu | Giá trị | Nguồn |
|---|---|---|
| Ảnh cần sửa | 13/100 mỗi task | `docs/01`, tab ④ |
| Thời gian sửa TB | 1,17 s (YOLO), 1,69 s (SAM) | `docs/01` |
| MAE baseline M31 | 36,5 s / 71,4 s | tab ④, `docs/01` |
| MAE luôn đoán 0 | 1,17 s / 1,69 s | tab ④, `docs/07` |
| Lần lặp thắng "đoán 0" | 0% | `docs/07` |
| Ảnh `00054602-3bf57337` | xoá 1 box, 8,5 s | `pilot100_correction_log.csv` |
