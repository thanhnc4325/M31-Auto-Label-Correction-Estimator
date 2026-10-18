# Nhật ký

> Nhật ký tổng hợp giai đoạn 05/10/2026–18/10/2026. Phần M31 dựa trên file trong repository;
> phần phân công, tỷ lệ và trạng thái thành viên dựa trên xác nhận của trưởng nhóm và liên kết Drive nhóm cung cấp.


## Phần M31 (repository này)

| Ngày | Việc đã làm | Bằng chứng |
|---|---|---|
| 2026-10-12 | Dựng MVP: schema CSV, baseline tạm thời, khung hiệu chỉnh/MAE, giao diện Streamlit tiếng Việt, dữ liệu minh họa sinh ngẫu nhiên, 11 unit test | `m31/`, `app.py`, `data/sample/`, `tests/` |
| 2026-10-12 | Tạo `.venv` (Python 3.14.2, không có 3.12), cài streamlit 1.65.0 / pandas 3.0.6 / numpy 2.5.4; sửa lỗi Spearman cần scipy | `requirements.txt`, `m31/calibration.py` |
| 2026-10-18 | Sao chép `end_corrected.zip` vào `data/real/local/` (hash khớp bản gốc); viết `prepare_real_data.py`; tích hợp pilot 100 ảnh vào app; thêm chống rò rỉ, mốc hằng số, bảng ưu tiên × cần sửa; 17 unit test; tài liệu `docs/` | `scripts/prepare_real_data.py`, `data/real/processed/`, `docs/` |
| 2026-10-18 | Kiểm kê nguồn dữ liệu; bảng 29 đặc trưng box/mask + data dictionary; CV phân tầng 50 lần + kiểm định hoán vị; kết luận chưa tích hợp vì không thắng "đoán 0"; 20 unit test | `scripts/build_features.py`, `scripts/analyze_features.py`, `docs/07_phan_tich_dac_trung.md` |
| 2026-10-18 | Chuẩn bị nộp: slide 8 trang, thumbnail, kịch bản video 3 phút, rà soát đường dẫn cá nhân và dữ liệu nhạy cảm, mô phỏng `.gitignore` và bản clone sạch, checklist nộp bài | `docs/M31_slides_team.pptx`, `docs/thumbnail.png`, `docs/05_…`, `docs/08_checklist_nop_bai.md` |

## Thành viên, công việc và bằng chứng

**Thời gian thực hiện:** 05/10/2026–18/10/2026. Tỷ lệ đóng góp và trạng thái rút lui dưới đây do trưởng nhóm xác nhận. Các liên kết Drive là nơi nhóm lưu bằng chứng; nhật ký không khẳng định nội dung cụ thể bên trong những thư mục này đã được kiểm tra độc lập.

| Thành viên | Mã học viên | Công việc ghi nhận | Bằng chứng | Đóng góp / trạng thái |
|---|---|---|---|---|
| Nguyễn Công Thành | 2A202602219 | Trưởng nhóm, phụ trách chính sản phẩm M31: xây dựng và tích hợp ứng dụng, xử lý pilot thật, phân tích đặc trưng/giới hạn, hoàn thiện tài liệu, slide và thumbnail. | Mã nguồn và tài liệu trong repository: `app.py`, `m31/`, `scripts/`, `tests/`, `docs/`; thư mục Drive nhóm do trưởng nhóm cung cấp: [Drive](https://drive.google.com/drive/folders/1M4Us5z2bfU986iPbZK6COfsNlipHL1OE?usp=sharing). | 45% |
| Lưu Quang Hùng | 2A202602142 | Phụ trách phần dữ liệu/model YOLO và SAM cho pilot; bàn giao dữ liệu dự đoán/segmentation và correction log để nhóm tích hợp, đối chiếu trong M31. | Gói dữ liệu pilot `end_corrected.zip` và các CSV/ghi chú liên quan; thư mục Drive TV2 do nhóm cung cấp: [Drive](https://drive.google.com/drive/folders/1M4Us5z2bfU986iPbZK6COfsNlipHL1OE?usp=sharing). | 40,1% |
| Kim Nguyên Khôi | 2A202602116 | Thành viên TV1; có tham gia công việc nhóm trong giai đoạn dự án. Chi tiết đầu việc riêng chưa được mô tả trong thông tin hiện có nên cần đối chiếu với sản phẩm bàn giao trước khi ghi cụ thể hơn. | Thư mục Drive TV1 do nhóm cung cấp: [Drive](https://drive.google.com/drive/folders/1ShHRvyP3VMuxv2QUCfr7ygFDumct_OqL). | 4,9% |
| Nguyễn Văn Trọng | 2A202602276 | Hoàn thành phần công việc tương ứng 10% đóng góp trước khi xin rút lui. Chi tiết nhiệm vụ cụ thể chưa được mô tả trong thông tin hiện có; không ghi nhận là thành viên tiếp tục thực hiện đến ngày nộp. | Tỷ lệ và trạng thái hoàn thành trước khi rút lui do trưởng nhóm xác nhận; cần đính kèm sản phẩm/tin nhắn/commit nếu có để làm bằng chứng cụ thể. | 10%; đã rút lui sau khi hoàn thành phần việc |

**Lưu ý về bằng chứng:** các tỷ lệ trên do nhóm xác nhận và cộng lại thành 100%. File trong repository là bằng chứng trực tiếp cho phần M31. Với đầu việc của Khôi và Trọng, hiện chưa có tên sản phẩm hoặc commit cụ thể trong nhật ký; nhóm nên bổ sung tên file, ảnh chụp bàn giao, tin nhắn xác nhận hoặc commit tương ứng nếu cần chứng minh chi tiết với ban tổ chức. Không coi link thư mục Drive tự nó là bằng chứng cho một đầu việc cụ thể nếu chưa chỉ rõ file bên trong.

### Mốc chung của dự án

| Thời gian | Mốc | Bằng chứng |
|---|---|---|
| 05/10/2026–17/10/2026 | Chuẩn bị dữ liệu pilot, chạy thử YOLO/SAM, ghi correction time và xây dựng ứng dụng M31 song song theo phần việc nhóm. | Gói bàn giao pilot, CSV correction log, thư mục Drive thành viên và lịch sử repository. |
| 18/10/2026 | Chốt pilot, kiểm tra giới hạn, hoàn thiện tài liệu và bộ nội dung nộp. | `docs/01`–`docs/08`, `docs/M31_slides_team.pptx`, `docs/thumbnail.png`, mã nguồn và kiểm thử trong repository. |