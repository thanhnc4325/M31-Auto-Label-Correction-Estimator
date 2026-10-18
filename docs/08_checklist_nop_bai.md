# Checklist nộp bài

Trạng thái kiểm kê ngày 2026-10-18.

## Đã có trong repository

| Hạng mục | File | Ghi chú |
|---|---|---|
| Hướng dẫn cài và chạy | `README.md`, `run_app.bat`, `requirements.txt` | Đã thử trên bản clone sạch (chỉ gồm file sẽ commit): 20/20 test pass, app chạy với cả 3 nguồn dữ liệu |
| Ứng dụng | `app.py`, `m31/` | Streamlit, giao diện tiếng Việt, 5 tab |
| Script dữ liệu / phân tích | `scripts/` | prepare, build_features, analyze_features, make_sample_data, make_slides, make_thumbnail |
| Kiểm thử | `tests/test_core.py` | 20 unit test |
| Bài toán, giải pháp, giới hạn | `docs/01_bai_toan_giai_phap_gioi_han.md` | Kết quả pilot trung thực |
| Sơ đồ kiến trúc | `docs/02_kien_truc.md` | Mermaid (GitHub hiển thị trực tiếp) |
| Thiết kế và kiểm thử | `docs/03_thiet_ke_va_kiem_thu.md` | |
| Nhật ký | `docs/04_nhat_ky.md` | **Chỉ có phần M31**; phần các thành viên khác còn trống |
| Kịch bản video ≤ 3 phút | `docs/05_kich_ban_video_demo.md` | Có checklist trước khi quay và danh sách số liệu kèm nguồn |
| Slide | `docs/M31_slides.pptx`, `docs/06_noi_dung_slide.md` | 8 slide + ghi chú người thuyết trình |
| Thumbnail | `docs/thumbnail.png` | 1280×720, không dùng ảnh BDD100K |
| Phân tích đặc trưng + data dictionary | `docs/07_phan_tich_dac_trung.md`, `data/real/processed/pilot100_features_dictionary.csv` | |
| Dữ liệu pilot đã xử lý | `data/real/processed/` | Đã rà soát: không có đường dẫn máy, email hay tên người |

## Chưa có (nhóm tự làm)

- [ ] Git repository / GitHub: thư mục chưa phải git repo; chưa push.
- [ ] Video demo: chưa quay, chưa đăng.
- [ ] Tên thành viên trên slide 1 và trong README, nếu đề bài yêu cầu (hiện chỉ ghi "Nhóm M31").
- [ ] Nhật ký các thành viên khác (YOLO, SAM, người sửa nhãn), kèm bằng chứng.
- [ ] LICENSE: chưa chọn. Lưu ý BDD100K có điều khoản riêng; repo không chứa ảnh hay nhãn BDD100K.
- [x] Nhóm đồng ý công khai `pilot100_correction_log.csv` (có ghi chú người sửa), xác nhận ngày 18/10/2026; file được commit cùng dữ liệu pilot. `docs/05`, `docs/01` và `docs/07` có trích một số dòng/tổng hợp từ log.
- [ ] Triển khai server (Hùng): xem mục "Triển khai lên server" trong README.
- [ ] Ảnh chụp màn hình app cho README (tùy chọn).

## Bước tạo repo (PowerShell, trong thư mục dự án)

```powershell
git init
git add .
git status                      # kiểm tra: KHÔNG có .zip, .png/.jpg trong data/, .venv/
git commit -m "M31: MVP + pilot 100 ảnh BDD100K"
# Tạo repo rỗng trên GitHub (không tick README/.gitignore), rồi:
git remote add origin <URL-repo>
git branch -M main
git push -u origin main
```

Kiểm tra nhanh sau `git add .`: `git ls-files | Select-String -Pattern "\.zip$|data/.*\.(png|jpg)$"` phải không trả về dòng nào.

## Bước quay và đăng video

1. Làm theo checklist "Chuẩn bị trước khi quay" trong `docs/05_kich_ban_video_demo.md`.
2. Quay màn hình (vd. Xbox Game Bar `Win + Alt + R`, OBS, hoặc chức năng Record của PowerPoint), không quá 3 phút.
3. Xem lại: không lộ đường dẫn cá nhân, email, tab khác; mọi con số khớp bảng nguồn trong kịch bản.
4. Đăng (vd. YouTube ở chế độ Unlisted hoặc Google Drive có quyền xem), rồi dán link vào README.

## Nộp bài

Gửi theo yêu cầu của lớp: link repo, link video, `docs/M31_slides.pptx` và `docs/thumbnail.png`.
