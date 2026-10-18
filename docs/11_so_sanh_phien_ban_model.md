# So sánh phiên bản YOLO và SAM trên 100 ảnh pilot (2026-10-18)

> **Dữ liệu trong repository công khai:** các file kết quả của lượt này nằm ở thư mục gốc `data/real/model_comparison/` (`per_image_results.csv`, `detections_*.csv`, `detection_gt_subset.csv`, `manual_review_template.csv`, `summary.json`, `paired_stats.json`, `run_config.json`) và được **giữ local, không có trong repository công khai**, vì phần lớn chứa `image_id` hoặc toạ độ box theo từng ảnh. Bốn model của lượt này được chạy lại trong lượt `20261018-222741` với kết quả trùng khớp; repository chỉ chứa 4 file tổng hợp của lượt đó (`run_config.json`, `detection_summary.json`, `segmentation_summary.json`, `sam_eval_recomputed.csv`), xem [12_so_sanh_nhieu_model.md](12_so_sanh_nhieu_model.md).

**Phạm vi:** 100 ảnh BDD100K (10k seg, split train) đã chọn cho pilot M31, chỉ lớp **car**. Đây là pilot, **không đại diện cho toàn bộ BDD100K**.
YOLO chỉ so với YOLO (detection), SAM chỉ so với SAM (segmentation); không so điểm YOLO với điểm SAM.

> **Cập nhật:** lượt mở rộng với 9 YOLO và 7 SAM nằm ở [12_so_sanh_nhieu_model.md](12_so_sanh_nhieu_model.md). Kết quả dưới đây (lượt đầu) được giữ nguyên.

## Kết luận ngắn

| Cặp | Tốc độ (cùng máy, cùng 100 ảnh) | Chất lượng |
|---|---|---|
| YOLOv8n (đang dùng) vs **YOLO11n** | YOLOv8n **nhanh hơn**: 13,7 so với 17,3 ms/ảnh; nhanh hơn ở 94/100 ảnh | Trên **14 ảnh** có nhãn box: precision 0,856 vs 0,879, recall 0,611 vs 0,635. Khoảng tin cậy chênh lệch **chứa 0**, nên **chưa có bằng chứng** model nào chính xác hơn |
| SAM-B (đang dùng) vs **SAM2-B** | SAM2-B **nhanh hơn**: 0,210 so với 0,309 s/ảnh; nhanh hơn ở 92/92 ảnh | IoU gộp 0,771 vs 0,784; IoU TB/ảnh 0,718 vs 0,721. Khoảng tin cậy chênh lệch **chứa 0**, nên **chưa có bằng chứng** model nào chính xác hơn |

Không thay model trong sản phẩm M31. Kết quả chỉ cho thấy khác biệt **tốc độ** đáng tin cậy trên máy này; khác biệt chất lượng chưa được chứng minh.

## Cấu hình chạy

| Mục | Giá trị |
|---|---|
| Máy | Intel Core i5-9400, 32 GB RAM, **NVIDIA RTX 2060 12 GB**, Windows 11 |
| Phần mềm | Python 3.14.2, torch 2.14.1+cu126 (CUDA 12.6), ultralytics 8.4.182, OpenCV 5.0.0 (venv riêng `.venv_models`) |
| Thiết bị suy luận | GPU (`device=0`), cả 4 model |
| YOLO | `imgsz=640`, `conf=0.25` (mặc định như pipeline gốc); dự đoán mọi lớp COCO rồi **lọc `car`**, giống `run_pipeline_100.py` |
| SAM | Prompt = **cùng 555 box YOLOv8n đã bàn giao** (`sam_predictions_100.csv`) cho cả hai SAM, để khác biệt chỉ đến từ SAM |
| Weights | `yolov8n.pt` và `sam_b.pt` (thư mục bàn giao của thành viên phụ trách YOLO/SAM), `yolo11n.pt` (thư mục của thành viên TV1), `sam2_b.pt` (tải từ release Ultralytics v8.4.0, 154 MB, SHA-256 bắt đầu `39722bb0ce2a`) |
| Ảnh / nhãn | Đọc thẳng (chỉ đọc) từ `archive.zip` BDD100K: `bdd100k_seg/.../images/train/<id>.jpg`, `.../labels/train/<id>_train_id.png`, `bdd100k_labels_images_train.json` |

**Đo thời gian:**
- Thời gian một ảnh = wall-clock quanh **một lời gọi model** trên ảnh đã giải mã sẵn trong RAM, có `torch.cuda.synchronize()` trước và sau.
- Bao gồm tiền xử lý, suy luận và hậu xử lý bên trong Ultralytics.
- **Không** tính: nạp model, đọc/giải mã ảnh, lọc lớp và tính chỉ số.
- Trước khi đo, chạy 3 ảnh khởi động (không tính).
- Các model chạy **lần lượt** (YOLOv8n → YOLO11n → SAM-B → SAM2-B), không xen kẽ.
- Thời gian nạp model được ghi riêng nhưng đo khi file đã nằm trong cache của hệ điều hành (sau lần chạy thử), nên **không** phản ánh lần nạp đầu tiên. Ở lần chạy thử, lần nạp `sam_b.pt` đầu tiên mất 5,3 s.

## Kiểm tra tái lập (model đang dùng)

- **YOLOv8n chạy lại vs đầu ra đã bàn giao:** 555/555 box đã bàn giao được tái tạo (IoU ≥ 0,9). Lần chạy lại có 557 box, thêm 2 box, nhiều khả năng do sai khác số học quanh ngưỡng `conf` 0,25.
- **SAM-B chạy lại vs mask đã bàn giao:** IoU trung bình 0,99995 trên 92 ảnh có prompt.

Như vậy cấu hình chạy lại tái tạo đúng pilot gốc, và kết quả so sánh dựa trên cùng điều kiện.

## Kết quả: số ảnh chạy

| Model | Thành công | Thất bại | Ghi chú |
|---|---:|---:|---|
| YOLOv8n | 100 | 0 | |
| YOLO11n | 100 | 0 | |
| SAM-B | 92 | 0 | 8 ảnh không có box prompt (YOLOv8n không bắt được car) nên không chạy SAM; mask rỗng |
| SAM2-B | 92 | 0 | như trên |

## Kết quả: thời gian (giây/ảnh)

| Model | Nạp model* | TB | Trung vị | P90 | Max | Tổng (ảnh đo) |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 0,045 | 0,0137 | 0,0130 | 0,0156 | 0,0192 | 1,37 (100 ảnh) |
| YOLO11n | 0,056 | 0,0173 | 0,0168 | 0,0195 | 0,0249 | 1,73 (100 ảnh) |
| SAM-B | 0,780 | 0,309 | 0,308 | 0,318 | 0,339 | 28,41 (92 ảnh) |
| SAM2-B | 0,655 | 0,210 | 0,208 | 0,219 | 0,235 | 19,29 (92 ảnh) |

\* Nạp model khi file trọng số đã có trong cache hệ điều hành; không tính vào thời gian ảnh.

Chênh lệch theo cặp (cùng ảnh, bootstrap 95% theo ảnh, 5000 lần, seed 42):
- YOLO11n − YOLOv8n = **+3,6 ms/ảnh** [3,2; 4,1]; YOLOv8n nhanh hơn ở 94/100 ảnh.
- SAM2-B − SAM-B = **−99 ms/ảnh** [−100; −98]; SAM2-B nhanh hơn ở 92/92 ảnh.

## Kết quả: detection (YOLOv8n vs YOLO11n)

**Trên cả 100 ảnh: chỉ thống kê mô tả**, vì không có nhãn box cho đủ 100 ảnh.

| | YOLOv8n | YOLO11n |
|---|---:|---:|
| Tổng box `car` | 557 | 605 |
| Box/ảnh TB | 5,57 | 6,05 (chênh +0,48 [0,15; 0,82]) |

- Hai model trùng khớp 493 box (IoU ≥ 0,5). 64 box chỉ YOLOv8n có, 112 box chỉ YOLO11n có.
- Box thêm của YOLO11n **không** thể kết luận là đúng hay sai nếu không có nhãn.

**Trên 14 ảnh có nhãn box `car`** (`bdd100k_labels_images_train.json`; 86/100 ảnh pilot không có trong file nhãn này). Ghép tham lam theo confidence, IoU ≥ 0,5, 126 box nhãn:

| | YOLOv8n | YOLO11n | Chênh lệch (B − A), CI 95% |
|---|---:|---:|---|
| TP / FP / FN | 77 / 13 / 49 | 80 / 11 / 46 | |
| Precision | 0,856 | 0,879 | [−0,051; +0,117] |
| Recall | 0,611 | 0,635 | [−0,038; +0,092] |
| IoU TB của cặp ghép | 0,842 | 0,835 | |

Ảnh có tên trùng giữa tập 10k seg và tập 100k được coi là cùng khung hình. Box của YOLOv8n ghép với nhãn ở IoU TB 0,84, cho thấy nhãn khớp ảnh.
Chỉ là **14 ảnh**: khoảng tin cậy rộng và chứa 0, không đủ để kết luận YOLO11n chính xác hơn. Không tính mAP vì tập quá nhỏ.

## Kết quả: segmentation (SAM-B vs SAM2-B)

Nhãn tham chiếu: `*_train_id.png` của BDD100K 10k seg, có cho đủ 100 ảnh. Pixel `car` = 13; pixel 255 (vd. nắp ca-pô xe chủ) bị **bỏ qua**.
Mask dự đoán = hợp các mask SAM của ảnh. Vì nhãn là **semantic** (không tách từng xe), chỉ số là **IoU/Dice theo ảnh**, không phải theo đối tượng.

| | SAM-B | SAM2-B | Chênh lệch (B − A), CI 95% |
|---|---:|---:|---|
| IoU gộp pixel (100 ảnh) | 0,771 | 0,784 | +0,014 [−0,003; +0,031] |
| Dice gộp pixel | 0,871 | 0,879 | |
| IoU TB/ảnh (98 ảnh có IoU xác định) | 0,718 | 0,721 | +0,003 [−0,002; +0,009]; SAM2-B tốt hơn 44 ảnh, SAM-B 41, ngang 13 |
| IoU gộp **trong vùng box prompt** | 0,894 | 0,908 | |
| IoU TB/ảnh trong vùng box prompt (92 ảnh) | 0,867 | 0,869 | +0,002 [−0,004; +0,009] |

- IoU trên cả ảnh bị giới hạn bởi **recall của box prompt YOLOv8n**: xe không được YOLO bắt thì SAM không có prompt.
  Giới hạn này **như nhau** cho cả hai SAM. Trong 8 ảnh không có prompt, 6 ảnh có car trong nhãn (IoU = 0 cho cả hai) và 2 ảnh không có car (IoU không xác định).
- IoU "trong vùng box prompt" cô lập chất lượng mask, bớt phụ thuộc vào YOLO.
- Mọi khoảng tin cậy của chênh lệch đều chứa 0, nên **chưa có bằng chứng** SAM2-B chính xác hơn SAM-B trên pilot này.

> **Rà soát (2026-10-18):** hai dòng IoU đầu bảng trên tính trên 100 ảnh / 98 ảnh, **gồm 6 ảnh không có prompt** (YOLOv8n bỏ sót xe, SAM không chạy, IoU = 0 cho cả hai).
> Đánh giá đúng nghĩa "SAM có prompt" nên dùng **92 ảnh có prompt**. Tính lại bằng `scripts/audit_sam_eval.py` từ cùng đầu ra (IoU theo ảnh của SAM-B/SAM2-B trùng khớp tuyệt đối giữa hai lượt chạy):
> IoU gộp SAM-B 0,775 vs SAM2-B 0,789, chênh lệch +0,014 [−0,002; +0,032]; IoU TB/ảnh 0,765 vs 0,768, chênh lệch +0,003 [−0,002; +0,009].
> Kết luận không đổi. Chi tiết: [12_so_sanh_nhieu_model.md](12_so_sanh_nhieu_model.md#rà-soát-phép-đánh-giá-sam).

## Correction log

Correction log hiện có ghi thời gian sửa **đầu ra YOLOv8n và SAM-B gốc**. Nó **không** khớp với đầu ra của YOLO11n hay SAM2-B,
nên không dùng làm nhãn cho phần so sánh này. Thời gian sửa cũng không phải chỉ số accuracy.

## Rà soát thủ công (chưa thực hiện)

`data/real/model_comparison/manual_review_template.csv` là phiếu **trống** (200 dòng: 100 ảnh × 2 cặp model) để rà soát cùng tiêu chí cho cả hai model:
số đối tượng bị bỏ sót, số dự đoán sai, chất lượng biên (1–3), model ưu tiên. Gợi ý: ẩn tên model (A/B ngẫu nhiên) khi rà soát.
**Chưa ai điền**; báo cáo này không chứa kết quả rà soát thủ công.

## Giới hạn

- 100 ảnh, 1 lớp; 14 ảnh có nhãn detection. Khoảng tin cậy rộng cho chất lượng.
- Một máy (RTX 2060), một lần chạy, các model chạy lần lượt (không xen kẽ). Tốc độ trên phần cứng khác có thể khác.
- Thời gian nạp model đo khi file đã nằm trong cache.
- SAM được đánh giá với prompt từ YOLOv8n; kết quả không nói gì về SAM dùng prompt khác.
- IoU theo ảnh với nhãn semantic, không phải IoU/AP theo đối tượng.

## Chạy lại

```powershell
# Môi trường riêng cho model (không ảnh hưởng .venv của app)
py -3.14 -m venv .venv_models
.\.venv_models\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv_models\Scripts\python.exe -m pip install ultralytics

.\.venv_models\Scripts\python.exe scripts\compare_model_versions.py --archive <đường dẫn archive.zip BDD100K> `
    --yolo-a <yolov8n.pt> --yolo-b <yolo11n.pt> --sam-a <sam_b.pt> --sam-b <sam2_b.pt>
.\.venv\Scripts\python.exe scripts\summarize_model_versions.py     # thống kê theo cặp + CI
```

Thêm `--limit 5` để chạy thử nhanh. Kết quả nằm trong `data/real/model_comparison/`:
- `per_image_results.csv`: model, task, image_id, trạng thái, thời gian, số box/mask, IoU/Dice;
- `detections_*.csv`;
- `detection_gt_subset.csv`;
- `summary.json`, `paired_stats.json`, `run_config.json`.

Trọng số (`*.pt`) và `.venv_models/` bị `.gitignore`.
