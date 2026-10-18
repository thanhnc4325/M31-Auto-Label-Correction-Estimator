# Bài toán, giải pháp, kết quả pilot và giới hạn

## 1. Bài toán

Khi dùng model (YOLO cho detection, SAM cho segmentation) để gán nhãn trước (pre-label) cho ảnh BDD100K,
con người vẫn phải review và sửa. Nhóm cần biết **ảnh nào nên review trước** và **tốn bao nhiêu công sửa**
để lập kế hoạch gán nhãn.

M31 nhận đầu vào là **thông tin pre-label** (có sẵn trước khi con người sửa) và trả về cho từng ảnh:

| Đầu ra | Ý nghĩa |
|---|---|
| `predicted_quality_score` | 0–100, ước lượng thử nghiệm về chất lượng pre-label |
| `estimated_correction_time_sec` | thời gian sửa dự kiến (giây) |
| `priority` | thấp / vừa / cao |
| `reason` | giải thích ngắn dựa trên đặc trưng đầu vào |

**Ràng buộc:** ground truth, thời gian sửa thật, số thao tác sửa, trạng thái review và ghi chú của người sửa
chỉ có **sau** khi con người sửa, nên **không được dùng làm đầu vào**. Chúng chỉ dùng để đánh giá.

## 2. Giải pháp hiện tại

1. **Đặc trưng pre-label**: số dự đoán, confidence trung bình/nhỏ nhất, số dự đoán có confidence < 0.5,
   số box nhỏ, diện tích box trung bình; với SAM thêm tỉ lệ lấp đầy mask trong box prompt và số mask rỗng.
2. **Điểm chất lượng** (công thức cố định, đặt **trước** khi có dữ liệu thật):
   `100 × (0.6·mean_confidence + 0.4·(1 − low_conf/num_pred))`; ảnh không có dự đoán = 50.
3. **Thời gian sửa (baseline tạm thời)**: `base + review·num_predictions + fix·low_confidence_count`,
   với hằng số **giả định** theo task (`m31/baseline.py`).
4. **Ưu tiên**: cao nếu điểm < 50 hoặc không có dự đoán; vừa nếu < 75; còn lại thấp (chỉnh được trên giao diện).
5. **Đối chiếu với correction log** (`m31/calibration.py`): MAE của baseline; hồi quy ridge theo task với
   5-fold cross-validation; **mốc hằng số** (đoán median của fold huấn luyện) để biết model có hơn đoán mò không;
   tương quan Spearman; bảng mức ưu tiên × ảnh thực sự cần sửa.
   Mô hình hiệu chỉnh chỉ được **bật** khi mỗi task có ≥ 20 ảnh có thời gian sửa > 0.

## 3. Dữ liệu pilot

Nguồn: `end_corrected.zip` do nhóm bàn giao (SHA-256 trong `data/real/processed/pilot100_manifest.json`).

- 100 ảnh BDD100K (split train, 1280×720), chỉ lớp **car**.
- YOLOv8n: 555 box trên 92 ảnh; 8 ảnh không có box nào.
- SAM-B: 555 mask, prompt bằng chính box YOLO. **Confidence của SAM là confidence của box YOLO**,
  SAM không có điểm riêng.
- Correction log: 100 dòng mỗi task. Chỉ **13/100 ảnh mỗi task có thời gian sửa > 0**
  (YOLO: tổng 117 s, TB 1,17 s/ảnh; SAM: tổng 168,5 s, TB 1,685 s/ảnh). Không có thao tác *add* nào.
- Theo `README_end.md` của gói: thời gian sửa do nhóm đo trực tiếp và không kiểm chứng độc lập được từ file;
  gói không có precision/recall/IoU theo đối tượng.

## 4. Kết quả trên pilot (đo ngày 2026-10-18)

Tính bằng ứng dụng trên `data/real/processed/` (200 dòng khớp, ngưỡng ưu tiên mặc định 50/75).

| Task | MAE baseline | MAE ridge (5-fold CV) | MAE mốc hằng số (CV) | Spearman(điểm chất lượng, thời gian sửa) |
|---|---:|---:|---:|---:|
| detection | 36,49 s | 2,00 s | **1,17 s** | −0,023 |
| segmentation | 71,36 s | 2,86 s | **1,69 s** | −0,035 |

| Task | Ưu tiên | Số ảnh | Ảnh cần sửa thật | Tỉ lệ |
|---|---|---:|---:|---:|
| detection / segmentation | cao | 27 | 4 | 14,8% |
| detection / segmentation | vừa | 51 | 6 | 11,8% |
| detection / segmentation | thấp | 22 | 3 | 13,6% |

(Hai task cho cùng phân bố ưu tiên vì SAM dùng lại confidence của YOLO.)

**Diễn giải trung thực:**
- Baseline giả định trung bình 37,7 s (YOLO) và 73 s (SAM) mỗi ảnh, trong khi thực tế 1,17 s và 1,69 s:
  **sai thang đo khoảng 30–40 lần**.
- Hồi quy ridge trên 3 đặc trưng **không** tốt hơn việc luôn đoán 0 giây.
- Điểm chất lượng và mức ưu tiên **chưa phân biệt được** ảnh cần sửa: tỉ lệ cần sửa gần như như nhau ở mọi mức.
- Theo ghi chú của người sửa, lỗi thường là box/mask bắt nhầm nắp ca-pô xe chủ hoặc viền mask lan sang xe kề;
  confidence trung bình không phản ánh được các lỗi này.
- Vì vậy M31 **giữ baseline**, **không** bật mô hình hiệu chỉnh và **không** tuyên bố độ chính xác.
  Công thức không được tinh chỉnh lại trên pilot, vì 100 ảnh này là dữ liệu đánh giá duy nhất.

### Cập nhật: Ridge đã huấn luyện (2026-10-18)

Đã train Ridge thật cho từng task trên 100 ảnh, giữ cả ảnh có thời gian sửa = 0, CV 5-fold theo nhóm image_id × 20 lần (seed 42).
MAE Ridge 2,007 s / 2,885 s so với luôn đoán 0 giây 1,170 s / 1,685 s (detection / segmentation). Ridge thắng ở 0% số lần lặp.
Mô hình có trong app dưới dạng lựa chọn thử nghiệm; chi tiết ở [09_mo_hinh_ridge.md](09_mo_hinh_ridge.md).

### Cập nhật: So sánh phiên bản model YOLO / SAM (pilot, 2026-10-18)

> **So sánh model (pilot 100 ảnh BDD100K, lớp car, RTX 2060):** đã thử 22 phiên bản YOLO và 11 phiên bản SAM trên cùng tập pilot, với cấu hình thống nhất trong từng nhóm model. Với cùng box prompt YOLOv8n, SAM2-T/S chạy nhanh hơn SAM-B khoảng 2,2–2,5 lần; IoU trên 92 ảnh có prompt cao hơn nhẹ (+0,012…+0,016), nhưng kết quả chỉ mang tính khám phá. Chất lượng detection chưa xếp hạng được một cách đáng tin cậy vì chỉ 14 ảnh (126 box) có nhãn. Nhóm giữ YOLOv8n + SAM-B làm mặc định; các ứng viên cần được kiểm chứng trên tập ảnh mới có nhãn.

**Model đã thử** (cùng 100 ảnh pilot, cùng máy i5-9400 / RTX 2060 12 GB, GPU):

- **22 YOLO:** YOLOv8 n/s/m/l/x, YOLO11 n/s/m/l/x, YOLOv9 t/s/c, YOLOv10 n/s/m, YOLO12 n/s/m, YOLO26 n/s/m.
- **11 SAM:** SAM-B, SAM-L, MobileSAM, SAM2 t/s/b/l, SAM2.1 t/s/b/l. Không chạy SAM-H và FastSAM.
- Mọi model chạy đủ: YOLO 100/100 ảnh; SAM 92/92 ảnh có prompt.

**Phương pháp và số mẫu:**

| Hạng mục | Cấu hình / số ảnh dùng |
|---|---|
| YOLO | `imgsz` 640, `conf` 0,25, NMS theo lớp IoU 0,7, `max_det` 300, lọc lớp `car` sau NMS; YOLOv10/YOLO26 cũng chạy qua NMS (`end2end=False`) |
| Precision / recall detection | **14 ảnh, 126 box car** có nhãn trong `bdd100k_labels_images_train.json`; 86 ảnh còn lại không chấm và không bị coi là sai |
| SAM | **Đánh giá với box prompt từ YOLOv8n**: mọi SAM nhận cùng 555 box YOLOv8n đã bàn giao |
| IoU / Dice SAM | Nhãn segmentation BDD100K (car = 13, bỏ qua pixel 255); **chỉ số chính trên 92 ảnh có prompt**. 8 ảnh không có prompt (YOLOv8n không bắt được xe) không chạy SAM |
| Thời gian | Mỗi lời gọi model trên ảnh đã giải mã, có đồng bộ GPU; không tính nạp model và 3 ảnh khởi động |
| Khoảng tin cậy | Bootstrap 95% theo ảnh, 2000 lần, seed 42; chênh lệch so với tham chiếu tính theo cặp |

**Chi tiết SAM2-T/S so với SAM-B (92 ảnh có prompt):**

| Model | Thời gian/ảnh | Nhanh hơn SAM-B | ΔIoU gộp so với SAM-B [CI 95%] |
|---|---:|---:|---|
| SAM-B (tham chiếu) | 0,308–0,310 s | — | — |
| SAM2-T | 0,122 s | 2,5× | +0,015 [+0,006; +0,029] |
| SAM2.1-T | 0,121 s | 2,5× | +0,013 [+0,007; +0,022] |
| SAM2-S | 0,138 s | 2,2× | +0,012 [−0,004; +0,029] |
| SAM2.1-S | 0,136 s | 2,3× | +0,016 [+0,001; +0,033] |

Dải +0,012…+0,016 trong đoạn tóm tắt bao gồm cả bốn model T/S (thấp nhất là SAM2-S +0,012, cao nhất là SAM2.1-S +0,016).
SAM 2 / 2.1 cỡ L chậm hơn SAM-B khoảng 1,5 lần và không cho thấy chất lượng cao hơn cỡ T/S.

**Giới hạn của phần so sánh:**

- Pilot 100 ảnh, chỉ lớp car, một máy (RTX 2060). Kết quả **không đại diện cho toàn bộ BDD100K** và có thể khác trên phần cứng khác.
- Nhiều phép so sánh trên mẫu nhỏ (42 so sánh YOLO trên cùng 14 ảnh; 20 so sánh SAM trên cùng 92 ảnh), chưa hiệu chỉnh.
  Khoảng tin cậy không chứa 0 chỉ là tín hiệu khám phá; **không có model nào được coi là tốt nhất**.
- IoU là IoU theo ảnh với nhãn semantic, không theo đối tượng. Chất lượng SAM phụ thuộc prompt YOLOv8n.
- Thời gian YOLO dao động giữa các lượt chạy (tới 19%), nên không xếp hạng tốc độ YOLO.
- Correction log hiện có chỉ ứng với đầu ra YOLOv8n / SAM-B. Muốn đổi model mặc định cần thu correction log mới.

Chi tiết đầy đủ: [11_so_sanh_phien_ban_model.md](11_so_sanh_phien_ban_model.md), [12_so_sanh_nhieu_model.md](12_so_sanh_nhieu_model.md),
[13_so_sanh_model_bo_sung.md](13_so_sanh_model_bo_sung.md).

## 5. Giới hạn

- Pilot nhỏ (100 ảnh, 1 lớp, 13 ca có sửa mỗi task); không đại diện cho BDD100K.
- Ngưỡng confidence thấp (0.5) và định nghĩa box nhỏ (32×32 px) do M31 chọn, không phải do nhóm bàn giao.
- Hằng số baseline là giả định; điểm chất lượng chưa được kiểm chứng.
- Không có đánh giá với ground truth (gói không có bảng đánh giá theo đối tượng).
- Thời gian sửa không kiểm chứng độc lập được.

## 6. Hướng tiếp theo (chưa làm)

- Thu thêm correction log (đặc biệt các ảnh cần sửa) cho đến khi mỗi task có ≥ 20 ca, rồi đánh giá lại trên
  tập **mới** chưa dùng để thiết kế.
- Thêm đặc trưng nhắm vào lỗi quan sát được (vd. box chạm đáy ảnh, box/mask chồng lấn) và kiểm tra trên tập mới.
- Tách mô hình thành hai bước: phân loại "có cần sửa không", rồi ước lượng thời gian cho ảnh cần sửa.
