# So sánh nhiều model YOLO và SAM trên 100 ảnh pilot (lượt 20261018-222741)

> **Dữ liệu trong repository công khai:** mỗi lượt chỉ có 4 file tổng hợp mà tab ⑥ của app đọc (`run_config.json`, `detection_summary.json`, `segmentation_summary.json`, `sam_eval_recomputed.csv`); các file này không chứa `image_id` hay toạ độ. File theo từng ảnh hoặc có toạ độ (`per_image_results.csv`, `detections_<model>.csv`, `sam_eval_image_sets.csv`, `sam_eval_audit.json`) và `model_inventory.csv`, `model_summary.csv` được **giữ local, không có trong repository công khai**.

Mở rộng của [11_so_sanh_phien_ban_model.md](11_so_sanh_phien_ban_model.md); kết quả lượt trước được giữ nguyên trong `data/real/model_comparison/`.
Lượt này nằm riêng trong `data/real/model_comparison/runs/20261018-222741/`.

**Phạm vi:** 100 ảnh BDD100K (10k seg) của pilot M31, chỉ lớp **car**; **không đại diện cho toàn bộ BDD100K**.
So YOLO với YOLO (detection) và SAM với SAM (segmentation **có prompt**); không xếp hạng chéo hai nhiệm vụ.

> **Đã rà soát (2026-10-18).** Bản trước của báo cáo tính IoU SAM trên 98 ảnh. 98 ảnh này gồm 92 ảnh SAM thực sự chạy
> **và 6 ảnh không có box prompt** (YOLOv8n bỏ sót xe), nơi IoU = 0 cho mọi SAM. Chỉ số chính nay được **tính lại trên 92 ảnh có prompt**
> bằng `scripts/audit_sam_eval.py`, từ số liệu theo ảnh, không chạy lại model và không sửa tay. Phép tính lại tái tạo đúng con số cũ trên 98 ảnh (sai khác 0).
> Bản trước cũng viết sai rằng MobileSAM "có IoU ngang SAM-B": điều đó chỉ đúng với IoU gộp; xét IoU theo ảnh, MobileSAM **thấp hơn**. Chi tiết ở phần [Rà soát phép đánh giá SAM](#rà-soát-phép-đánh-giá-sam).

> **Bổ sung:** lượt 13 YOLO + 4 SAM khác ở [13_so_sanh_model_bo_sung.md](13_so_sanh_model_bo_sung.md).

## Kết luận ngắn

Mọi so sánh chất lượng dưới đây là **khám phá**: đã thử nhiều model (8 YOLO, 6 SAM so với tham chiếu) trên mẫu nhỏ, chưa hiệu chỉnh cho so sánh nhiều lần. **Không có model nào được tuyên bố là thắng.**

- **Giữ nguyên mặc định** (YOLOv8n + SAM-B). Chưa đủ bằng chứng chất lượng để đổi, và correction log của M31 gắn với đầu ra của đúng hai model này.
- **Detection:** trên **14 ảnh / 126 box có nhãn**, yolov8s, yolo11s và yolo11m có recall cao hơn YOLOv8n với khoảng tin cậy chênh lệch không chứa 0; yolo26n có precision cao hơn nhưng recall thấp hơn.
  Tập rất nhỏ và có 16 phép so sánh, nên các tín hiệu này cần kiểm chứng trên nhiều ảnh có nhãn hơn.
  Thời gian YOLO chỉ khác nhau vài ms và **không ổn định giữa các lần chạy** (lệch tới 19%), nên không xếp hạng tốc độ YOLO.
- **Segmentation có prompt** (92 ảnh có prompt):
  - Mọi biến thể SAM 2 / 2.1 nhanh hơn SAM-B 1,5–2,5 lần ở mọi ảnh. IoU gộp cao hơn khoảng +0,012…+0,015.
    SAM2-T có khoảng tin cậy chênh lệch không chứa 0 ở cả IoU gộp (+0,015 [+0,006; +0,029]) lẫn IoU TB theo ảnh. SAM2.1-B có cận dưới sát 0 (+0,0001).
  - MobileSAM nhanh nhất (65 ms/ảnh). IoU gộp ngang SAM-B, nhưng **IoU TB theo ảnh thấp hơn** (−0,011 [−0,018; −0,005]; kém hơn ở 60/92 ảnh).
  - SAM-L chậm nhất (0,74 s/ảnh) và không cho thấy khác biệt chất lượng so với SAM-B.
  - SAM2-T là ứng viên **đáng kiểm chứng thêm** trên tập khác, không phải model đã được chứng minh là tốt hơn.

## Danh sách model đã thử

Mọi model chạy trên **cùng máy** (i5-9400, 32 GB RAM, RTX 2060 12 GB, GPU), cùng **100 ảnh**, theo thứ tự trong cột "thứ tự".

| Nhiệm vụ | Model | Weights | Nguồn weights | Thứ tự | Trạng thái |
|---|---|---|---|---:|---|
| Detection | YOLOv8n (tham chiếu, đang dùng) | `yolov8n.pt` | bàn giao của nhóm | 0 | Chạy đủ 100/100 |
| Detection | YOLO11n | `yolo11n.pt` | bàn giao của nhóm (TV1) | 1 | Chạy đủ 100/100 |
| Detection | YOLOv10n | `yolov10n.pt` | tải từ release Ultralytics (đã được duyệt) | 2 | Chạy đủ 100/100 |
| Detection | YOLO12n | `yolo12n.pt` | tải từ release Ultralytics | 3 | Chạy đủ 100/100 |
| Detection | YOLO26n | `yolo26n.pt` | tải từ release Ultralytics | 4 | Chạy đủ 100/100 |
| Detection | YOLOv8s | `yolov8s.pt` | tải từ release Ultralytics | 5 | Chạy đủ 100/100 |
| Detection | YOLO11s | `yolo11s.pt` | tải từ release Ultralytics | 6 | Chạy đủ 100/100 |
| Detection | YOLOv8m | `yolov8m.pt` | tải từ release Ultralytics | 7 | Chạy đủ 100/100 |
| Detection | YOLO11m | `yolo11m.pt` | tải từ release Ultralytics | 8 | Chạy đủ 100/100 |
| Segmentation | SAM-B (tham chiếu, đang dùng) | `sam_b.pt` | bàn giao của nhóm | 0 | Chạy đủ 92/92 ảnh có prompt |
| Segmentation | SAM 2 base | `sam2_b.pt` | tải từ release Ultralytics | 1 | Chạy đủ 92/92 |
| Segmentation | SAM 2.1 base | `sam2.1_b.pt` | tải từ release Ultralytics | 2 | Chạy đủ 92/92 |
| Segmentation | SAM 2 tiny | `sam2_t.pt` | tải từ release Ultralytics | 3 | Chạy đủ 92/92 |
| Segmentation | SAM 2 small | `sam2_s.pt` | tải từ release Ultralytics | 4 | Chạy đủ 92/92 |
| Segmentation | MobileSAM | `mobile_sam.pt` | tải từ release Ultralytics | 5 | Chạy đủ 92/92 |
| Segmentation | SAM-L | `sam_l.pt` | tải từ release Ultralytics | 6 | Chạy đủ 92/92 |

**Không chạy (không phù hợp, đã kiểm tra trên máy):**

| File | Lý do |
|---|---|
| `model.onnx` (một dự án khác) | YOLO 1 lớp `GreenSM`, không có lớp car; cần `onnxruntime` |
| `best-yolov8.onnx` (phần mềm ClassIn) | YOLO nhận diện mã QR |
| CVAT `sam/decoder.onnx` | Chỉ có decoder của SAM, thiếu encoder |
| CVAT `eomt-dinov3/model.safetensors` | Không phải SAM, không nhận box prompt |

8 ảnh không có box prompt YOLOv8n nên không chạy SAM (mask rỗng). Trong đó 6 ảnh có car trong nhãn, 2 ảnh không có car. Chỉ số SAM chính tính trên **92 ảnh có prompt**; xem [Rà soát phép đánh giá SAM](#rà-soát-phép-đánh-giá-sam).

## Phương pháp và cấu hình

- **Phần mềm:** Python 3.14.2, torch 2.14.1+cu126, ultralytics 8.4.182, OpenCV 5.0.0 (`.venv_models`).
- **YOLO:** `imgsz=640`, `conf=0.25`, batch 1, NMS theo lớp IoU 0,7 (cả 9 model, gồm YOLOv10n/YOLO26n với `end2end=False`); dự đoán mọi lớp COCO rồi lọc `car` (giống pipeline gốc).
- **SAM có prompt:** mọi SAM nhận **cùng 555 box YOLOv8n đã bàn giao**. Mask dự đoán = hợp các mask của ảnh.
- **Thời gian:**
  - wall-clock quanh một lời gọi model trên ảnh đã giải mã sẵn, có `torch.cuda.synchronize()`;
  - **không** tính nạp model, 3 ảnh khởi động, giải mã ảnh và tính chỉ số;
  - nạp model đo riêng, khi file đã nằm trong cache của hệ điều hành.
- **Nhãn tham chiếu:**
  - **Segmentation:** BDD100K `*_train_id.png`, car = 13, bỏ qua pixel 255. có cho 100/100 ảnh. **Chỉ số chính trên 92 ảnh có prompt**; chỉ số đầu-cuối trên 98 ảnh có IoU xác định. Chỉ số theo ảnh (nhãn semantic), không theo đối tượng.
  - **Detection:** `bdd100k_labels_images_train.json`, category `car`. Chỉ **14/100 ảnh (126 box)** có nhãn; 86 ảnh còn lại **không được chấm** và **không** bị coi là model sai.
    Ghép tham lam theo confidence, IoU ≥ 0,5. Không tính mAP (tập quá nhỏ).
- **Khoảng tin cậy 95%:** bootstrap theo ảnh, 2000 lần, seed 42. Chênh lệch so với tham chiếu được tính **theo cặp** trên cùng ảnh lấy mẫu lại.
- **Tái lập:** YOLOv8n và YOLO11n cho **đúng cùng số box** (557, 605) như lượt trước; SAM-B và SAM2-B cho cùng IoU.

## Kết quả: detection (tham chiếu YOLOv8n)

P/R trên 14 ảnh có nhãn, 126 box car. Chênh lệch là model − YOLOv8n; CI 95% trong ngoặc.

| Model | ms/ảnh (TB) | Box car (100 ảnh) | TP/FP/FN | Precision | Recall | ΔPrecision | ΔRecall | Trùng box với YOLOv8n / chỉ YOLOv8n / chỉ model |
|---|---:|---:|---|---|---|---|---|---|
| YOLOv8n | 14,5 | 557 | 77/13/49 | 0,856 [0,787; 0,925] | 0,611 [0,537; 0,711] | — | — | — |
| YOLO11n | 20,5 | 605 | 80/11/46 | 0,879 [0,812; 0,952] | 0,635 [0,535; 0,739] | [−0,051; +0,119] | [−0,039; +0,090] | 493 / 64 / 112 |
| YOLOv10n | 16,2 | 527 | 72/11/54 | 0,867 [0,802; 0,932] | 0,571 [0,477; 0,667] | [−0,045; +0,073] | [−0,098; +0,010] | 451 / 106 / 76 |
| YOLO12n | 28,1 | 559 | 77/11/49 | 0,875 [0,816; 0,938] | 0,611 [0,516; 0,716] | [−0,050; +0,109] | [−0,053; +0,047] | 464 / 93 / 95 |
| YOLO26n | 21,0 | 517 | 73/5/53 | 0,936 [0,885; 0,986] | 0,579 [0,454; 0,727] | **[+0,024; +0,148]** | [−0,092; +0,032] | 451 / 106 / 66 |
| YOLOv8s | 16,3 | 673 | 84/16/42 | 0,840 [0,768; 0,921] | 0,667 [0,591; 0,759] | [−0,058; +0,034] | **[+0,017; +0,091]** | 504 / 53 / 169 |
| YOLO11s | 19,1 | 673 | 85/15/41 | 0,850 [0,787; 0,920] | 0,675 [0,609; 0,754] | [−0,055; +0,047] | **[+0,012; +0,101]** | 493 / 64 / 180 |
| YOLOv8m | 18,5 | 706 | 82/17/44 | 0,828 [0,750; 0,918] | 0,651 [0,556; 0,758] | [−0,096; +0,066] | [−0,016; +0,099] | 489 / 68 / 217 |
| YOLO11m | 22,1 | 676 | 88/8/38 | 0,917 [0,868; 0,963] | 0,698 [0,620; 0,786] | [−0,004; +0,137] | **[+0,033; +0,137]** | 481 / 76 / 195 |

**Đọc kết quả:**
- **In đậm** = CI 95% của chênh lệch không chứa 0. Có 16 phép so sánh (8 model × 2 chỉ số) trên **cùng 14 ảnh**, chưa hiệu chỉnh cho so sánh nhiều lần.
  Vì vậy cần coi đây là **tín hiệu khám phá**: model cỡ s/m có xu hướng recall cao hơn; YOLO26n ít FP hơn nhưng bỏ sót nhiều hơn.
- Recall của mọi model chỉ khoảng 0,57–0,70, vì nhãn BDD100K có nhiều xe nhỏ hoặc bị che mà COCO-YOLO ở `imgsz=640` khó bắt.
- Số box nhiều hơn (vd. YOLOv8m: 706) **không** có nghĩa là tốt hơn. 86 ảnh không có nhãn, nên không biết các box thêm là đúng hay sai.
- **Thời gian YOLO không đủ ổn định để xếp hạng:**
  - Cùng YOLO11n, lượt trước đo 17,3 ms/ảnh, lượt này 20,5 ms (lệch 19%); YOLOv8n lệch 6%.
  - Ở batch 1, thời gian nano/small bị chi phối bởi chi phí cố định (tiền/hậu xử lý, khởi chạy kernel), nên model lớn hơn (YOLOv8m) có thể không chậm hơn model nhỏ.
  - Chênh lệch theo cặp vẫn có CI hẹp *trong một lượt* (vd. YOLO11m +7,6 ms [7,2; 8,1]), nhưng không lặp lại ổn định giữa các lượt.

## Kết quả: segmentation có prompt (tham chiếu SAM-B)

### Chỉ số chính: 92 ảnh có box prompt (mọi SAM đã chạy)

Tính lại bằng `scripts/audit_sam_eval.py` từ `per_image_results.csv`; CI 95% bootstrap theo ảnh **trong 92 ảnh**, 2000 lần, seed 42; chênh lệch theo cặp (model − SAM-B).
1 ảnh có prompt nhưng nhãn không có car (YOLOv8n bắt nhầm) nên IoU = 0 cho mọi SAM; ảnh này vẫn nằm trong 92 ảnh.

| Model | s/ảnh (TB) | IoU gộp [CI] | Dice gộp | IoU TB/ảnh [CI] | ΔIoU gộp [CI] | ΔIoU TB/ảnh [CI] | Ảnh tốt hơn / kém hơn / ngang* |
|---|---:|---|---:|---|---|---|---|
| SAM-B | 0,310 | 0,775 [0,693; 0,844] | 0,873 | 0,765 [0,723; 0,803] | — | — | — |
| SAM2-B | 0,211 | 0,789 [0,707; 0,859] | 0,882 | 0,768 [0,724; 0,808] | +0,014 [−0,002; +0,032] | +0,003 [−0,002; +0,009] | 44 / 41 / 7 |
| SAM2.1-B | 0,212 | 0,790 [0,709; 0,859] | 0,883 | 0,770 [0,725; 0,811] | +0,015 [+0,0001; +0,031] | +0,005 [−0,001; +0,011] | 50 / 34 / 8 |
| SAM2-T | 0,122 | 0,790 [0,710; 0,861] | 0,883 | 0,771 [0,728; 0,811] | **+0,015 [+0,006; +0,029]** | **+0,006 [+0,001; +0,011]** | 46 / 41 / 5 |
| SAM2-S | 0,138 | 0,787 [0,705; 0,856] | 0,881 | 0,769 [0,726; 0,808] | +0,012 [−0,004; +0,029] | +0,004 [−0,002; +0,009] | 50 / 35 / 7 |
| MobileSAM | 0,065 | 0,778 [0,694; 0,847] | 0,875 | 0,753 [0,712; 0,792] | +0,003 [−0,011; +0,023] | **−0,011 [−0,018; −0,005]** | 26 / 60 / 6 |
| SAM-L | 0,744 | 0,776 [0,694; 0,849] | 0,874 | 0,768 [0,724; 0,807] | +0,001 [−0,014; +0,012] | +0,003 [−0,003; +0,008] | 59 / 20 / 13 |

\* Theo IoU từng ảnh so với SAM-B, ngưỡng ngang ±0,001.

IoU **trong vùng box prompt** (cô lập chất lượng mask khỏi phần xe không có prompt) chỉ tính được trên 92 ảnh có prompt, nên không đổi so với bản trước:
SAM-B 0,894; SAM2-B 0,908; SAM2.1-B 0,910; SAM2-T 0,911; SAM2-S 0,907; MobileSAM 0,895; SAM-L 0,895.
CI của chỉ số này trong `segmentation_summary.json` được bootstrap trên 100 ảnh, trong đó 8 ảnh không đóng góp gì.

### Chỉ số đầu-cuối: 98 ảnh có IoU xác định (gồm lỗi bỏ sót của prompt)

= 92 ảnh trên + 6 ảnh **không có prompt** (YOLOv8n không bắt được xe nào) nhưng nhãn có car.
Ở 6 ảnh này SAM **không chạy**; IoU = 0 cho mọi SAM, phản ánh việc YOLOv8n bỏ sót xe.
2 ảnh không có prompt và nhãn cũng không có car có IoU 0/0 nên không chấm. Đây là cách tính của bản trước; phép tính lại tái tạo đúng các con số đó (sai khác 0).

| Model | IoU gộp | IoU TB/ảnh | ΔIoU gộp so với SAM-B [CI, bootstrap trong 98 ảnh] |
|---|---:|---:|---|
| SAM-B | 0,771 | 0,718 | — |
| SAM2-B | 0,784 | 0,721 | +0,014 [−0,002; +0,031] |
| SAM2.1-B | 0,786 | 0,723 | +0,015 [+0,0002; +0,031] |
| SAM2-T | 0,786 | 0,723 | +0,015 [+0,006; +0,028] |
| SAM2-S | 0,783 | 0,722 | +0,012 [−0,005; +0,029] |
| MobileSAM | 0,774 | 0,707 | +0,003 [−0,011; +0,022] |
| SAM-L | 0,771 | 0,721 | +0,001 [−0,013; +0,012] |

Bản trước bootstrap trên 100 ảnh (gồm 2 ảnh không chấm), nên CI hơi khác bảng này. Kết luận không đổi.

### Đọc kết quả

- **Tốc độ SAM ổn định:** SAM-B và SAM2-B lệch dưới 1% so với lượt trước. Mọi SAM 2 / MobileSAM nhanh hơn SAM-B ở **92/92 ảnh**; SAM-L chậm hơn ở 92/92 ảnh.
- **Chất lượng (khám phá):** khoảng cách nhỏ (IoU gộp 0,775–0,790 trên 92 ảnh).
  - SAM2-T là model duy nhất có CI chênh lệch không chứa 0 ở **cả** IoU gộp lẫn IoU TB/ảnh.
  - SAM2.1-B có cận dưới IoU gộp sát 0 (+0,0001), nên không đủ để kết luận.
  - MobileSAM: IoU gộp ngang, nhưng **IoU TB/ảnh thấp hơn** và kém hơn ở 60/92 ảnh. Có thể MobileSAM tốt ở xe lớn (chiếm nhiều pixel) nhưng kém ở xe nhỏ; chưa kiểm chứng.
  - Có 12 phép so sánh (6 model × 2 chỉ số), chưa hiệu chỉnh cho so sánh nhiều lần, nên **không tuyên bố model nào thắng**.
- **Phụ thuộc prompt YOLOv8n:** xe không được YOLO bắt thì không có mask, như nhau cho mọi SAM. Kết quả không nói gì về SAM dùng prompt khác.

## Rà soát phép đánh giá SAM

| Câu hỏi | Kết quả rà soát |
|---|---|
| Vì sao SAM chỉ chạy 92 ảnh? | 8/100 ảnh không có box YOLOv8n nào trong `sam_predictions_100.csv` (YOLOv8n không bắt được car), nên không có prompt. Script ghi `no_prompt` và dùng mask rỗng |
| 98 ảnh được chấm là những ảnh nào? | 92 ảnh có prompt + 6 ảnh không prompt nhưng nhãn có car. Danh sách theo `image_id`: `sam_eval_image_sets.csv` (giữ local) |
| Xử lý ảnh không có prompt/mask? | Mask dự đoán = rỗng. Nếu nhãn có car thì giao = 0, hợp = số pixel car, IoU = 0. Nếu nhãn không có car thì 0/0, không chấm |
| Mẫu số có nhất quán giữa các model không? | Có: cả 7 SAM dùng cùng 92 ảnh chạy và cùng 98 ảnh có IoU (script kiểm tra bằng `assert`) |
| Chênh lệch theo cặp/CI có khớp dữ liệu? | Có: chênh lệch tính trên cùng ảnh lấy mẫu lại. Tính lại trên 98 ảnh khớp đúng IoU gộp đã báo (sai khác 0) |
| Đã sửa gì? | Chỉ số chính chuyển sang **92 ảnh có prompt** (đúng nghĩa "SAM có prompt"); 98 ảnh giữ làm chỉ số đầu-cuối; sửa nhận định sai về MobileSAM |

### Rà soát cấu hình YOLO

| Mục | Kết quả kiểm tra (cả 9 model) |
|---|---|
| Ngưỡng confidence | `conf=0.25`; confidence nhỏ nhất trong các file `detections_*.csv` đều ≥ 0,250 |
| Kích thước đầu vào | `imgsz=640`, batch 1 |
| NMS | Ultralytics 8.4.182, NMS theo lớp (`agnostic_nms=False`), `iou=0.7`, `max_det=300`. Cả 9 model chạy với `end2end=False`, tức đều qua NMS. YOLOv10n và YOLO26n có nhánh one-to-one cho chế độ không NMS, nhưng chế độ đó **không** được dùng ở đây; kết quả của hai model này có thể khác ở chế độ gốc. Không có cặp box `car` nào trùng nhau với IoU > 0,7 |
| Lọc lớp | Lớp `car` (COCO id 2, kiểm tra theo tên ở cả 9 model), lọc **sau** NMS theo lớp |
| Tập chấm | Cùng **14 ảnh / 126 box** có nhãn cho mọi model |

## Đề xuất về model mặc định

| | Đề xuất | Lý do |
|---|---|---|
| Detection | **Giữ YOLOv8n** | Lợi thế recall của s/m chỉ thấy trên 14 ảnh, 16 phép so sánh; thời gian không ổn định để cân nhắc đánh đổi; correction log hiện tại gắn với đầu ra YOLOv8n |
| Segmentation | **Giữ SAM-B**; ghi nhận **SAM2-T** là ứng viên cần kiểm chứng | SAM2-T nhanh hơn khoảng 2,5 lần ở mọi ảnh, và CI chênh lệch IoU trên 92 ảnh có prompt không chứa 0. Nhưng đây là 1 trong 12 phép so sánh trên 92 ảnh của một lớp; cần kiểm chứng trên tập khác và thu correction log mới trước khi đổi |

Không sửa model mặc định trong app hay pipeline.

## Giới hạn

- 100 ảnh, 1 lớp; nhãn detection chỉ có cho 14 ảnh; IoU semantic theo ảnh, không theo đối tượng.
- Một máy, một lượt cho đa số model, chạy lần lượt (không xen kẽ). Thời gian YOLO dao động giữa các lượt.
- Nạp model đo khi đã có cache; lần nạp đầu tiên chậm hơn (vd. SAM-B khoảng 5 s ở lần chạy thử).
- Đánh giá SAM phụ thuộc prompt YOLOv8n; SAM-L/MobileSAM/SAM 2 có thể cho kết quả khác với prompt khác.
- Không hiệu chỉnh cho so sánh nhiều lần (16 so sánh YOLO, 12 so sánh SAM); các khoảng tin cậy không chứa 0 chỉ là tín hiệu khám phá.
- YOLOv10n/YOLO26n chạy qua NMS, không ở chế độ không-NMS gốc.
- Chưa có rà soát thủ công (phiếu mẫu trong `manual_review_template.csv` vẫn trống).

## File kết quả (`data/real/model_comparison/runs/20261018-222741/`)

| File | Nội dung |
|---|---|
| `model_inventory.csv` | Mọi model đã thử: weights, thứ tự chạy, trạng thái, số ảnh chạy/lỗi, thời gian nạp, tổng/TB/trung vị/P90/max |
| `per_image_results.csv` | Theo ảnh × model: trạng thái, thời gian, số box/prompt/mask, IoU/Dice, IoU trong box |
| `detections_<model>.csv` | Box `car` của từng YOLO |
| `detection_summary.json`, `segmentation_summary.json` | Chỉ số, CI, chênh lệch so với tham chiếu |
| `model_summary.csv` | Bảng tổng hợp phẳng; cột `metric_eval_set` ghi tập ảnh dùng cho từng chỉ số |
| `sam_eval_image_sets.csv` | 100 ảnh: có prompt hay không, số pixel car, thuộc tập 92 / 98, ghi chú |
| `sam_eval_recomputed.csv` | Chỉ số SAM tính lại trên tập 92 (chính) và 98 (đầu-cuối), CI, chênh lệch theo cặp, số ảnh tốt hơn/kém hơn |
| `sam_eval_audit.json` | Danh sách ảnh của từng tập, kiểm tra tái tạo số cũ |
| `run_config.json` | Phần cứng, phần mềm, cấu hình |

> Trong repository công khai chỉ có 4 file tổng hợp ở trên (`run_config.json`, `detection_summary.json`, `segmentation_summary.json`, `sam_eval_recomputed.csv`); các file còn lại trong bảng được giữ local.

## Chạy lại

```powershell
$W = "<thư mục weights>"
.\.venv_models\Scripts\python.exe scripts\compare_model_zoo.py --archive <archive.zip BDD100K> `
  --yolo $W\yolov8n.pt $W\yolo11n.pt $W\yolov10n.pt $W\yolo12n.pt $W\yolo26n.pt $W\yolov8s.pt $W\yolo11s.pt $W\yolov8m.pt $W\yolo11m.pt `
  --sam  $W\sam_b.pt $W\sam2_b.pt $W\sam2.1_b.pt $W\sam2_t.pt $W\sam2_s.pt $W\mobile_sam.pt $W\sam_l.pt
```

Model đầu tiên trong mỗi danh sách là tham chiếu. Mỗi lần chạy tạo thư mục `runs/<thời điểm>/` mới (khoảng 4 phút cho 16 model trên RTX 2060).
Weights tải thêm nằm trong `data/real/model_comparison/weights/` (bị `.gitignore`); tải lại bằng
`ultralytics.utils.downloads.attempt_download_asset("<tên>.pt")` trong `.venv_models`.
