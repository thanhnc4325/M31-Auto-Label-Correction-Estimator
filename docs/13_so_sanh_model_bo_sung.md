# So sánh model bổ sung: 13 YOLO và 4 SAM (lượt 20261018-231455)

> **Dữ liệu trong repository công khai:** lượt này chỉ có 4 file tổng hợp mà tab ⑥ của app đọc (`run_config.json`, `detection_summary.json`, `segmentation_summary.json`, `sam_eval_recomputed.csv`); các file này không chứa `image_id` hay toạ độ. File theo từng ảnh hoặc có toạ độ (`per_image_results.csv`, `detections_<model>.csv`, `sam_eval_image_sets.csv`, `sam_eval_audit.json`) và `model_inventory.csv`, `model_summary.csv` được **giữ local, không có trong repository công khai**.

Bổ sung cho [12_so_sanh_nhieu_model.md](12_so_sanh_nhieu_model.md); không ghi đè lượt trước.
Kết quả: `data/real/model_comparison/runs/20261018-231455/`.

> **KẾT QUẢ CHẤT LƯỢNG LÀ KHÁM PHÁ.** Precision/recall detection chỉ chấm trên **14 ảnh / 126 box có nhãn**. Riêng lượt này có 26 phép so sánh YOLO và 8 phép so sánh SAM;
> tính cả lượt trước là 42 phép so sánh YOLO trên **cùng 14 ảnh** và 20 phép so sánh SAM trên cùng 92 ảnh. Không hiệu chỉnh cho so sánh nhiều lần.
> Một khoảng tin cậy không chứa 0 **không** đủ để kết luận model nào tốt hơn. **Không model nào được tuyên bố là thắng; mặc định giữ YOLOv8n + SAM-B.**

## Model đã chạy

Cùng máy (i5-9400, 32 GB RAM, RTX 2060 12 GB, GPU), cùng 100 ảnh pilot, cùng cấu hình như lượt trước. Tham chiếu chạy lại trong cùng phiên.

| Nhiệm vụ | Model | Weights (MB) | Trạng thái |
|---|---|---:|---|
| Detection | YOLOv8n (tham chiếu) | 6,5 | 100/100 |
| Detection | YOLOv8l, YOLO11l | 87,8 / 51,4 | 100/100 |
| Detection | YOLO26s, YOLO26m | 20,4 / 44,3 | 100/100 |
| Detection | YOLO12s, YOLO12m | 19,0 / 40,9 | 100/100 |
| Detection | YOLOv10s, YOLOv10m | 16,6 / 33,6 | 100/100 |
| Detection | YOLOv8x, YOLO11x | 136,9 / 114,6 | 100/100 |
| Detection | YOLOv9t, YOLOv9s, YOLOv9c | 5,0 / 15,4 / 51,8 | 100/100 |
| Segmentation | SAM-B (tham chiếu) | 375 | 92/92 ảnh có prompt |
| Segmentation | SAM2.1-T, SAM2.1-S | 78,1 / 92,3 | 92/92 |
| Segmentation | SAM2.1-L, SAM2-L | 449,2 / 449,2 | 92/92 |

Không model nào lỗi. Theo yêu cầu, **không chạy** `sam_h` và FastSAM.

**Kiểm tra trùng checkpoint:**
- `sam_l.pt` (lượt trước) là SAM thế hệ 1, encoder ViT-L: 24 block, chiều 1024, 312 triệu tham số, không có trunk Hiera.
- `sam2_l` và `sam2.1_l` là SAM 2 / 2.1 với encoder Hiera-L.
- SHA-256 của cả 33 file trọng số trên máy đều khác nhau. `sam2_l` và `sam2.1_l` cùng dung lượng 449,2 MB nhưng hash khác nhau (`fd618bcf…` và `ab7e1ac9…`).

**Tái lập:** YOLOv8n cho lại đúng P/R 0,856/0,611 (14 ảnh). SAM-B cho lại đúng IoU gộp 0,771 (98 ảnh) và 0,775 (92 ảnh).

## Cấu hình và số ảnh dùng cho từng chỉ số

| Chỉ số | Số ảnh thực sự dùng |
|---|---|
| Thời gian YOLO, số box, trùng khớp với YOLOv8n | 100 ảnh (không cần nhãn) |
| Precision / recall YOLO | **14 ảnh, 126 box car** có nhãn trong `bdd100k_labels_images_train.json`; 86 ảnh không chấm, **không** bị coi là sai |
| Thời gian SAM | 92 ảnh có box prompt |
| IoU / Dice SAM, **chỉ số chính** | **92 ảnh có prompt** (nhãn seg BDD100K, car = 13, bỏ qua 255) |
| IoU SAM đầu-cuối | 98 ảnh = 92 + 6 ảnh không prompt nhưng có car (IoU = 0 cho mọi SAM) |

Cấu hình YOLO đã kiểm tra trên cả 13 model mới:
- `imgsz` 640, `conf` 0,25 (confidence nhỏ nhất ≥ 0,250);
- NMS theo lớp, IoU 0,7, `max_det` 300; lọc `car` (COCO id 2) sau NMS;
- không có cặp box trùng nhau với IoU > 0,7.

YOLO26s/m và YOLOv10s/m có nhánh one-to-one (chế độ không NMS), nhưng chạy với `end2end=False`, tức **cũng qua NMS** như các model khác.

## Kết quả detection (khám phá; tham chiếu YOLOv8n)

P/R trên 14 ảnh, 126 box; CI 95% bootstrap theo ảnh (2000 lần, seed 42); Δ = model − YOLOv8n theo cặp.
**In đậm** = CI của chênh lệch không chứa 0 (cận bằng đúng 0,000 không in đậm). Thời gian là TB ms/ảnh trong **lượt này**.

| Model | ms/ảnh | Box car (100 ảnh) | TP/FP/FN | Precision [CI] | Recall [CI] | ΔPrecision [CI] | ΔRecall [CI] | Trùng / chỉ YOLOv8n / chỉ model |
|---|---:|---:|---|---|---|---|---|---|
| YOLOv8n | 13,8 | 557 | 77/13/49 | 0,856 [0,787; 0,925] | 0,611 [0,537; 0,711] | — | — | — |
| YOLOv8l | 23,3 | 724 | 84/12/42 | 0,875 [0,828; 0,926] | 0,667 [0,571; 0,764] | [−0,059; +0,098] | [−0,009; +0,118] | 480 / 77 / 244 |
| YOLO11l | 30,1 | 736 | 88/15/38 | 0,854 [0,792; 0,911] | 0,698 [0,633; 0,772] | [−0,080; +0,076] | **[+0,024; +0,143]** | 489 / 68 / 247 |
| YOLO26s | 20,5 | 714 | 84/16/42 | 0,840 [0,782; 0,915] | 0,667 [0,560; 0,778] | [−0,097; +0,082] | **[+0,007; +0,104]** | 486 / 71 / 228 |
| YOLO26m | 21,9 | 749 | 86/8/40 | 0,915 [0,864; 0,965] | 0,683 [0,613; 0,764] | [−0,005; +0,131] | **[+0,016; +0,114]** | 482 / 75 / 267 |
| YOLO12s | 24,1 | 623 | 76/11/50 | 0,874 [0,802; 0,947] | 0,603 [0,485; 0,729] | [−0,045; +0,087] | [−0,068; +0,041] | 480 / 77 / 143 |
| YOLO12m | 25,6 | 639 | 83/7/43 | 0,922 [0,847; 0,987] | 0,659 [0,566; 0,769] | [−0,031; +0,158] | [0,000; +0,089] | 475 / 82 / 164 |
| YOLOv10s | 17,0 | 634 | 80/8/46 | 0,909 [0,862; 0,961] | 0,635 [0,557; 0,726] | **[+0,005; +0,108]** | [0,000; +0,045] | 486 / 71 / 148 |
| YOLOv10m | 21,0 | 672 | 87/5/39 | 0,946 [0,895; 0,989] | 0,690 [0,614; 0,783] | **[+0,038; +0,149]** | **[+0,048; +0,110]** | 483 / 74 / 189 |
| YOLOv8x | 32,0 | 765 | 88/15/38 | 0,854 [0,812; 0,899] | 0,698 [0,619; 0,787] | [−0,077; +0,085] | **[+0,031; +0,139]** | 484 / 73 / 281 |
| YOLO11x | 31,4 | 730 | 89/8/37 | 0,918 [0,879; 0,957] | 0,706 [0,623; 0,798] | [−0,006; +0,140] | **[+0,040; +0,150]** | 481 / 76 / 249 |
| YOLOv9t | 33,0 | 632 | 79/11/47 | 0,878 [0,815; 0,951] | 0,627 [0,538; 0,731] | [−0,031; +0,086] | [−0,025; +0,063] | 487 / 70 / 145 |
| YOLOv9s | 32,3 | 656 | 84/11/42 | 0,884 [0,849; 0,928] | 0,667 [0,591; 0,750] | [−0,023; +0,082] | **[+0,009; +0,093]** | 488 / 69 / 168 |
| YOLOv9c | 25,8 | 708 | 83/9/43 | 0,902 [0,830; 0,975] | 0,659 [0,583; 0,750] | [−0,040; +0,146] | **[+0,008; +0,084]** | 476 / 81 / 232 |

**Đọc kết quả (khám phá):**
- **Xu hướng chung:** model cỡ s trở lên thường có recall cao hơn YOLOv8n trên 14 ảnh này; recall cao nhất là YOLO11x (0,706). Phù hợp với giả thuyết model lớn bắt được nhiều xe nhỏ hơn, nhưng mới thấy trên 126 box.
- **YOLOv10m** có CI không chứa 0 ở cả precision lẫn recall. Đây là 1 trong 26 phép so sánh của lượt này (42 nếu tính cả lượt trước) trên cùng 14 ảnh, nên có thể là ngẫu nhiên. **Không** kết luận YOLOv10m tốt hơn.
- Model lớn sinh nhiều box hơn (YOLOv8x: 765 so với 557). Phần lớn box thêm nằm ở 86 ảnh **không có nhãn**, nên không biết đúng hay sai.
- **Thời gian:** không xếp hạng tốc độ YOLO.
  - Ở batch 1 trên RTX 2060, thời gian bị chi phí cố định chi phối: YOLOv9t (33,0 ms) chậm hơn YOLOv9c (25,8 ms).
  - Thời gian giữa các lượt dao động tới 19% (xem docs/12).

## Kết quả segmentation có prompt (khám phá; tham chiếu SAM-B)

Chỉ số chính trên **92 ảnh có prompt**, tính bằng `scripts/audit_sam_eval.py`; CI 95% bootstrap theo ảnh trong 92 ảnh, 2000 lần, seed 42.

| Model | s/ảnh | Tổng (92 ảnh) | Nạp* | IoU gộp [CI] | IoU TB/ảnh [CI] | ΔIoU gộp [CI] | ΔIoU TB/ảnh [CI] | Tốt hơn / kém hơn / ngang | IoU trong box | Δ thời gian |
|---|---:|---:|---:|---|---|---|---|---|---:|---|
| SAM-B | 0,308 | 28,3 s | 0,80 s | 0,775 [0,693; 0,844] | 0,765 [0,723; 0,803] | — | — | — | 0,894 | — |
| SAM2.1-T | 0,121 | 11,1 s | 0,36 s | 0,788 [0,705; 0,860] | 0,771 | **+0,013 [+0,007; +0,022]** | **+0,006 [+0,002; +0,010]** | 49 / 36 / 7 | 0,908 | −187 ms, nhanh hơn ở 92/92 ảnh |
| SAM2.1-S | 0,136 | 12,5 s | 0,42 s | 0,791 [0,710; 0,859] | 0,771 | **+0,016 [+0,001; +0,033]** | **+0,007 [+0,001; +0,012]** | 57 / 29 / 6 | 0,911 | −172 ms, nhanh hơn ở 92/92 |
| SAM2.1-L | 0,464 | 42,7 s | 1,81 s | 0,790 [0,711; 0,861] | 0,769 | +0,015 [−0,001; +0,034] | +0,004 [−0,002; +0,010] | 46 / 41 / 5 | 0,909 | +156 ms, chậm hơn ở 92/92 |
| SAM2-L | 0,467 | 43,0 s | 1,80 s | 0,791 [0,707; 0,860] | 0,769 | **+0,016 [+0,002; +0,034]** | +0,004 [−0,002; +0,010] | 45 / 43 / 4 | 0,911 | +159 ms, chậm hơn ở 92/92 |

\* Nạp model khi file đã nằm trong cache hệ điều hành.

Chỉ số đầu-cuối trên 98 ảnh (gồm 6 ảnh không prompt, IoU = 0 cho mọi SAM): SAM-B 0,771; SAM2.1-T 0,784; SAM2.1-S 0,787; SAM2.1-L 0,786; SAM2-L 0,787.
Chênh lệch và CI có trong `sam_eval_recomputed.csv`; xu hướng giống bảng trên.

**Đọc kết quả (khám phá):**
- **Tốc độ ổn định:** SAM 2.1 cỡ t/s nhanh hơn SAM-B khoảng 2,3–2,5 lần ở mọi ảnh. SAM 2 / 2.1 cỡ L **chậm hơn** SAM-B khoảng 1,5 lần ở mọi ảnh.
- **Chất lượng:** mọi biến thể SAM 2 / 2.1 có IoU gộp cao hơn SAM-B khoảng +0,013…+0,016. Khác biệt này nhỏ.
  - SAM2.1-T và SAM2.1-S có CI không chứa 0 ở cả hai chỉ số; SAM2-L chỉ ở IoU gộp.
  - Đây là 8 phép so sánh của lượt này (20 nếu tính cả lượt trước) trên cùng 92 ảnh một lớp, nên **không** kết luận model nào thắng.
- Các model SAM 2 cỡ L **không** cho thấy chất lượng cao hơn cỡ t/s trên pilot này, trong khi chậm hơn khoảng 3,5 lần.
- Mọi SAM dùng cùng prompt YOLOv8n; kết quả không nói gì về SAM dùng prompt khác.

## Đề xuất

- **Giữ mặc định YOLOv8n + SAM-B.** Chưa đổi model, chưa sửa app hay slide.
- Nếu muốn kiểm chứng tiếp, nên **đăng ký trước** một số rất ít ứng viên (vd. 1 YOLO cỡ m/l và SAM2-T hoặc SAM2.1-T/S),
  rồi đánh giá trên **ảnh mới** có nhãn đầy đủ (gợi ý ≥ 100 ảnh có nhãn box), thay vì chọn trên chính 14 / 92 ảnh này.

## Giới hạn

- 1 lớp (car); nhãn detection chỉ có 14 ảnh; IoU semantic theo ảnh, không theo đối tượng.
- Nhiều phép so sánh, chưa hiệu chỉnh; các khoảng tin cậy không chứa 0 chỉ là tín hiệu khám phá.
- Một lượt chạy cho mỗi model mới; chạy lần lượt (không xen kẽ); thời gian YOLO dao động giữa các lượt.
- YOLOv10 / YOLO26 chạy qua NMS, không ở chế độ không-NMS gốc.
- Thời gian nạp model đo khi đã có cache.

## File kết quả (`runs/20261018-231455/`)

`model_inventory.csv`, `per_image_results.csv`, `detections_<model>.csv`, `detection_summary.json`, `segmentation_summary.json`,
`model_summary.csv` (có cột `metric_eval_set`), `sam_eval_image_sets.csv`, `sam_eval_recomputed.csv`, `sam_eval_audit.json`, `run_config.json`.

> Trong repository công khai chỉ có 4 file tổng hợp ở trên (`run_config.json`, `detection_summary.json`, `segmentation_summary.json`, `sam_eval_recomputed.csv`); các file còn lại trong bảng được giữ local.

## Chạy lại

```powershell
$W = "<thư mục weights>"
.\.venv_models\Scripts\python.exe scripts\compare_model_zoo.py --archive <archive.zip BDD100K> `
  --yolo $W\yolov8n.pt $W\yolov8l.pt $W\yolo11l.pt $W\yolo26s.pt $W\yolo26m.pt $W\yolo12s.pt $W\yolo12m.pt `
         $W\yolov10s.pt $W\yolov10m.pt $W\yolov8x.pt $W\yolo11x.pt $W\yolov9t.pt $W\yolov9s.pt $W\yolov9c.pt `
  --sam  $W\sam_b.pt $W\sam2.1_t.pt $W\sam2.1_s.pt $W\sam2.1_l.pt $W\sam2_l.pt
.\.venv\Scripts\python.exe scripts\audit_sam_eval.py --run data\real\model_comparison\runs\<run_id>
```
