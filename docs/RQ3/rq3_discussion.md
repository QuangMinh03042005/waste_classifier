# RQ3 – Thảo luận và kết luận

Tài liệu này là phần **diễn giải** (viết tay, không sinh tự động) của RQ3. Số liệu gốc lấy
từ [rq3_results.md](rq3_results.md) (bản 50 epoch, 5 seed cho 3 protocol chính — kết quả
chính); bản 30 epoch để so sánh ngân sách ở [rq3_results_e30.md](rq3_results_e30.md).

**Câu hỏi:** Kết hợp dữ liệu từ nhiều nguồn có cải thiện khả năng tổng quát hóa của mô
hình CNN trên các nguồn dữ liệu khác nhau không?

## 1. Tóm tắt

Trên 27 run (3 protocol cross-dataset/pooled × 5 seed + 4 mức mix-ratio × 3 seed):

- Mô hình huấn luyện một nguồn **sụp mạnh** khi kiểm thử nguồn còn lại: Macro-F1
  0,2971 ± 0,0432 (RealWaste → TrashNet) và 0,4694 ± 0,0154 (TrashNet → RealWaste),
  trong khi validation in-domain vẫn ~0,90.
- Khi cả hai nguồn có mặt trong train (pooled), Macro-F1 giữ được **0,8992 / 0,8869**
  trên held-out của từng nguồn (0,8945 trên toàn bộ 823 ảnh test).
- Delta ghép cặp theo seed (cùng seed, cùng held-out test) là **+0,4320 ± 0,0283**
  (RealWaste) và **+0,5679 ± 0,0383** (TrashNet) — cả 5 seed đều dương và lớn hơn nhiều
  so với nhiễu giữa seed.
- Không cần 100% dữ liệu nguồn thứ hai: thêm **25%** đã đưa Macro-F1 TrashNet từ
  0,3826 lên **0,7860** (+0,40), mức 50% đạt 0,8413, 100% đạt 0,8869; RealWaste gần
  như không đổi (0,8697 → 0,9006 ở mức 25%, sau đó ~0,89–0,90).

**Kết luận:** câu trả lời là **có**, nhưng cơ chế là *mô hình không còn phải chuyển
miền* chứ không phải *mô hình chuyển miền tốt hơn*. Khi train chỉ có một nguồn, mô hình
học đặc trưng bối cảnh của nguồn đó (nền, ánh sáng, cách chụp) và dự đoán sụp ở nguồn
kia; thêm dữ liệu nguồn thứ hai giúp mô hình học được cả hai bối cảnh.

## 2. Đối chiếu giả thuyết đã đăng ký

| Giả thuyết / tiêu chí (trong plan) | Kết quả | Kết luận |
|---|---|---|
| Pooled cao hơn single-source ≥ 0,02 trên **cả hai** nguồn | +0,4320 (RealWaste), +0,5679 (TrashNet), ghép cặp theo seed | **Ủng hộ** |
| Mức cải thiện lớn hơn ở TrashNet | +0,57 so với +0,43 | **Ủng hộ** |
| TrashNet F1 tăng ≥ 0,05 khi 0% → 25% | +0,4034 | **Đạt** |
| Ở mức 50%, TrashNet F1 ≥ 0,80 | 0,8413 | **Đạt** |
| RealWaste giảm ≤ 0,03 ở mức 50% | −0,0236 (tức tăng nhẹ) | **Đạt** |
| Phần lớn cải thiện đến sớm (25–50%) | 25% đã đạt ~89% mức pooled trên TrashNet | **Ủng hộ một phần**: vẫn tăng đều đến 100% |

Kế hoạch dự phòng (phân tích domain shift khi giả thuyết bị bác bỏ) không phải kích
hoạt, nhưng phần phân tích domain shift vẫn được thực hiện và trình bày dưới đây.

## 3. Cơ chế domain shift

Bằng chứng định lượng từ cùng predictions (không đánh giá lại test):

| Hiện tượng | RealWaste → TrashNet | TrashNet → RealWaste |
|---|---|---|
| Lớp bị dự đoán nhiều nhất | cardboard **60,4%** | paper **32,9%** |
| F1 thấp nhất | glass 0,219 | cardboard 0,298 |
| F1 cao nhất | metal 0,396 | metal 0,613 |
| Cặp nhầm lớn nhất | paper → cardboard (72,2% số ảnh paper) | cardboard → paper (55,1% số ảnh cardboard) |

- Sụp về một lớp (mode collapse) là dấu hiệu mô hình dựa vào **bối cảnh** thay vì hình
  dạng vật thể: ảnh TrashNet là vật thể trên nền trắng, còn RealWaste là ảnh chụp hiện
  trường với nhiều nền.
- Nhóm lớp có hình dạng/texture đặc trưng (metal, paper) chịu ít hơn; glass và
  plastic chịu nhiều nhất — phù hợp với việc hai lớp này phụ thuộc phản xạ/ánh sáng,
  yếu tố thay đổi mạnh giữa hai nguồn.
- Grad-CAM trên ảnh dự đoán sai (mục 4.6 của báo cáo, hình trong
  `rq3_analysis/gradcam_cross/`) cho thấy mô hình chú ý vào **vùng bóng/phản xạ** hoặc
  **viền nền trắng** thay vì bề mặt vật liệu — ví dụ ảnh cardboard bị đoán glass vì
  vùng băng dính bóng, hoặc khay cardboard trên nền trắng bị đoán paper vì viền nền.
- Năng lực mô hình không phải nguyên nhân: cùng kiến trúc 1,52 M tham số đạt ~0,89–0,90
  in-domain; vấn đề nằm ở phân phối dữ liệu, không ở capacity.

## 4. Đường cong mix-ratio

| TrashNet trong train | 0% | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|
| Macro-F1 TrashNet (held-out) | 0,3826 | 0,7860 | 0,8413 | 0,8614 | 0,8869 |
| Macro-F1 RealWaste (held-out) | 0,8697 | 0,9006 | 0,8932 | 0,8903 | 0,8992 |

- Lợi ích biên giảm dần: +0,40 (0→25%), +0,055 (25→50%), +0,020 (50→75%),
  +0,026 (75→100%). Từ 50% trở đi, khác biệt **nhỏ hơn CI 95%** của từng mức
  (mix50 ±0,0058; mix75 ±0,0065) nên **không** kết luận 50% khác 75%.
- RealWaste dao động 0,87–0,90 ở mọi mức; thêm dữ liệu nguồn khác **không** gây quên
  thảm họa (không có catastrophic forgetting rõ rệt). Mức 0% có phương sai lớn nhất
  (±0,0357 trên RealWaste, ±0,0518 trên TrashNet) vì đây là mô hình duy nhất chỉ thấy
  một nguồn.

## 5. Ngân sách epoch và số seed

**Ngân sách 30 vs 50 epoch.** Bản 50 epoch chạy lại toàn bộ để đồng bộ quy ước
(`plans/rq3/*_e50.yaml`):

| Chỉ số | Giá trị |
|---|---|
| Số run chạm mốc 50 epoch | **0/27** (early stopping luôn kích hoạt trước) |
| Epoch cuối cùng | 10–47, trung bình 21,5 |
| Epoch tốt nhất (theo val) | 3–40, trung bình 14,5 |
| Tổng thời gian GPU | 93,3 phút cho 27 run |

So với bản 30 epoch ([rq3_results_e30.md](rq3_results_e30.md)), khi so **trên cùng 3 seed
42/123/2024**, chênh lệch Macro-F1 lớn nhất trên mọi protocol/nguồn test là **0,012**
(mix 0% trên TrashNet); các giá trị chính lệch ≤0,009. Tất cả nằm trong CI 95% ⇒ **kết
luận không phụ thuộc ngân sách 30 hay 50 epoch**; early stopping mới là yếu tố quyết
định số epoch thực tế.

**Số seed.** 3 protocol chính được chạy 5 seed (42, 123, 2024, 7, 3407) để đồng bộ với
RQ1; mix-ratio giữ 3 seed. Điều đáng chú ý: thêm seed **không** làm CI hẹp đi một nửa
như kỳ vọng, vì độ lệch chuẩn ước lượng của P1 tăng từ 0,0270 lên 0,0432. Cụ thể:

| Arm | CI95 (3 seed) | CI95 (5 seed) |
|---|---:|---:|
| RealWaste → TrashNet | ±0,0671 | ±0,0537 |
| TrashNet → RealWaste | ±0,0230 | ±0,0191 |
| Pooled – RealWaste | ±0,0062 | ±0,0045 |
| Pooled – TrashNet | ±0,0244 | ±0,0100 |

Đây là lý do nên báo cáo cả std: với 3 seed, phương sai của arm cross-dataset bị **đánh
giá thấp**, nên CI 3 seed trông "hẹp" hơn thực tế ở một số arm.

## 6. Giới hạn (phải nêu trong báo cáo)

1. **Số seed không đều**: 5 seed cho 3 protocol chính, 3 seed cho mix-ratio. CI 95% của
   cross-dataset là ±0,054 (P1) và ±0,019 (P2) — vẫn rộng; mọi so sánh chênh lệch
   < 0,02 giữa các arm không nên coi là khác biệt thực.
2. **Val trộn hai nguồn** ở toàn bộ sweep mix. Với mức 0%, val TrashNet là ngoài miền
   nên epoch được chọn có thể chưa tối ưu cho RealWaste (RealWaste 0,8697 ± 0,0357,
   phương sai cao hơn hẳn pooled 0,8992 ± 0,0037). Kết luận chính không đổi nhưng đây là
   yếu tố cần nêu.
3. **Bản chất test khác nhau giữa arm**: pooled/mix test trên held-out (360/463 ảnh),
   single-source test trên toàn bộ nguồn còn lại (2.390/3.092). Đã chấm lại trên cùng
   held-out để so sánh công bằng, nhưng **train** vẫn khác (pooled đã thấy nguồn đích).
4. **Chỉ một cặp nguồn** (RealWaste ↔ TrashNet). Không suy rộng trực tiếp cho mọi loại
   domain shift; cần thêm nguồn thứ ba để khẳng định.
5. **Chưa có arm adaptation** (fine-tune trên một phần nguồn đích) nên chưa tách được
   "tổng quát hóa" và "thích nghi".
6. **Chưa hiệu chỉnh multiple comparisons** cho 27 run × nhiều chỉ số; các so sánh ngoài
   tiêu chí đăng ký chỉ là exploratory.
7. **Phi tất định trên GPU**: seed cố định, `cudnn.benchmark=False`, nhưng AMP và một số
   kernel CUDA không đảm bảo bit-exact; chênh lệch giữa seed phản ánh cả hai yếu tố.
8. **Test held-out nhỏ** (360/463 ảnh) nên per-class F1 có CI rộng; không nên đọc chênh
   lệch per-class nhỏ.

## 7. Liên hệ với RQ1

Pooled của RQ3 huấn luyện đúng trên split v1 (cùng dữ liệu với arm MobileNetV3_small
pretrained của RQ1). Kết quả tương ứng:

- RQ1 MobileNet pretrained (5 seed): 0,8901 ± 0,0114 trên test v1 (823 ảnh).
- RQ3 pooled (5 seed, 50 epoch): 0,8945 ± 0,0059 trên cùng test v1.

Hai con số nhất quán ⇒ pipeline RQ3 không lệch; RQ3 **không** phát hiện thêm "pooled tốt
hơn RQ1", mà dùng pooled làm mốc để đo mức sụp của mô hình đơn nguồn.

## 8. Hàm ý thực tiễn

- Không nên triển khai mô hình huấn luyện một nguồn cho dữ liệu ngoài miền: mất
  ~0,43–0,57 Macro-F1 và dự đoán dồn về một lớp — nguy hiểm nếu dùng để phân loại rác.
- Nếu có dữ liệu từ nguồn mới, chỉ cần **~25%** lượng ảnh để mô hình đạt phần lớn lợi
  ích; đây là con số hữu ích khi lập kế hoạch thu thập dữ liệu.
- Với demo Streamlit: nên hiển thị cảnh báo khi ảnh khác miền huấn luyện (ví dụ
  confidence thấp hoặc entropy cao) thay vì tin vào nhãn dự đoán.

## 9. Việc nên làm tiếp

| Ưu tiên | Việc | Chi phí |
|---|---|---|
| P1 | Thêm seed cho mix 0% (arm nhiễu nhất: ±0,0518 trên TrashNet) | ~10 phút GPU |
| P2 | Arm trần in-domain (TrashNet-only → TrashNet test) | ~10 phút GPU |
| P2 | Arm mix00 với val chỉ RealWaste để kiểm tra ảnh hưởng mixed val | ~10 phút GPU, cần plan mới |
| P2 | Arm fine-tune adaptation để tách generalization vs adaptation | ~15 phút GPU, cần plan mới |

## 10. Tái lập

- 27 run bản chính (50 epoch): `C:\waste_classifier_experiments_e50\rq3\`.
- 21 run bản 30 epoch (phụ lục so sánh): `C:\waste_classifier_experiments\rq3\`.
- Bundle đầy đủ bản chính: `exports/waste_experiments_20261128_130050.zip`
  (27 run, 168,4 MiB, sha256 `c6c9237d…9dd259`), chỉ gồm `checkpoints/best.pt`.
- Môi trường từng run: `environment.json` (torch 2.15.0+cu132, CUDA 13.2, RTX 5070,
  commit hash, dirty flag).
- Sinh lại số liệu/hình/tracking:

  ```bash
  uv run python scripts/analyze_rq3.py --arm-suffix _e50
  uv run python scripts/analyze_rq3.py --output-md docs/RQ3/rq3_results_e30.md \
    --output-csv docs/RQ3/rq3_metrics_e30.csv --figures-dir docs/RQ3/rq3_analysis_e30 \
    --title "Phụ lục: RQ3 với ngân sách 30 epoch"
  uv run python scripts/track_rq3.py
  ```

- Notebook: `notebooks/rq3/analyze_rq3.ipynb` + PDF trong `docs/notebooks/rq3/`.

## 11. Kết luận

Kết hợp hai nguồn dữ liệu cải thiện rõ rệt khả năng phân loại trên cả hai nguồn, và
phần lớn lợi ích đạt được chỉ với 25% dữ liệu của nguồn thứ hai. Tuy nhiên, kết quả này
nên được phát biểu là *"mô hình có dữ liệu của cả hai miền thì không còn phải chuyển
miền"*, chứ không phải *"mô hình tổng quát hóa tốt hơn khi thiếu dữ liệu đích"* — bằng
chứng cross-dataset cho thấy khả năng chuyển miền thuần túy vẫn rất yếu (Macro-F1
0,30–0,47). Ngân sách 50 epoch không thay đổi kết luận: 0/27 run chạm 50 epoch và mọi
chênh lệch so với bản 30 epoch (trên cùng seed) đều nằm trong CI.
