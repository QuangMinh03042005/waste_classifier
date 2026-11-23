# RQ3 – Thảo luận và kết luận

Tài liệu này là phần **diễn giải** (viết tay, không sinh tự động) của RQ3. Số liệu gốc
lấy từ [rq3_results.md](rq3_results.md); quy trình ở [rq3_overview.md](rq3_overview.md).

**Câu hỏi:** Kết hợp dữ liệu từ nhiều nguồn có cải thiện khả năng tổng quát hóa của mô
hình CNN trên các nguồn dữ liệu khác nhau không?

## 1. Tóm tắt

Trên 21 run (3 protocol cross-dataset/pooled + 4 mức mix-ratio, mỗi cấu hình 3 seed):

- Mô hình huấn luyện một nguồn **sụp mạnh** khi kiểm thử nguồn còn lại: Macro-F1
  0,2894 ± 0,0261 (RealWaste → TrashNet) và 0,4788 ± 0,0265 (TrashNet → RealWaste),
  trong khi validation in-domain vẫn ~0,90.
- Khi cả hai nguồn có mặt trong train (pooled), Macro-F1 giữ được **0,9057 / 0,8889**
  trên held-out của từng nguồn.
- Delta ghép cặp theo seed (cùng seed, cùng held-out test) là **+0,4307 ± 0,0270**
  (RealWaste) và **+0,5795 ± 0,0210** (TrashNet) — mọi seed đều dương, lớn hơn nhiều
  so với nhiễu giữa seed.
- Không cần 100% dữ liệu nguồn thứ hai: thêm **25%** đã đưa Macro-F1 TrashNet từ
  0,3706 lên **0,7908** (+0,42), mức 50% đạt 0,8389, 100% đạt 0,8889; RealWaste gần
  như không đổi.

**Kết luận:** câu trả lời là **có**, nhưng cơ chế là *mô hình không còn phải chuyển
miền* chứ không phải *mô hình chuyển miền tốt hơn*. Cụ thể: khi train chỉ có một nguồn,
mô hình học đặc trưng bối cảnh của nguồn đó (nền, ánh sáng, cách chụp) và dự đoán sụp
ở nguồn kia; thêm dữ liệu nguồn thứ hai giúp mô hình học được cả hai bối cảnh.

## 2. Đối chiếu giả thuyết đã đăng ký

| Giả thuyết / tiêu chí (trong plan) | Kết quả | Kết luận |
|---|---|---|
| Pooled cao hơn single-source ≥ 0,02 trên **cả hai** nguồn | +0,43 (RealWaste) và +0,58 (TrashNet), ghép cặp theo seed | **Ủng hộ** |
| Mức cải thiện lớn hơn ở TrashNet | +0,58 so với +0,43 | **Ủng hộ** |
| TrashNet F1 tăng ≥ 0,05 khi 0% → 25% | +0,4202 | **Đạt** |
| Ở mức 50%, TrashNet F1 ≥ 0,80 | 0,8389 | **Đạt** |
| RealWaste giảm ≤ 0,03 ở mức 50% | −0,0104 (tức tăng nhẹ) | **Đạt** |
| Phần lớn cải thiện đến sớm (25–50%) | 25% đã đạt 89% mức pooled trên TrashNet | **Ủng hộ một phần**: vẫn còn tăng đều đến 100% |

Kế hoạch dự phòng (phân tích domain shift khi giả thuyết bị bác bỏ) không phải kích
hoạt, nhưng phần phân tích domain shift vẫn được thực hiện và trình bày dưới đây.

## 3. Cơ chế domain shift

Bằng chứng định lượng từ cùng predictions (không đánh giá lại test):

| Hiện tượng | RealWaste → TrashNet | TrashNet → RealWaste |
|---|---|---|
| Lớp bị dự đoán nhiều nhất | cardboard **58,2%** | plastic **31,1%** |
| F1 thấp nhất | glass 0,188 | glass 0,323 |
| F1 cao nhất | metal 0,385 | metal 0,599 |

- Sụp về một lớp (mode collapse) là dấu hiệu mô hình dựa vào **bối cảnh** thay vì hình
  dạng vật thể: ảnh TrashNet là vật thể trên nền trắng, còn RealWaste là ảnh chụp hiện
  trường với nhiều nền.
- Nhóm lớp có hình dạng/texture đặc trưng (metal, paper) chịu ít hơn; glass và
  plastic chịu nhiều nhất — phù hợp với việc hai lớp này phụ thuộc phản xạ/ánh sáng,
  yếu tố thay đổi mạnh giữa hai nguồn.
- Năng lực mô hình không phải nguyên nhân: cùng kiến trúc 1,52 M tham số đạt ~0,90
  in-domain; vấn đề nằm ở phân phối dữ liệu, không ở capacity.

## 4. Đường cong mix-ratio

| TrashNet trong train | 0% | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|
| Macro-F1 TrashNet (held-out) | 0,3706 | 0,7908 | 0,8389 | 0,8608 | 0,8889 |
| Macro-F1 RealWaste (held-out) | 0,8803 | 0,8980 | 0,8907 | 0,8922 | 0,9057 |

- Lợi ích biên giảm dần: +0,42 (0→25%), +0,048 (25→50%), +0,022 (50→75%),
  +0,028 (75→100%). Từ 50% trở đi, khác biệt **nhỏ hơn CI 95%** của từng mức
  (mix50 ±0,0045; mix75 ±0,0222) nên **không** kết luận 50% khác 75%.
- RealWaste dao động trong khoảng 0,88–0,91 ở mọi mức, tức thêm dữ liệu nguồn khác
  **không** gây quên thảm họa (không có catastrophic forgetting rõ rệt).

## 5. Giới hạn (phải nêu trong báo cáo)

1. **Chỉ 3 seed** theo quy ước RQ3. CI 95% của cross-dataset là ±0,065 — rộng; mọi
   so sánh chênh lệch < 0,02 giữa các arm không nên coi là khác biệt thực.
2. **Val trộn hai nguồn** ở toàn bộ sweep mix. Với mức 0%, val TrashNet là ngoài miền
   nên epoch được chọn có thể chưa tối ưu cho RealWaste (RealWaste 0,8803 ± 0,0418,
   phương sai cao hơn hẳn pooled 0,9057 ± 0,0042). Kết luận chính không đổi (delta lớn)
   nhưng đây là yếu tố cần nêu.
3. **Bản chất test khác nhau giữa arm**: pooled/mix test trên held-out (360/463 ảnh),
   single-source test trên toàn bộ nguồn còn lại (2.390/3.092). Đã chấm lại trên cùng
   held-out để so sánh công bằng, nhưng **train** vẫn khác (pooled đã thấy nguồn đích).
4. **Chỉ một cặp nguồn** (RealWaste ↔ TrashNet). Không suy rộng trực tiếp cho mọi loại
   domain shift; cần thêm nguồn thứ ba để khẳng định.
5. **Chưa có arm adaptation** (fine-tune trên một phần nguồn đích) nên chưa tách được
   "tổng quát hóa" và "thích nghi".
6. **Chưa hiệu chỉnh multiple comparisons** cho 21 run × nhiều chỉ số; các so sánh
   ngoài tiêu chí đăng ký chỉ là exploratory.
7. **Phi tất định trên GPU**: seed cố định, `cudnn.benchmark=False`, nhưng AMP và một số
   kernel CUDA không đảm bảo bit-exact; chênh lệch giữa seed phản ánh cả hai yếu tố.
8. **Test held-out nhỏ** (360/463 ảnh) nên per-class F1 có CI rộng; không nên đọc
   chênh lệch per-class nhỏ.

## 6. Hàm ý thực tiễn

- Không nên triển khai mô hình huấn luyện một nguồn cho dữ liệu ngoài miền: mất
  ~0,5–0,6 Macro-F1 và dự đoán dồn về một lớp — nguy hiểm nếu dùng để phân loại rác.
- Nếu có dữ liệu từ nguồn mới, chỉ cần **~25%** lượng ảnh để mô hình đạt phần lớn lợi
  ích; đây là con số hữu ích khi lập kế hoạch thu thập dữ liệu.
- Với demo Streamlit: nên hiển thị cảnh báo khi ảnh khác miền huấn luyện (ví dụ
  confidence thấp hoặc entropy cao) thay vì tin vào nhãn dự đoán.

## 7. Việc nên làm tiếp

| Ưu tiên | Việc | Chi phí |
|---|---|---|
| P1 | Grad-CAM trên ảnh sai của P1/P2 để xác nhận cơ chế "nhìn nền" | ~30 phút GPU |
| P1 | Thêm 2 seed cho 3 protocol chính để hẹp CI | ~20 phút GPU |
| P2 | Arm trần in-domain (TrashNet-only → TrashNet test) | ~10 phút GPU |
| P2 | Arm mix00 với val chỉ RealWaste để kiểm tra ảnh hưởng mixed val | ~10 phút GPU, cần plan mới |
| P2 | Arm fine-tune adaptation để tách generalization vs adaptation | ~15 phút GPU, cần plan mới |

## 8. Tái lập

- 21 run: `C:\waste_classifier_experiments\rq3\` (ngoài repo, không commit).
- Bundle đầy đủ: `exports/waste_experiments_20261123_145624.zip` (21 run, 125,1 MiB,
  sha256 `76b5db28…c25061e7`), chỉ gồm `checkpoints/best.pt`.
- Môi trường từng run: `environment.json` (torch 2.15.0+cu132, CUDA 13.2, RTX 5070,
  commit hash, dirty flag).
- Sinh lại số liệu/hình/tracking:

  ```bash
  uv run python scripts/make_rq3_splits.py --source-split data/splits/v1/split.csv --output-dir data/splits/rq3
  uv run python scripts/make_rq3_mix_splits.py
  uv run python scripts/analyze_rq3.py
  uv run python scripts/track_rq3.py
  ```

- Notebook: `notebooks/rq3/analyze_rq3.ipynb` + PDF trong `docs/notebooks/rq3/`.

## 9. Kết luận

Kết hợp hai nguồn dữ liệu cải thiện rõ rệt khả năng phân loại trên cả hai nguồn, và
phần lớn lợi ích đạt được chỉ với 25% dữ liệu của nguồn thứ hai. Tuy nhiên, kết quả
này nên được phát biểu là *"mô hình có dữ liệu của cả hai miền thì không còn phải
chuyển miền"*, chứ không phải *"mô hình tổng quát hóa tốt hơn khi thiếu dữ liệu đích"*
— bằng chứng cross-dataset cho thấy khả năng chuyển miền thuần túy vẫn rất yếu
(Macro-F1 0,29–0,48).
