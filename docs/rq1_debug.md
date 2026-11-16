# Debug Custom CNN (RQ1)

Mục đích: xác định Custom CNN yếu vì pipeline, optimization, budget, capacity hay
giới hạn kiến trúc — **trước khi đánh giá test**.

- Plan: `plans/rq1/custom_cnn_debug.yaml`
- Config: `configs/rq1/debug/`
- Báo cáo sinh tự động: `docs/customcnn_debug.md`

## Phases

### Phase 0 — Sanity (không train lâu)

- Overfit 1 batch 64 ảnh: train accuracy phải → ~1.0.
- Shuffle label: val accuracy ~0.2.
- Kiểm tra transform, normalization, label mapping.
- Kiểm tra không group nào nằm 2 split.

### Phase 1 — Optimization & budget

| ID | Thay đổi | Budget |
|---|---|---:|
| D0 | Baseline B | 30 |
| D1 | Baseline B | 50 |
| D2 | LR 3e-4 | 30 |
| D3 | LR 3e-3 | 30 |
| D4 | Batch 32 | 30 |
| D5 | Weight decay 1e-3 | 30 |
| R1 | Residual stage | 30 |

Tất cả dùng seed 42, `evaluate_test: false`.

### Phase 2 — Chẩn đoán

- Learning curves: train/val loss và Macro-F1.
- Overfit gap.
- Per-class F1, confusion matrix trên validation.
- Per-source: RealWaste vs TrashNet.
- Grad-CAM mẫu đúng/sai.

## Quy tắc quyết định

- Overfit 1 batch thất bại → dừng, sửa pipeline.
- D1 > D0 ≥ 0.01 → final dùng 50 epoch cho cả 3 arm.
- R1 > D0 ≥ 0.01 và ổn định → dùng residual làm baseline.
- D2/D3 tốt hơn rõ → cập nhật LR.
- Không dùng test trong bất kỳ bước nào.

## Sau debug

1. Chốt kiến trúc, LR, batch size, weight decay, budget.
2. Cập nhật `configs/rq1/rq1_customcnn.yaml`.
3. Cập nhật `plans/rq1/model_comparison.yaml`.
4. Đánh dấu plan debug `completed`.
5. Chạy final RQ1 3 arm × 5 seed.
