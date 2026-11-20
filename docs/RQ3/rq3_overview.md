# RQ3 – Generalization

**Câu hỏi:** Việc kết hợp dữ liệu từ nhiều nguồn có cải thiện khả năng tổng quát hóa của
mô hình CNN trên các nguồn dữ liệu khác nhau không?

Plan đã đăng ký trước khi chạy: [plans/rq3/cross_dataset_generalization.yaml](../../plans/rq3/cross_dataset_generalization.yaml)
(`status: active`). Không sửa giả thuyết/tiêu chí sau khi xem test.

## 1. Ba protocol

| Protocol | Experiment | Train / Val | Test |
|---|---|---|---|
| P1 – RealWaste → TrashNet | `rq3_train_realwaste_test_trashnet` | RealWaste 2.163 / 466 | toàn bộ TrashNet 2.390 |
| P2 – TrashNet → RealWaste | `rq3_train_trashnet_test_realwaste` | TrashNet 1.670 / 360 | toàn bộ RealWaste 3.092 |
| P3 – Pooled | `rq3_train_pooled_test_each` | 3.833 / 826 (cả 2 nguồn) | 823 (463 RealWaste + 360 TrashNet) |

- Split sinh một lần bằng `uv run python scripts/make_rq3_splits.py`, lưu tại
  `data/splits/rq3/` (`rw_to_tn.csv`, `tn_to_rw.csv`, `pooled.csv`, `summary.json`).
- Không sửa split v1; nhóm ảnh chứa cả 2 nguồn bị loại khỏi split cross-dataset để
  chống leakage.
- Mọi config dùng `data.split_csv` (không dùng `splits_version`).

## 2. Cấu hình cố định

| Thành phần | Giá trị |
|---|---|
| Model | MobileNetV3_small pretrained ImageNet, dropout 0.2 |
| Input | 224×224, normalize mean/std ImageNet |
| Augmentation / Class weighting | không / không |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4, cosine |
| Batch / Epoch | 64 / tối đa 30, early stopping patience 7 |
| Chọn epoch | theo `val_macro_f1` |
| Seed | 42, 123, 2024 |

Mỗi biến thể chỉ đổi protocol dữ liệu; các yếu tố còn lại giữ nguyên.

## 3. Chạy

```bash
# một run
uv run python -m research.rq3.train \
  --config configs/rq3/rq3_train_realwaste_test_trashnet.yaml \
  --plan plans/rq3/cross_dataset_generalization.yaml \
  --seed 42

# cả 3 protocol × 3 seed
uv run python scripts/run_batch.py \
  --configs configs/rq3/rq3_train_realwaste_test_trashnet.yaml \
            configs/rq3/rq3_train_trashnet_test_realwaste.yaml \
            configs/rq3/rq3_train_pooled_test_each.yaml \
  --plan plans/rq3/cross_dataset_generalization.yaml \
  --seeds 42 123 2024
```

Smoke test (1 epoch, 1 batch; không dùng cho kết quả báo cáo) phải dùng `--run-id`
riêng, ví dụ `--run-id rq3_smoke`, và xoá run đó sau khi kiểm tra.

## 4. Kết quả mỗi run

`experiments/rq3/<experiment>_s<seed>/` gồm `run.json`, `config.yaml`, `plan.yaml`,
`environment.json`, `history.jsonl`, `metrics.json`, `checkpoints/{last,best}.pt`,
`predictions/test.csv` (có cột `dataset`).

Với P3, test set gồm cả 2 nguồn trong một lần đánh giá. Báo cáo riêng từng nguồn
bằng `predictions/test.csv`, **không** đánh giá lại test lần hai:

```bash
uv run python -m research.evaluate \
  --config configs/rq3/rq3_train_pooled_test_each.yaml \
  --checkpoint experiments/rq3/rq3_train_pooled_test_each_s42/checkpoints/best.pt \
  --split-csv data/splits/rq3/pooled.csv --split test --datasets trashnet \
  --output-dir experiments/rq3/rq3_train_pooled_test_each_s42/evaluations/trashnet
```

Lệnh trên chỉ dùng khi cần confusion matrix/prediction riêng cho một nguồn; nếu không
thì tách `predictions/test.csv` theo cột `dataset`.

## 5. Tiêu chí thành công

Pooled cao hơn single-source ≥ 0.02 Macro-F1 (trung bình 3 seed) trên **cả hai** test
set ngoài miền. Nếu không đạt, báo cáo trung thực và phân tích domain shift, phân phối
lớp, chất lượng ảnh (kế hoạch dự phòng đã ghi trong plan).

## 6. Trạng thái

- [x] Split RQ3, plan active, config + entrypoint + test.
- [ ] Chạy P1/P2/P3 × 3 seed.
- [ ] Phân tích cross-dataset, domain shift, confusion matrix theo nguồn.
- [ ] Báo cáo kết quả RQ3.
