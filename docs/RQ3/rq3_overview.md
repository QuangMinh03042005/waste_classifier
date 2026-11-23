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

## 1b. Mix-ratio (cần bao nhiêu dữ liệu nguồn thứ hai)

Giữ nguyên 100% train RealWaste, tăng dần tỉ lệ group TrashNet train:

| Experiment | TrashNet train | Train / Val / Test | Plan |
|---|---|---:|---|
| `rq3_mix_tn00` | 0% | 2.163 / 826 / 823 | `plans/rq3/source_mix.yaml` |
| `rq3_mix_tn25` | 25% (~418 ảnh) | 2.577 / 826 / 823 | `plans/rq3/source_mix.yaml` |
| `rq3_mix_tn50` | 50% (~835 ảnh) | 2.995 / 826 / 823 | `plans/rq3/source_mix.yaml` |
| `rq3_mix_tn75` | 75% (~1.252 ảnh) | 3.416 / 826 / 823 | `plans/rq3/source_mix.yaml` |
| `rq3_train_pooled_test_each` | 100% | 3.833 / 826 / 823 | plan cross-dataset |

- Val/test giữ nguyên ở mọi mức để so sánh được; chỉ train thay đổi (một biến).
- Split sinh bằng `uv run python scripts/make_rq3_mix_splits.py`, lưu
  `data/splits/rq3/mix_tn_*.csv` + `mix_summary.json` (không sửa split cũ).
- Val dùng cả hai nguồn: ở mức 0% val TrashNet là ngoài miền nên chọn epoch có thể
  chưa tối ưu cho RealWaste — cần nêu khi báo cáo.

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

# mix-ratio × 3 seed
uv run python scripts/run_batch.py \
  --configs configs/rq3/rq3_mix_tn00.yaml configs/rq3/rq3_mix_tn25.yaml \
            configs/rq3/rq3_mix_tn50.yaml configs/rq3/rq3_mix_tn75.yaml \
  --plan plans/rq3/source_mix.yaml \
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

## 5. Phân tích

```bash
uv run python scripts/analyze_rq3.py
```

Script chỉ đọc artifact (không đánh giá lại test) và sinh:

- `docs/RQ3/rq3_results.md`: bảng mean ± std theo protocol/nguồn test, so sánh pooled
  với single-source (kèm chấm công bằng trên cùng held-out test), đường cong mix-ratio,
  per-class F1, tỉ lệ sụp đổ dự đoán.
- `docs/RQ3/rq3_metrics.csv`: số liệu chi tiết từng run.
- `docs/RQ3/rq3_analysis/`: hình Macro-F1 theo nguồn, mix-ratio và confusion matrix.

Bảng theo dõi thí nghiệm (Excel) cho báo cáo:

```bash
uv run python scripts/track_rq3.py
```

Sinh `docs/RQ3/rq3_tracking.xlsx` từ artifact: sheet **Runs** (mỗi run một dòng),
**Tổng hợp** (mean ± std theo experiment), **Kế hoạch** (baseline/variant của các
plan RQ3), **Môi trường** (torch/CUDA/GPU/commit của từng run).

Notebook trình bày kết quả: `notebooks/rq3/analyze_rq3.ipynb` (bản PDF trong
`docs/notebooks/rq3/`). Notebook chỉ đọc artifact và gọi `scripts/analyze_rq3.py`,
không huấn luyện lại hay đánh giá lại test.

Diễn giải, đối chiếu giả thuyết, giới hạn và hàm ý thực tiễn:
[rq3_discussion.md](rq3_discussion.md).

## 6. Tiêu chí thành công

- Cross-dataset: pooled cao hơn single-source ≥ 0.02 Macro-F1 (trung bình 3 seed) trên
  **cả hai** nguồn; đã xác nhận thêm bằng cách chấm trên cùng held-out test.
- Mix-ratio (plan `source_mix`): TrashNet F1 tăng ≥ 0.05 khi 0% → 25%, đạt ≥ 0.80 ở
  mức 50%, RealWaste giảm ≤ 0.03.

Nếu không đạt, báo cáo trung thực và phân tích domain shift, phân phối lớp, chất lượng
ảnh (kế hoạch dự phòng đã ghi trong plan).

## 7. Trạng thái

- [x] Split RQ3, plan active, config + entrypoint + test.
- [x] Chạy P1/P2/P3 × 3 seed (9 run).
- [x] Chạy mix-ratio 0/25/50/75% × 3 seed (12 run) — **tổng 21 run**.
- [x] Phân tích cross-dataset, domain shift, mix-ratio, confusion matrix theo nguồn.
- [x] Tracking Excel + notebook kèm PDF.
- [x] Thảo luận, giới hạn, kết luận và hướng dẫn tái lập ([rq3_discussion.md](rq3_discussion.md)).
- [ ] Viết mục RQ3 trong báo cáo cuối kỳ.
