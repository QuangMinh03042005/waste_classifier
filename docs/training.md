# Training

Entrypoint: `research/rq1/train.py`. Engine: `src/training/runner.py`.

## Chạy

```bash
python -m research.rq1.train \
  --config configs/rq1/selection/customcnn_cand_b.yaml \
  --plan plans/rq1/custom_cnn_selection.yaml \
  --seed 42
```

Smoke test (1 epoch, 1 batch):

```bash
python -m research.rq1.train \
  --config configs/rq1/selection/customcnn_cand_a.yaml \
  --plan plans/rq1/custom_cnn_selection.yaml \
  --seed 42 --epochs 1 --limit-batches 1
```

- `--epochs`, `--limit-batches`: chỉ dùng smoke/debug, không dùng cho kết quả báo cáo.
- `--plan` bắt buộc; thiếu plan sẽ bị từ chối (trừ `--allow-unplanned` cho debug).
- `--resume`: tiếp tục từ `checkpoints/last.pt` (chưa hỗ trợ khi `freeze_epochs > 0`).

## Config

```yaml
experiment: rq1_customcnn_cand_b
rq: rq1
seed: 42

data:
  root: null              # null = WASTE_DATA_ROOT hoặc data/
  splits_version: v1
  image_size: 224
  batch_size: 64
  num_workers: 2
  augmentation: false
  class_weighting: false

model:
  name: custom_cnn
  num_classes: 5
  channels: [32, 64, 128, 256]
  convs_per_stage: [1, 2, 2, 2]
  dropout: 0.3

training:
  epochs: 12
  optimizer: adamw
  lr: 0.001
  weight_decay: 0.0001
  scheduler: cosine
  early_stopping_patience: 4
  monitor: val_macro_f1
  amp: true
  evaluate_test: false
```

- `evaluate_test: false` → không tạo test dataset/loader, không ghi predictions.
- `class_weighting: true` → `w_c = N/(K·n_c)`, clip [0.5, 3.0].
- `freeze_epochs`, `unfreeze_blocks` cho MobileNet.
- `amp` chỉ bật khi có CUDA.
- Đường dẫn split: `data/splits/<splits_version>/split.csv` (hoặc `data.split_csv`).

## Output

```text
experiments/rq1/<experiment>_s<seed>/
├── run.json
├── plan.yaml
├── config.yaml
├── environment.json
├── history.jsonl
├── metrics.json
├── checkpoints/{last,best}.pt
└── predictions/test.csv     # chỉ khi evaluate_test=true
```

## Quy tắc

- Chọn epoch theo `val_macro_f1`; test chỉ đánh giá một lần sau khi train xong.
- `best.pt` theo val Macro-F1; `last.pt` mỗi epoch để resume.
- Không chạy test trong bước chọn cấu hình.
- Mọi run phải có plan active và snapshot vào run dir.
- Không sửa kết quả cũ; chạy lại tạo run mới.

## Tổng hợp

```bash
python scripts/collect_results.py --rq rq1 --markdown
python scripts/selection_report.py
```

## Debug và selection

Custom CNN:

```bash
python scripts/selection_report.py \
  --config-dir configs/rq1/debug \
  --prefix rq1_customcnn_ \
  --experiments rq1_customcnn_d0 rq1_customcnn_d1 rq1_customcnn_d2 \
    rq1_customcnn_d3 rq1_customcnn_d4 rq1_customcnn_d5 rq1_customcnn_r1 \
  --title "Debug Custom CNN" \
  --output docs/customcnn_debug.md
```

MobileNet:

```bash
python scripts/selection_report.py \
  --config-dir configs/rq1/selection/mobilenet \
  --prefix rq1_mobilenet_ \
  --title "Chọn cấu hình MobileNetV3_small" \
  --id-prefix rq1_mobilenet_ \
  --output docs/mobilenet_selection.md
```

Cả hai chỉ dùng validation (`evaluate_test: false`).

## Chạy batch

Thay vì tự viết loop:

```bash
python scripts/run_batch.py \
  --configs configs/rq1/debug/customcnn_d0.yaml configs/rq1/debug/customcnn_d1_50ep.yaml \
  --plan plans/rq1/custom_cnn_debug.yaml \
  --seeds 42
```

Batch runner tự:

- Skip run đã `completed`.
- Resume run dở nếu có `last.pt`.
- Báo lỗi nếu run dở không có checkpoint.
- Tùy chọn `--clean-incomplete`, `--continue-on-error`, `--export`.

## Đánh giá checkpoint ngoài miền

```bash
python -m research.evaluate \
  --config configs/rq1/rq1_mobilenetv3s_full.yaml \
  --checkpoint experiments/rq1/<run>/checkpoints/best.pt \
  --split-csv data/splits/rq3/pooled.csv \
  --split test --datasets realwaste \
  --output-dir experiments/rq1/<run>/evaluations/realwaste
```

Dùng cho RQ3 P3: train pooled một lần, evaluate riêng RealWaste và TrashNet.

## Phân tích RQ1

```bash
python scripts/analyze_rq1.py
```

Sinh `docs/rq1_results.md` + hình trong `docs/rq1_analysis/`.
