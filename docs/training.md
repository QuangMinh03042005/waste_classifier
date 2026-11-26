# Training

Entrypoint: `research/rq1/train.py`, `research/rq2/train.py`. Engine: `src/training/runner.py`.

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

## RQ2 – Ablation augmentation và class weighting

Ba cấu hình, mỗi cấu hình chỉ khác baseline đúng một biến:

| Config | Augmentation | Class weighting |
|---|---|---|
| `configs/rq2/rq2_mobilenetv3s_full.yaml` | bật | bật |
| `configs/rq2/rq2_mobilenetv3s_noaug.yaml` | tắt | bật |
| `configs/rq2/rq2_mobilenetv3s_nocw.yaml` | bật | tắt |

Seed chính thức: 42, 123, 2024. Ngân sách chung: 50 epoch, batch 64, early stopping
patience 7 theo `val_macro_f1`.

Chạy một cấu hình:

```bash
python -m research.rq2.train \
  --config configs/rq2/rq2_mobilenetv3s_full.yaml \
  --plan plans/rq2/augmentation_ablation.yaml \
  --seed 42
```

Chạy batch 3 cấu hình × 3 seed (9 run), tự skip run đã xong và resume run dở:

```bash
python scripts/run_batch.py \
  --configs configs/rq2/rq2_mobilenetv3s_full.yaml \
            configs/rq2/rq2_mobilenetv3s_noaug.yaml \
            configs/rq2/rq2_mobilenetv3s_nocw.yaml \
  --plan plans/rq2/augmentation_ablation.yaml \
  --seeds 42 123 2024 \
  --continue-on-error --export
```

Repo local không chứa `data/raw/` (chỉ commit split CSV), nên chạy thật cần GPU và
dữ liệu trên Colab theo [docs/colab.md](colab.md); local chỉ dùng cho kiểm tra tĩnh,
pytest và phân tích kết quả đã tải về.

### Smoke test an toàn

`run_id` luôn là `<experiment>_s<seed>`, nên **phải** trỏ experiments sang thư mục
tạm. Nếu smoke test vào `experiments/` thật, run 1 epoch sẽ được đánh dấu `completed`
và batch thật sẽ `SKIP` chính run đó.

Dùng config debug `configs/rq2/debug/smoke_full.yaml` (1 epoch,
`evaluate_test: false`, cùng kiến trúc và cờ ablation với baseline) để không đánh
giá test và không tạo `predictions/test.csv`:

```bash
python scripts/run_batch.py \
  --configs configs/rq2/debug/smoke_full.yaml \
  --plan plans/rq2/augmentation_ablation.yaml \
  --seeds 42 --epochs 1 --limit-batches 1 \
  --experiments-root /tmp/rq2_smoke
```

Chạy lẻ (entrypoint không có `--experiments-root`, dùng biến môi trường):

```bash
WASTE_EXPERIMENTS_ROOT=/tmp/rq2_smoke python -m research.rq2.train \
  --config configs/rq2/debug/smoke_full.yaml \
  --plan plans/rq2/augmentation_ablation.yaml \
  --seed 42 --epochs 1 --limit-batches 1
```

Trên Windows, thay `/tmp/rq2_smoke` bằng thư mục tạm, ví dụ
`$env:TEMP\rq2_smoke` (đặt `$env:WASTE_EXPERIMENTS_ROOT` trước khi chạy). Xoá thư
mục smoke sau khi kiểm tra. Lần smoke đầu cần Internet để tải pretrained weights.

Kỳ vọng sau smoke test:

- `run.json` ghi đúng `data.augmentation` và `data.class_weighting` của cấu hình.
- Có `plan.yaml`, `config.yaml`, `environment.json`, `history.jsonl`, `metrics.json`
  trong thư mục smoke, không nằm trong `experiments/` thật.
- `metrics.json` có `test: null` (không đánh giá test, không ghi
  `predictions/test.csv`) vì config debug đặt `evaluate_test: false`.
- Đường dẫn dữ liệu lấy từ `WASTE_DATA_ROOT`; split lấy từ
  `data/splits/v1/split.csv` (hoặc `WASTE_SPLITS_DIR`).

### Resume và chống ghi đè

- Chạy lại đúng lệnh batch: run có `checkpoints/last.pt` được `RESUME`, run
  `completed` được `SKIP`.
- Chạy lẻ: thêm `--resume`.
- `last.pt` lưu model, optimizer, scheduler, scaler và RNG state nên resume liền
  mạch về mặt tối ưu hoá; vẫn có thể lệch nhẹ trên GPU do phi tất định.
- RQ2 dùng `freeze_epochs = 0`, nên resume được hỗ trợ (engine chưa hỗ trợ resume
  khi `freeze_epochs > 0`).
- Không chạy lại vào run đã `completed`: engine từ chối ghi đè (`RunExistsError`).
  Muốn đổi cấu hình thì tạo config/experiment mới, không sửa kết quả cũ.

Kiểm tra tĩnh cấu hình RQ2 (không cần dữ liệu, không train):

```bash
uv run pytest tests/test_rq2_configs.py tests/test_rq2_entrypoint.py
```

Tổng hợp kết quả:

```bash
python scripts/collect_results.py --rq rq2 --markdown
```

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
  --output docs/RQ1/customcnn_debug.md
```

MobileNet:

```bash
python scripts/selection_report.py \
  --config-dir configs/rq1/selection/mobilenet \
  --prefix rq1_mobilenet_ \
  --title "Chọn cấu hình MobileNetV3_small" \
  --id-prefix rq1_mobilenet_ \
  --output docs/RQ1/mobilenet_selection.md
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

Sinh `docs/RQ1/rq1_results.md` + hình trong `docs/RQ1/rq1_analysis/`.

## Đo efficiency

```bash
python scripts/measure_efficiency.py
```

Mặc định đo các run final `rq1_customcnn`, `rq1_mobilenetv3s_full`, `rq1_mobilenetv3s_scratch`; có thể truyền `--runs` cụ thể. Kết quả: params, GFLOPs, latency batch 1/32, throughput, VRAM.

## Grad-CAM

Trên **validation** (không dùng test trước final):

```bash
python -m research.evaluate \
  --config configs/rq1/rq1_customcnn.yaml \
  --checkpoint experiments/rq1/<run>/checkpoints/best.pt \
  --split val \
  --output-dir experiments/rq1/<run>/evaluations/val

python scripts/gradcam.py \
  --run experiments/rq1/<run> \
  --predictions experiments/rq1/<run>/evaluations/val/predictions.csv \
  --num 8
```

Ảnh lưu ở `experiments/rq1/<run>/figures/gradcam/`.
