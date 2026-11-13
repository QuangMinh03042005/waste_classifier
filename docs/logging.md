# Log schema

Mỗi lần chạy là một thư mục riêng, không ghi đè. Log dùng JSON/JSONL để đọc bằng
máy và CSV để đưa vào báo cáo. Code: `src/utils/run_logger.py`,
`src/training/checkpoint.py`, `src/training/metrics.py`.

## Cấu trúc

```text
experiments/<rq>/<run_id>/
├── run.json                 # metadata, config, trạng thái, kết quả tóm tắt
├── config.yaml              # bản copy config
├── environment.json         # git, python, torch, CUDA, GPU, versions
├── history.jsonl            # 1 dòng / epoch / phase
├── metrics.json             # best validation + test (ghi cuối run)
├── error.json               # chỉ có khi run thất bại
├── checkpoints/
│   ├── last.pt              # đầy đủ state để resume
│   └── best.pt              # weights + metadata để inference
└── predictions/
    └── test.csv             # nhãn thật, nhãn dự đoán, xác suất
```

`run_id` theo quy ước `<exp_id>_s<seed>`, chỉ chứa chữ, số, `.`, `_`, `-`.

## `run.json`

```json
{
  "schema_version": "1.0",
  "run_id": "rq1_mobilenetv3s_aug_s42",
  "rq": "rq1",
  "experiment": "rq1_mobilenetv3s_aug",
  "seed": 42,
  "status": "running",
  "started_at": "2026-11-13T09:00:00+00:00",
  "finished_at": null,
  "duration_sec": null,
  "git": {"commit": "c55f619...", "branch": "feature/rq1-model-comparison", "dirty": false},
  "config": {"...": "bản copy đầy đủ"},
  "environment": "environment.json",
  "checkpoint": {
    "last": {"path": "checkpoints/last.pt", "epoch": 12, "metric": 0.80, "bytes": 20123456},
    "best": {"path": "checkpoints/best.pt", "epoch": 12, "metric": 0.80, "bytes": 7012345}
  },
  "resume": {"count": 0, "from_epoch": null}
}
```

Trạng thái: `running`, `completed`, `failed`.

## `history.jsonl`

Mỗi epoch ghi 2 dòng:

```json
{"epoch": 1, "phase": "train", "lr": 0.001, "loss": 1.234, "accuracy": 0.55, "macro_f1": 0.50, "duration_sec": 32.1, "gpu_mem_mb": 1024, "at": "..."}
{"epoch": 1, "phase": "val", "loss": 1.100, "accuracy": 0.60, "macro_f1": 0.57, "per_class_f1": {"cardboard": 0.51, "glass": 0.62, "metal": 0.66, "paper": 0.55, "plastic": 0.52}, "duration_sec": 4.3, "at": "..."}
```

Sự kiện khác (checkpoint, early stop) cũng ghi vào file này với trường `event`.

Đọc lại:

```python
from src.utils.run_logger import read_history
history = read_history(run_dir / "history.jsonl")
```

## `metrics.json`

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "class_names": ["cardboard", "glass", "metal", "paper", "plastic"],
  "best_val": {"epoch": 12, "loss": 0.61, "accuracy": 0.81, "balanced_accuracy": 0.79, "macro_f1": 0.80},
  "test": {
    "loss": 0.65,
    "accuracy": 0.79,
    "balanced_accuracy": 0.77,
    "precision_macro": 0.78,
    "recall_macro": 0.77,
    "macro_f1": 0.78,
    "per_class": {
      "cardboard": {"precision": 0.80, "recall": 0.75, "f1": 0.77, "support": 70}
    },
    "confusion_matrix": [[62, 3, 1, 2, 2]],
    "n_samples": 823
  },
  "efficiency": {
    "params": 1520000,
    "train_time_sec": 1800,
    "inference_ms_per_image": 6.4
  }
}
```

`class_names` bắt buộc: xác định thứ tự lớp cho checkpoint, confusion matrix và predictions.

## Checkpoint

| File | Nội dung | Khi ghi |
|---|---|---|
| `last.pt` | model + optimizer + scheduler + scaler + RNG + epoch | mỗi epoch (ghi đè) |
| `best.pt` | model weights + class_names + config + metric | khi val Macro-F1 cải thiện |

`last.pt` dùng để resume; `best.pt` dùng để inference/demo. Ghi atomic qua file `.tmp`.

## Predictions

`predictions/test.csv`:

```text
path,dataset,true_label,pred_label,correct,prob_cardboard,prob_glass,prob_metal,prob_paper,prob_plastic
```

`path` là đường dẫn tương đối so với data root, đọc được ở cả local và Colab.

## Tổng hợp kết quả

```bash
uv run python scripts/collect_results.py --markdown
```

Xuất:

- `experiments/results.csv` — 1 dòng / run
- `experiments/results_per_class.csv` — 1 dòng / run / lớp
- `experiments/results_summary.csv` — mean ± std theo experiment
- `experiments/results.md` — bảng Markdown cho báo cáo

Script chỉ đọc run, không sửa dữ liệu. Mặc định bỏ qua run chưa `completed`.
