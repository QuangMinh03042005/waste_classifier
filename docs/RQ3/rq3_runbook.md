# RQ3 – Hướng dẫn tự chạy (chỉ phần RQ3)

Tài liệu này chỉ nói về **RQ3 – Generalization**. Không cần chạy RQ1/RQ2.

Mọi lệnh chạy trong **terminal của VS Code (PowerShell)**, đứng tại thư mục repo:

```powershell
cd C:\Users\duyle\OneDrive\Desktop\waste_classifier
```

Cách mở terminal: VS Code → **File → Open Folder** → chọn thư mục trên → **Terminal → New Terminal**
(`Ctrl` + `` ` ``). Prompt phải có dạng `PS C:\Users\duyle\OneDrive\Desktop\waste_classifier>`.
Không dùng `cmd` (cú pháp `$env:...` sẽ lỗi).

## 0. Bảng việc cần làm

| # | Việc | Lệnh | Thời gian | Ra ở đâu |
|---|---|---|---:|---|
| 1 | Kiểm tra môi trường | `uv sync` + check GPU | 1 phút | – |
| 2 | Kiểm tra dữ liệu + split RQ3 | lệnh ở mục 1.2 | 10 giây | – |
| 3 | Test code liên quan RQ3 | `uv run pytest tests/test_rq3_configs.py tests/test_rq3_mix_splits.py tests/test_rq3_split_integrity.py tests/test_analyze_rq3.py tests/test_track_rq3.py` | 10 giây | – |
| 4 | Chạy thử 2 batch | `--epochs 1 --limit-batches 2` | 20 giây | `…\rq3\smoke_rq3` |
| 5 | Chạy 1 run thật | `research.rq3.train` | 2–5 phút | `…\rq3\<experiment>_s42` |
| 6 | Batch cross-dataset + pooled (9 run) | `scripts/run_batch.py` | **~31 phút** | `…\rq3\` |
| 7 | Batch mix-ratio (12 run) | `scripts/run_batch.py` | **~40 phút** | `…\rq3\` |
| 8 | Phân tích + Excel | `analyze_rq3.py`, `track_rq3.py` | 1 phút | `docs/RQ3/` |
| 9 | Notebook + PDF (nếu cần) | `jupyter nbconvert` | 2 phút | `docs/notebooks/rq3/` |

Tổng nếu chạy đủ 21 run: **~72 phút GPU** (đã đo: 70,5 phút).

## 1. Chuẩn bị (một lần)

### 1.1. Môi trường + GPU

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Kỳ vọng: `2.15.0+cu132 13.2 True NVIDIA GeForce RTX 5070`.
Nếu ra `+cpu`/`False`: `uv sync --reinstall-package torch --reinstall-package torchvision`.

### 1.2. Dữ liệu và split RQ3

Dữ liệu ảnh nằm ở `C:\waste_classifier_data\raw` (do biến `WASTE_DATA_ROOT`), split RQ3
đã commit sẵn trong `data/splits/rq3/`.

Kiểm tra đường dẫn đang dùng:

```powershell
uv run python -c "from src.utils.paths import data_root, experiments_root; print(data_root()); print(experiments_root())"
```

Kiểm tra split khớp ảnh thật (kỳ vọng `5019 0`):

```powershell
uv run python -c "from src.data.split import read_split_csv; from src.utils.paths import data_root; r=data_root(); rows=read_split_csv('data/splits/rq3/rw_to_tn.csv'); print(len(rows), sum(not (r/x.path).is_file() for x in rows))"
```

Nếu chưa có ảnh: `uv run python -m src.data.download --dataset all`.

### 1.3. Test code liên quan RQ3

```powershell
uv run pytest tests/test_rq3_configs.py tests/test_rq3_mix_splits.py tests/test_rq3_split_integrity.py tests/test_analyze_rq3.py tests/test_track_rq3.py
```

Kỳ vọng: `53 passed` (kiểm tra config khớp plan, split không leakage, phân tích, Excel).

## 2. Chạy thử (20 giây, không dùng cho báo cáo)

```powershell
uv run python -m research.rq3.train --config configs/rq3/rq3_mix_tn25.yaml --plan plans/rq3/source_mix.yaml --seed 42 --run-id smoke_rq3 --epochs 1 --limit-batches 2
```

Kỳ vọng in ra `Run: ...\rq3\smoke_rq3`, `Best val Macro-F1: 0.0xxx`.
Xoá sau khi xem:

```powershell
Remove-Item -Recurse -Force "$env:WASTE_EXPERIMENTS_ROOT\rq3\smoke_rq3"
```

## 3. Chạy 1 run thật (để xem learning curve và artifact)

```powershell
uv run python -m research.rq3.train --config configs/rq3/rq3_train_pooled_test_each.yaml --plan plans/rq3/cross_dataset_generalization.yaml --seed 42
```

- Khoảng 5 phút. Kỳ vọng `Test Macro-F1` ≈ 0,89–0,91.
- Kết quả: `…\rq3\rq3_train_pooled_test_each_s42\` gồm `run.json`, `config.yaml`, `plan.yaml`,
  `environment.json`, `history.jsonl`, `metrics.json`, `checkpoints\best.pt`, `predictions\test.csv`.

## 4. Chạy đủ 21 run

### 4.1. Cross-dataset + pooled (9 run, ~31 phút)

```powershell
uv run python scripts/run_batch.py --configs configs/rq3/rq3_train_realwaste_test_trashnet.yaml configs/rq3/rq3_train_trashnet_test_realwaste.yaml configs/rq3/rq3_train_pooled_test_each.yaml --plan plans/rq3/cross_dataset_generalization.yaml --seeds 42 123 2024
```

### 4.2. Mix-ratio (12 run, ~40 phút)

```powershell
uv run python scripts/run_batch.py --configs configs/rq3/rq3_mix_tn00.yaml configs/rq3/rq3_mix_tn25.yaml configs/rq3/rq3_mix_tn50.yaml configs/rq3/rq3_mix_tn75.yaml --plan plans/rq3/source_mix.yaml --seeds 42 123 2024
```

> **Hai batch dùng hai plan khác nhau.** Config mix dùng `source_mix.yaml`; config
> cross/pooled dùng `cross_dataset_generalization.yaml`. Đổi lẫn sẽ bị
> `PlanMismatchError`.

Batch tự **skip** run đã xong và **resume** run dở. Nếu bị đứt, chạy lại đúng lệnh trên.

### 4.3. Muốn chạy lại từ đầu mà không đụng kết quả cũ

Runner không cho ghi đè (`RunExistsError`). Dùng root riêng cho bản tự kiểm tra:

```powershell
$env:WASTE_EXPERIMENTS_ROOT = "C:\waste_classifier_runs_selfcheck"
```

Sau đó chạy lại lệnh ở mục 4.1/4.2. Biến này chỉ có tác dụng trong terminal hiện tại.

## 5. Phân tích kết quả + bảng Excel

```powershell
uv run python scripts/analyze_rq3.py
uv run python scripts/track_rq3.py
```

| File sinh ra | Nội dung |
|---|---|
| `docs/RQ3/rq3_results.md` | bảng mean ± std, CI95, delta ghép cặp, đường cong mix-ratio, per-class F1, hình |
| `docs/RQ3/rq3_metrics.csv` | metric từng run |
| `docs/RQ3/rq3_analysis/test_macro_f1.png` | so sánh Macro-F1 theo nguồn test |
| `docs/RQ3/rq3_analysis/dose_response.png` | Macro-F1 theo tỉ lệ TrashNet (0→100%) |
| `docs/RQ3/rq3_analysis/confusion_matrices.png` | confusion matrix từng protocol × nguồn |
| `docs/RQ3/rq3_tracking.xlsx` | 4 sheet: Runs / Tổng hợp / Kế hoạch / Môi trường |

`track_rq3.py` phải in `Runs: 21 (completed 21)` (nếu bạn chạy đủ 3 seed).

## 6. Notebook + PDF (cần khi nộp)

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/rq3/analyze_rq3.ipynb
uv run jupyter nbconvert --to pdf notebooks/rq3/analyze_rq3.ipynb --output-dir docs/notebooks/rq3/
uv run jupyter nbconvert --clear-output --inplace notebooks/rq3/analyze_rq3.ipynb
```

PDF cần MiKTeX + Pandoc (xem `docs/gpu_local.md` §6). Notebook phải commit ở trạng thái
sạch output, kèm PDF.

## 7. Kiểm tra "chạy đúng"

| Kiểm tra | Kỳ vọng |
|---|---|
| `…\rq3\<run>\run.json` | `"status": "completed"` |
| `…\checkpoints\best.pt` | có, ~6 MB |
| `…\predictions\test.csv` | 2.390 dòng (P1) / 3.092 (P2) / 823 (pooled, mix) |
| `environment.json` | `torch 2.15.0+cu132`, GPU của bạn |
| Macro-F1 test (dao động ±0,01–0,03) | P1 0,27–0,32 · P2 0,45–0,50 · pooled 0,89–0,91 |
| TrashNet held-out theo mix | 0% ≈0,37 · 25% ≈0,79 · 50% ≈0,84 · 75% ≈0,86 · 100% ≈0,89 |

## 8. Lỗi thường gặp (riêng RQ3)

| Lỗi | Nguyên nhân | Xử lý |
|---|---|---|
| `PlanMismatchError: Experiment ... không có trong plan` | mix ↔ plan sai | mix → `source_mix.yaml`; còn lại → `cross_dataset_generalization.yaml` |
| `PlanMismatchError: Seed ... không có trong plan` | seed lạ | RQ3 chỉ 42 / 123 / 2024 |
| `RunExistsError` | run đã tồn tại | đổi `--run-id` hoặc đổi `WASTE_EXPERIMENTS_ROOT` |
| `FileNotFoundError: Thiếu N ảnh` | `WASTE_DATA_ROOT` sai | đặt lại biến, mở terminal mới |
| `torch ... +cpu` | môi trường cũ | `uv sync --reinstall-package torch --reinstall-package torchvision` |

## 9. Vì sao có từng arm (giải thích ngắn)

| Arm | Trả lời câu hỏi | Nếu kết quả ngược |
|---|---|---|
| P1 `rw_to_tn` | Train RealWaste → test TrashNet được bao nhiêu? | Domain shift nhẹ, RQ3 ít ý nghĩa |
| P2 `tn_to_rw` | Chiều ngược có đối xứng? | Kết luận phải nêu rõ phụ thuộc chiều |
| P3 `pooled` | Kết hợp 2 nguồn có cải thiện? | Bác bỏ giả thuyết, báo cáo trung thực |
| `mix_tn00` | Không có dữ liệu nguồn kia thì sao? | Thêm nguồn không cần thiết |
| `mix_tn25/50/75` | Cần bao nhiêu dữ liệu nguồn thứ hai? | Càng nhiều càng tốt (tuyến tính) |

Nguyên tắc: plan `active` và đã commit **trước** khi chạy; test chỉ chấm **một lần** mỗi
run; không đổi tiêu chí sau khi xem test; không sửa/xoá run cũ.

## 10. Kết quả để trình bày

- `docs/RQ3/rq3_results.md` – số liệu chính.
- `docs/RQ3/rq3_discussion.md` – thảo luận, giới hạn, kết luận.
- `docs/RQ3/rq3_overview.md` – protocol, cách chạy, trạng thái.
- `docs/RQ3/rq3_tracking.xlsx` – bảng theo dõi 21 run.
- `docs/notebooks/rq3/analyze_rq3.pdf` – notebook đã chạy.
