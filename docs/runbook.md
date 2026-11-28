# Runbook – tự chạy lại dự án

Tài liệu này hướng dẫn chạy **từng phần** của dự án từ máy local (Windows + GPU), kèm
thời gian thực tế đã đo và cách kiểm tra kết quả. Mọi lệnh chạy tại thư mục repo:

```powershell
cd C:\Users\duyle\OneDrive\Desktop\waste_classifier
```

Đường dẫn dữ liệu/kết quả trên máy hiện tại:

| Biến | Giá trị |
|---|---|
| Repo | `C:\Users\duyle\OneDrive\Desktop\waste_classifier` |
| `WASTE_DATA_ROOT` | `C:\waste_classifier_data` (chứa `raw/`) |
| `WASTE_EXPERIMENTS_ROOT` | `C:\waste_classifier_experiments` (kết quả, không commit) |

## Bắt đầu: chạy ở đâu?

**Dùng terminal của VS Code (PowerShell). Không dùng `cmd`** — nhiều lệnh trong tài liệu
dùng cú pháp PowerShell (`$env:...`, dấu `` ` `` để nối dòng) sẽ lỗi trong `cmd`.

### Mở đúng thư mục trong VS Code

1. Mở VS Code → menu **File → Open Folder…**
2. Dán `C:\Users\duyle\OneDrive\Desktop\waste_classifier` → **Select Folder**
3. Menu **Terminal → New Terminal** (hoặc `Ctrl` + `` ` ``)
4. Khung terminal hiện ở nửa dưới. Dòng cuối cùng là **prompt**, phải trông như:

   ```text
   PS C:\Users\duyle\OneDrive\Desktop\waste_classifier>
   ```

   Nếu prompt đang ở thư mục khác, gõ:

   ```powershell
   cd C:\Users\duyle\OneDrive\Desktop\waste_classifier
   ```

### Cách dán và chạy lệnh

- Copy lệnh trong tài liệu → click vào terminal → `Ctrl` + `V` (hoặc chuột phải → Paste).
- Nhấn `Enter` để chạy. Lệnh dài nhiều dòng (có dấu `` ` `` ở cuối): dán **cả khối**.
- Dừng giữa chừng: `Ctrl` + `C`.
- Muốn xem lại lệnh đã chạy: phím `↑`.

### Kiểm tra `uv` đã sẵn sàng

```powershell
uv --version
```

Kỳ vọng: `uv 0.13.9 ...`. Nếu báo *"uv is not recognized"*: đóng hẳn VS Code rồi mở lại
(để nhận PATH mới), hoặc dùng đường dẫn đầy đủ:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" --version
```

### Kết quả hiện ở đâu

| Loại | Vị trí | Xem bằng |
|---|---|---|
| Code, config, docs | trong repo | cây thư mục bên trái VS Code |
| Bảng kết quả, hình, PDF | `docs/RQ3/…` | click file trong VS Code |
| Run (checkpoint, metrics) | `C:\waste_classifier_experiments\rq3\…` | File Explorer (ngoài repo) |
| Excel tracking | `docs/RQ3/rq3_tracking.xlsx` | Excel |

> **Không bấm nút ▶ (Run) trên file `.py`.** Các script cần tham số dòng lệnh
> (`--config`, `--plan`, `--seed`), phải chạy bằng lệnh trong terminal như tài liệu.

## 0. Bảng tổng quan

> Chỉ làm RQ3? Xem bản gọn, tự chứa: [RQ3/rq3_runbook.md](RQ3/rq3_runbook.md).

| # | Việc | Lệnh chính | Thời gian | Kết quả |
|---|---|---|---:|---|
| 1 | Cài môi trường + kiểm tra GPU | `uv sync` | 1–3 phút (lần đầu ~5 phút tải torch) | `.venv`, torch cu132 |
| 2 | Test code | `uv run pytest` | 30 giây | 159 passed |
| 3 | Tải/kiểm tra dữ liệu | `src.data.download`, script kiểm tra | 0–20 phút | 5.482 ảnh khớp split |
| 4 | RQ1: 3 arm × 5 seed | `scripts/run_batch.py` | ~2–3 giờ (ước tính) | 15 run |
| 5 | RQ1 phụ: efficiency, Grad-CAM, tổng hợp | `scripts/*.py` | ~10 phút | bảng + hình |
| 6 | RQ3: 3 protocol × 5 seed (bản 50 epoch) | `scripts/run_batch.py` | **~50 phút** | 15 run |
| 7 | RQ3: mix-ratio × 3 seed (bản 50 epoch) | `scripts/run_batch.py` | **~45 phút** | 12 run |
| 8 | RQ3: phân tích + tracking | `scripts/analyze_rq3.py --arm-suffix _e50`, `scripts/track_rq3.py` | 1 phút | `rq3_results.md`, `rq3_tracking.xlsx` |
| 9 | Notebook + PDF | `jupyter nbconvert` | 2 phút | `.pdf` trong `docs/notebooks/rq3/` |

RQ2 **chưa có** `configs/rq2/` và `research/rq2/` nên hiện chưa chạy được (mới có plan).

## 1. Chuẩn bị (làm một lần)

### 1.1. Cài môi trường và kiểm tra GPU

```powershell
uv sync
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Kỳ vọng:

```text
2.15.0+cu132 13.2 True NVIDIA GeForce RTX 5070
```

Nếu ra `+cpu` hoặc `False`:

```powershell
uv sync --reinstall-package torch --reinstall-package torchvision
```

### 1.2. Biến môi trường (user-level, mở terminal mới sau khi đặt)

```powershell
[Environment]::SetEnvironmentVariable("WASTE_DATA_ROOT", "C:\waste_classifier_data", "User")
[Environment]::SetEnvironmentVariable("WASTE_EXPERIMENTS_ROOT", "C:\waste_classifier_experiments", "User")
```

Kiểm tra:

```powershell
uv run python -c "from src.utils.paths import data_root, experiments_root; print(data_root()); print(experiments_root())"
```

Nếu không đặt biến, mặc định là `data/` và `experiments/` **trong repo**.

### 1.3. Dữ liệu

Nếu chưa có ảnh:

```powershell
uv run python -m src.data.download --dataset all
```

Kiểm tra split khớp ảnh thật (kỳ vọng `5482 0`):

```powershell
uv run python -c "from src.data.split import read_split_csv; from src.utils.paths import data_root; root=data_root(); rows=read_split_csv('data/splits/v1/split.csv'); print(len(rows), sum(not (root/r.path).is_file() for r in rows))"
```

## 2. Test code (luôn chạy trước khi tin số liệu)

```powershell
uv run pytest
uv run ruff check .
```

Kỳ vọng: `159 passed`, `All checks passed!`. Test gồm: split/leakage, dataset, transforms,
model, logging/checkpoint, plan, phân tích RQ3, tracking Excel, toàn vẹn 7 file split RQ3.

## 3. RQ1 – kiến trúc (3 arm × 5 seed = 15 run)

```powershell
uv run python scripts/run_batch.py `
  --configs configs/rq1/rq1_customcnn.yaml `
            configs/rq1/rq1_mobilenetv3s_full.yaml `
            configs/rq1/rq1_mobilenetv3s_scratch.yaml `
  --plan plans/rq1/model_comparison.yaml `
  --seeds 42 123 2024 7 3407
```

- Mỗi run tạo `$env:WASTE_EXPERIMENTS_ROOT\rq1\<experiment>_s<seed>\`.
- Batch tự **skip** run đã `completed`, tự **resume** run dở (có `last.pt`).
- Chưa đo trên máy này (kết quả RQ1 trước đây chạy Colab); ước tính 2–3 giờ vì 50 epoch.
  Muốn biết trước, chạy 1 run: thêm `--epochs 1 --limit-batches 2` để smoke.

Tổng hợp RQ1:

```powershell
uv run python scripts/analyze_rq1.py
uv run python scripts/measure_efficiency.py
```

Grad-CAM (đánh giá val trước, rồi mới vẽ):

```powershell
uv run python -m research.evaluate `
  --config configs/rq1/rq1_mobilenetv3s_full.yaml `
  --checkpoint "$env:WASTE_EXPERIMENTS_ROOT\rq1\rq1_mobilenetv3s_full_s42\checkpoints\best.pt" `
  --split val --output-dir "$env:WASTE_EXPERIMENTS_ROOT\rq1\rq1_mobilenetv3s_full_s42\evaluations\val"

uv run python scripts/gradcam.py `
  --run "$env:WASTE_EXPERIMENTS_ROOT\rq1\rq1_mobilenetv3s_full_s42" `
  --predictions "$env:WASTE_EXPERIMENTS_ROOT\rq1\rq1_mobilenetv3s_full_s42\evaluations\val\predictions.csv" `
  --num 8
```

External GD (chỉ khi có ảnh GD):

```powershell
uv run python scripts/prepare_external_gd.py
uv run python scripts/evaluate_external.py
```

## 4. RQ3 – tổng quát hóa (21 run, bản chính 50 epoch)

> Bản chính dùng `configs/rq3/e50/` + `plans/rq3/*_e50.yaml`, ghi vào root riêng
> `C:\waste_classifier_experiments_e50` (~76 phút). Bản 30 epoch cũ vẫn giữ để so sánh
> ngân sách (xem `docs/RQ3/rq3_results_e30.md`).

### 4.1. Cross-dataset + pooled (9 run, ~30 phút)

```powershell
uv run python scripts/run_batch.py `
  --configs configs/rq3/e50/rq3_e50_train_realwaste_test_trashnet.yaml `
            configs/rq3/e50/rq3_e50_train_trashnet_test_realwaste.yaml `
            configs/rq3/e50/rq3_e50_train_pooled_test_each.yaml `
  --plan plans/rq3/cross_dataset_e50.yaml `
  --seeds 42 123 2024
```

### 4.2. Mix-ratio (12 run, ~45 phút)

```powershell
uv run python scripts/run_batch.py `
  --configs configs/rq3/e50/rq3_e50_mix_tn00.yaml configs/rq3/e50/rq3_e50_mix_tn25.yaml `
            configs/rq3/e50/rq3_e50_mix_tn50.yaml configs/rq3/e50/rq3_e50_mix_tn75.yaml `
  --plan plans/rq3/source_mix_e50.yaml `
  --seeds 42 123 2024
```

> Hai batch dùng **hai plan khác nhau**. Đưa config mix vào plan cross-dataset sẽ bị
> từ chối (`Experiment ... không có trong plan`).

### 4.3. Chạy một run lẻ (để demo)

```powershell
uv run python -m research.rq3.train `
  --config configs/rq3/e50/rq3_e50_train_realwaste_test_trashnet.yaml `
  --plan plans/rq3/cross_dataset_e50.yaml `
  --seed 42
```

Smoke test (1 epoch, 2 batch; **không** dùng cho kết quả báo cáo):

```powershell
uv run python -m research.rq3.train `
  --config configs/rq3/e50/rq3_e50_mix_tn25.yaml `
  --plan plans/rq3/source_mix_e50.yaml `
  --seed 42 --run-id smoke_demo --epochs 1 --limit-batches 2
```

Sau khi xem xong, xoá `$env:WASTE_EXPERIMENTS_ROOT\rq3\smoke_demo` để không lẫn vào báo cáo.

### 4.4. Nếu muốn chạy lại từ đầu mà không đụng kết quả cũ

Runner **không cho ghi đè** run đã tồn tại (`RunExistsError`). Dùng root riêng:

```powershell
$env:WASTE_EXPERIMENTS_ROOT = "C:\waste_classifier_runs_selfcheck"
uv run python scripts/run_batch.py --configs configs/rq3/rq3_mix_tn25.yaml `
  --plan plans/rq3/source_mix.yaml --seeds 42
```

## 5. Phân tích RQ3 + tracking Excel + notebook

```powershell
uv run python scripts/analyze_rq3.py --arm-suffix _e50
uv run python scripts/track_rq3.py
```

Kết quả:

| File | Nội dung |
|---|---|
| `docs/RQ3/rq3_results.md` | bảng mean ± std, CI95, delta ghép cặp, dose-response, per-class, hình |
| `docs/RQ3/rq3_metrics.csv` | metric từng run |
| `docs/RQ3/rq3_analysis/*.png` | Macro-F1 theo nguồn, dose-response, confusion matrix |
| `docs/RQ3/rq3_tracking.xlsx` | 4 sheet: Runs / Tổng hợp / Kế hoạch / Môi trường |

Notebook (cần MiKTeX + pandoc cho PDF, xem `docs/gpu_local.md` §6):

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/rq3/analyze_rq3.ipynb
uv run jupyter nbconvert --to pdf notebooks/rq3/analyze_rq3.ipynb --output-dir docs/notebooks/rq3/
uv run jupyter nbconvert --clear-output --inplace notebooks/rq3/analyze_rq3.ipynb
```

## 6. Kiểm tra "đã chạy đúng"

| Kiểm tra | Kỳ vọng |
|---|---|
| `...\rq3\<run>\run.json` | `"status": "completed"` |
| `...\checkpoints\best.pt` + `last.pt` | có, ~6 MB / ~18 MB |
| `...\predictions\test.csv` | có, đúng số ảnh test (2.390 / 3.092 / 823) |
| `environment.json` | torch `2.15.0+cu132`, `device_name` = GPU của bạn |
| `track_rq3.py` | `Runs: 21 (completed 21)` |
| Test Macro-F1 (khoảng, ± do GPU phi tất định) | P1 0,27–0,32 · P2 0,45–0,50 · pooled 0,89–0,91 |
| Mix TrashNet F1 | 0% ≈0,37 · 25% ≈0,79 · 50% ≈0,84 · 75% ≈0,86 · 100% ≈0,89 |

Sai lệch nhỏ giữa các lần chạy là bình thường (AMP + CUDA không bit-exact).

## 7. Lỗi thường gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `torch.cuda.is_available() == False`, version `+cpu` | `uv sync` cũ / wheel CPU | `uv sync --reinstall-package torch --reinstall-package torchvision` |
| `FileNotFoundError: Thiếu N ảnh ...` | `WASTE_DATA_ROOT` sai | Đặt lại biến, mở terminal mới |
| `PlanMismatchError: Experiment ... không có trong plan` | Dùng sai plan cho config | mix → `source_mix.yaml`; còn lại → `cross_dataset_generalization.yaml` |
| `PlanMismatchError: Seed ... không có trong plan` | Seed ngoài plan | Dùng 42/123/2024 (RQ3) hoặc 42/123/2024/7/3407 (RQ1) |
| `RunExistsError` | Run đã tồn tại | Đổi `--run-id` hoặc đổi `WASTE_EXPERIMENTS_ROOT` |
| `--help` lỗi `UnicodeEncodeError` | Console cp1252 | `$env:PYTHONUTF8 = "1"` |
| `PandocMissing` / `No counter 'none' defined` | Thiếu pandoc / bảng markdown | Xem `docs/gpu_local.md` §6 |
| Hết VRAM | App khác chiếm GPU | Đóng app, hoặc giảm `data.batch_size` (chỉ để thử, không dùng cho báo cáo) |

## 8. Vì sao cần từng test (giải thích)

| Test / arm | Câu hỏi nó trả lời | Vì sao phải có | Kết quả ngược lại thì sao |
|---|---|---|---|
| `pytest` (159 test) | Code/split/logging có đúng như mô tả? | Số liệu chỉ đáng tin khi split không leakage, metric viết đúng; test toàn vẹn split phát hiện ảnh trùng/group lọt split | Sửa code trước, **không** chạy thí nghiệm |
| RQ1 `rq1_customcnn` | Học từ đầu đạt mức nào? | Baseline không pretrained, để quy công cho transfer learning | Nếu ngang MobileNet pretrained → pretrained không giúp |
| RQ1 `rq1_mobilenetv3s_full` | Transfer learning từ ImageNet có tốt hơn? | Arm chính của RQ1 | Nếu kém baseline → giả thuyết transfer sai |
| RQ1 `rq1_mobilenetv3s_scratch` | Cải thiện đến từ kiến trúc hay trọng số? | Tách 2 yếu tố, tránh quy công sai | Nếu scratch ≈ pretrained → lợi ích đến từ kiến trúc |
| RQ3 P1 `rw_to_tn` | Train RealWaste, test TrashNet ra sao? | Đo transfer một chiều, baseline của plan RQ3 | Nếu P1 cao → domain shift nhẹ |
| RQ3 P2 `tn_to_rw` | Chiều ngược lại có đối xứng? | Domain shift thường **bất đối xứng**; thiếu P2 sẽ kết luận phiến diện | Nếu P2 ≫ P1 → kết luận phải nêu rõ phụ thuộc chiều |
| RQ3 P3 `pooled` | Kết hợp 2 nguồn có cải thiện? | Trả lời trực tiếp câu hỏi RQ3 | Nếu pooled ≈ single → bác bỏ giả thuyết, báo cáo trung thực |
| RQ3 `mix_tn00` | Nếu **không** có dữ liệu nguồn kia thì sao? | Đối chứng 0% với cùng val/test, để đo đúng giá trị của việc thêm dữ liệu | Nếu 0% đã cao → không cần thêm nguồn |
| RQ3 `mix_tn25/50/75` | Cần bao nhiêu dữ liệu nguồn thứ hai? | Dose-response: tìm ngưỡng "đủ", tránh khuyến nghị thu thập 100% một cách lãng phí | Nếu tuyến tính mãi → cần càng nhiều càng tốt |
| Phân tích held-out (không train) | So sánh có công bằng? | Single-source test trên toàn nguồn (2.390) còn pooled test trên held-out (360) — khác kích thước; chấm lại cùng held-out để loại bỏ confound | Nếu delta biến mất → kết luận cũ sai, phải sửa báo cáo |
| Delta ghép cặp theo seed | Cải thiện có phải do may mắn 1 seed? | n=3 nên phương sai lớn; ghép cặp khử biến thiên giữa seed | Nếu seed nào cũng âm → không có cải thiện |
| Tracking Excel | Số liệu có truy vết được? | Audit trail cho báo cáo: config, seed, commit, GPU, thời gian, metric từng run | Thiếu file này thì không chứng minh được run thật |

**Nguyên tắc học thuật khi chạy:** plan phải `active` và được commit **trước** khi chạy;
test chỉ đánh giá **một lần** mỗi run; không đổi tiêu chí sau khi xem test; không ghi
đè run cũ.

## 9. Thứ tự đề xuất cho buổi demo (~10 phút)

```powershell
uv run pytest                                   # 159 passed
uv run python -m research.rq3.train `
  --config configs/rq3/rq3_train_realwaste_test_trashnet.yaml `
  --plan plans/rq3/cross_dataset_generalization.yaml `
  --seed 42 --run-id demo_thay --epochs 2 --limit-batches 5
uv run python scripts/analyze_rq3.py --arm-suffix _e50
uv run python scripts/track_rq3.py
```

Sau đó mở `docs/RQ3/rq3_results.md`, `docs/RQ3/rq3_tracking.xlsx` và
`docs/notebooks/rq3/analyze_rq3.pdf` để trình bày. Nhớ xoá run `demo_thay` sau buổi demo.
