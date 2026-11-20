# Chạy trên GPU local (Windows)

Dùng khi train/evaluate ngay trên máy có GPU NVIDIA thay vì Colab. Code, config và
artifact giống hệt Colab; chỉ khác môi trường cài đặt.

## 1. Vì sao cần cấu hình riêng

Wheel `torch` trên PyPI cho Windows là bản **CPU-only** (`2.15.0+cpu`), nên
`torch.cuda.is_available()` trả về `False` dù máy có GPU. Vì vậy `pyproject.toml`
khai báo thêm index CUDA của PyTorch và chỉ áp dụng cho Windows:

```toml
[[tool.uv.index]]
name = "pytorch-cu132"
url = "https://download.pytorch.org/whl/cu132"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu132", marker = "sys_platform == 'win32'" }]
torchvision = [{ index = "pytorch-cu132", marker = "sys_platform == 'win32'" }]
```

- Windows: lấy `torch 2.15.0+cu132`, `torchvision 0.29.2+cu132`.
- Linux/Colab: vẫn dùng PyPI như cũ (`requirements-colab.txt` không đổi).
- Chọn `cu132` vì GPU RTX 50xx (Blackwell, compute capability 12.0) cần build
  CUDA ≥ 12.8; `cu132` hỗ trợ đủ và tương thích driver hiện hành.

## 2. Cài đặt

Yêu cầu: driver NVIDIA mới (CUDA UMD ≥ 12.8), Python 3.12 (uv tự cài).

```bash
uv sync
```

Lần đầu tải khoảng 2 GiB (wheel CUDA). Không cần cài CUDA Toolkit riêng, wheel đã
kèm runtime.

Kiểm tra GPU:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Kỳ vọng (ví dụ trên RTX 5070):

```text
2.15.0+cu132 13.2 True NVIDIA GeForce RTX 5070
```

Nếu thấy `+cpu` hoặc `False`:

```bash
uv sync --reinstall-package torch --reinstall-package torchvision
```

## 3. Dữ liệu

```bash
uv run python -m src.data.download --dataset all
```

Kiểm tra số ảnh khớp `manifest`:

```bash
uv run python -m src.data.download --dataset trashnet
```

- Split đã commit sẵn: `data/splits/v1/` (RQ1/RQ2) và `data/splits/rq3/` (RQ3).
  Không tạo lại split.
- Kiểm tra nhanh đường dẫn trong split khớp dữ liệu thật:

  ```bash
  uv run python -c "from src.data.split import read_split_csv; from src.utils.paths import data_root; r=data_root(); rows=read_split_csv('data/splits/v1/split.csv'); print(len(rows), sum(not (r/x.path).is_file() for x in rows))"
  ```

  Kỳ vọng: `5482 0` (5.482 dòng, không thiếu ảnh nào).
- Nếu muốn để dữ liệu và kết quả ngoài repo (tránh OneDrive sync ~700 MB và
  checkpoint), đặt biến môi trường **user-level** rồi mở lại terminal:

  ```powershell
  [Environment]::SetEnvironmentVariable("WASTE_DATA_ROOT", "D:\waste_data", "User")
  [Environment]::SetEnvironmentVariable("WASTE_EXPERIMENTS_ROOT", "D:\waste_experiments", "User")
  ```

  - `WASTE_DATA_ROOT` phải là thư mục **chứa `raw/`** (ví dụ
    `D:\waste_data\raw\realwaste\...`).
  - `WASTE_SPLITS_DIR` chỉ đổi khi muốn tách split khỏi repo; mặc định giữ
    `data/splits/` để mọi máy dùng đúng một split đã commit.
  - Kiểm tra lại: `uv run python -c "from src.utils.paths import data_root, experiments_root; print(data_root(), experiments_root())"`.

## 4. Chạy training

Ví dụ với entrypoint hiện có (RQ1):

```bash
uv run python -m research.rq1.train \
  --config configs/rq1/rq1_mobilenetv3s_full.yaml \
  --plan plans/rq1/model_comparison.yaml \
  --seed 42
```

Smoke test (1 epoch, 1 batch; không dùng cho kết quả báo cáo):

```bash
uv run python -m research.rq1.train \
  --config configs/rq1/rq1_mobilenetv3s_full.yaml \
  --plan plans/rq1/model_comparison.yaml \
  --seed 42 --epochs 1 --limit-batches 1
```

RQ3 dùng cùng engine với split riêng (`data/splits/rq3/`):

```bash
uv run python -m research.rq3.train \
  --config configs/rq3/rq3_train_realwaste_test_trashnet.yaml \
  --plan plans/rq3/cross_dataset_generalization.yaml \
  --seed 42
```

Xem [RQ3/rq3_overview.md](RQ3/rq3_overview.md) cho cả 3 protocol và cách báo cáo
kết quả theo từng nguồn.

## 5. Khác biệt so với Colab

| Hạng mục | Local GPU | Colab |
|---|---|---|
| Cài đặt | `uv sync` (torch cu132) | torch/torchvision có sẵn |
| Dữ liệu | `data/raw/` hoặc `WASTE_DATA_ROOT` | giải nén bundle vào `/content/data` |
| Kết quả | `experiments/` (không commit) | ghi thẳng Drive, export zip |
| Thời gian | không giới hạn session | ~12 giờ, dễ ngắt khi idle |

- `environment.json` của mỗi run ghi torch/CUDA/GPU thực tế, nên báo cáo phải nêu
  rõ run nào chạy local và run nào chạy Colab.
- Determinism: `cudnn.benchmark=False`, seed cố định; GPU vẫn có thể lệch rất nhỏ
  giữa các lần chạy. Kết quả báo cáo lấy mean ± std trên ≥3 seed.
