# Chạy trên Google Colab

Local dùng cho EDA, phân tích, test. Training chạy trên Colab (GPU free) để tiết kiệm
thời gian. Code giống nhau; chỉ khác đường dẫn, cấu hình qua biến môi trường.

## Sơ đồ

```text
LOCAL                                    COLAB
D:\workspace\waste_classifier            /content/waste_classifier     (clone repo)
├── data/raw/                            /content/data/raw/            (giải nén bundle)
├── data/splits/  (commit git)           data/splits/                  (từ git)
└── experiments/  (copy về)              /content/drive/.../experiments (ghi trực tiếp)

GOOGLE DRIVE: MyDrive/waste_classifier/
├── data/waste_raw_v1.zip
└── experiments/rqX/<run_id>/
```

## 1. Tạo bundle dữ liệu (local, một lần)

```bash
uv run python scripts/make_data_bundle.py
```

Kết quả:

```text
dist/waste_raw_v1.zip
dist/waste_raw_v1.zip.sha256
```

Upload cả hai file lên `MyDrive/waste_classifier/data/`.

Bundle giữ tiền tố `raw/`, nên giải nén vào `/content/data` sẽ tạo
`/content/data/raw/realwaste/...` và `/content/data/raw/trashnet/...`.

## 2. Chuẩn bị Colab

Mở `notebooks/colab_setup.ipynb`, chọn Runtime → GPU (T4), chạy lần lượt:

1. Mount Drive.
2. Clone/pull repo.
3. Cài `uv` + `requirements-colab.txt`.
4. Giải nén bundle vào `/content/data`.
5. Kiểm tra GPU và số lượng ảnh.

Notebook đặt các biến môi trường:

```python
WASTE_DATA_ROOT=/content/data
WASTE_EXPERIMENTS_ROOT=/content/drive/MyDrive/waste_classifier/experiments
```

Mọi entrypoint đọc các biến này qua `src/utils/paths.py`, không cần sửa code.

## 3. Chạy training

```bash
python -m research.rqX.train --config configs/rqX/<config>.yaml
```

Kết quả ghi trực tiếp vào Drive:

```text
/content/drive/MyDrive/waste_classifier/experiments/rqX/<exp_id>_s<seed>/
├── config.yaml
├── environment.json
├── metrics.json
├── checkpoints/
└── logs/
```

## 4. Lấy kết quả về local

Copy thư mục `experiments/rqX/<run_id>` từ Drive về `experiments/rqX/` trong repo
rồi chạy notebook phân tích. Không commit `experiments/`.

## 5. Lưu ý

- **Torch:** Colab dùng torch/torchvision có sẵn (CUDA phù hợp GPU). Không chạy
  `uv sync` trên Colab vì torch 2.15.0 Linux kéo CUDA 13.2, có thể không tương thích T4.
- **Version:** `environment.json` ghi torch/CUDA/GPU thực tế của mỗi run. Báo cáo phải
  nêu rõ training chạy trên Colab, không phải môi trường local.
- **Session:** Colab free giới hạn ~12 giờ và tự ngắt khi idle. Checkpoint mỗi epoch và
  dùng `--resume` khi cần chạy tiếp.
- **Drive I/O:** chỉ đọc một file bundle lớn từ Drive; không đọc trực tiếp hàng nghìn
  ảnh nhỏ từ Drive.
- **Split:** `data/splits/**.csv` được commit vào git để local và Colab dùng đúng một
  split, tránh lệch kết quả.
- **Dữ liệu:** không commit dataset; Drive để chế độ private.
