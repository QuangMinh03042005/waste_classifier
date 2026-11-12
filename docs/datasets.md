# Dataset

## Nguồn dữ liệu

| Dataset | Số ảnh | Số lớp | Dung lượng | Giấy phép | Nguồn |
|---|---:|---:|---:|---|---|
| RealWaste | 4.752 | 9 | ~656 MB | CC BY 4.0 | [UCI 908](https://archive.ics.uci.edu/dataset/908/realwaste) |
| TrashNet | 2.527 | 6 | 42,8 MB | MIT | [garythung/trashnet](https://github.com/garythung/trashnet) |

Dataset **không được commit** lên Git. Chỉ commit script, manifest và tài liệu.

## Tải dữ liệu

```bash
uv sync
uv run python -m src.data.download --dataset all
```

Tải riêng từng bộ:

```bash
uv run python -m src.data.download --dataset realwaste
uv run python -m src.data.download --dataset trashnet --force
```

Tùy chọn:

| Flag | Ý nghĩa |
|---|---|
| `--data-root PATH` | Thư mục gốc dữ liệu (mặc định `data/raw`) |
| `--force` | Tải lại và giải nén đè dù đã có |
| `--keep-archive` | Giữ file zip sau khi giải nén |
| `--skip-validation` | Bỏ kiểm tra số ảnh so với manifest |

## Cấu trúc sau khi tải

```text
data/raw/
├── realwaste/
│   ├── MANIFEST.json
│   └── RealWaste/
│       ├── Cardboard/
│       ├── Glass/
│       └── ...
└── trashnet/
    ├── MANIFEST.json
    └── dataset-resized/
        ├── cardboard/
        ├── glass/
        └── ...
```

Cấu trúc gốc được giữ nguyên để bảo toàn provenance. Việc mapping về 5 lớp chung
(Cardboard, Glass, Metal, Paper, Plastic) thực hiện ở bước preprocessing, không sửa raw.

## Kiểm tra toàn vẹn

- Script kiểm tra số ảnh từng lớp so với manifest và báo lỗi nếu lệch.
- SHA256 của archive được ghi vào `MANIFEST.json` ở lần tải đầu (trust on first use).
  Các lần tải lại sẽ đối chiếu hash nếu manifest còn tồn tại.
- Giải nén chặn path traversal (zip-slip).

## Số lượng ảnh

| Lớp | RealWaste | TrashNet |
|---|---:|---:|
| Cardboard | 461 | 403 |
| Glass | 420 | 501 |
| Metal | 790 | 410 |
| Paper | 500 | 594 |
| Plastic | 921 | 482 |
| Food Organics | 411 | – |
| Miscellaneous Trash | 495 | – |
| Textile Trash | 318 | – |
| Vegetation | 436 | – |
| Trash | – | 137 |
| **Tổng** | **4.752** | **2.527** |

## Trích dẫn

- Single, S., Iranmanesh, S., & Raad, R. (2023). *RealWaste* [Dataset]. UCI Machine
  Learning Repository. <https://doi.org/10.24432/C5SS4G> (CC BY 4.0).
- Thung, G., & Yang, M. (2016). *Classification of Trash for Recyclability Status*.
  Stanford University. <https://github.com/garythung/trashnet> (MIT).
