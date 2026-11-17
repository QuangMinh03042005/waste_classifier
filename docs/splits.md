# Split dữ liệu

## Cách tạo

```bash
uv run python scripts/make_splits.py
```

Mặc định tạo `data/splits/v1/` gồm:

```text
data/splits/v1/
├── split.csv       # 5.482 dòng, commit vào git
└── summary.json    # thống kê + cấu hình split, commit vào git
```

`split.csv` dùng đường dẫn tương đối so với data root:

```text
path,dataset,label,raw_label,split,group_id
raw/realwaste/RealWaste/Cardboard/Cardboard_1.jpg,realwaste,cardboard,Cardboard,val,g000001
```

Trên Colab với `WASTE_DATA_ROOT=/content/data`, đường dẫn được đọc thành
`/content/data/raw/...` mà không cần sửa CSV.

## Cấu hình v1

| Tham số | Giá trị |
|---|---|
| Seed | 42 |
| Tỷ lệ | 70% train / 15% val / 15% test |
| Stratify | theo `(dataset, label)` |
| Chống leakage | gom ảnh gần trùng bằng pHash, Hamming ≤ 5 |
| Đơn vị chia | group (không tách ảnh trùng khác split) |
| Số ảnh | 5.482 |
| Số group | 5.447 |
| Ảnh trùng phát hiện | 35 |
| Group lớn nhất | 3 ảnh |

## Kết quả

| Split | RealWaste | TrashNet | Tổng |
|---|---:|---:|---:|
| Train | 2.163 | 1.670 | 3.833 |
| Validation | 466 | 360 | 826 |
| Test | 463 | 360 | 823 |
| **Tổng** | **3.092** | **2.390** | **5.482** |

Theo lớp:

| Lớp | Train | Val | Test | Tổng |
|---|---:|---:|---:|---:|
| cardboard | 603 | 131 | 130 | 864 |
| glass | 645 | 138 | 138 | 921 |
| metal | 839 | 181 | 180 | 1.200 |
| paper | 766 | 164 | 164 | 1.094 |
| plastic | 980 | 212 | 211 | 1.403 |

Mỗi lớp đều có ≥130 ảnh ở val và test.

## Quy tắc sử dụng

- `split.csv` được tạo **một lần**, commit, dùng cho mọi thí nghiệm so sánh.
- Không chia lại cho từng RQ; RQ3 dùng test ngoài miền theo protocol riêng nhưng
  vẫn tham chiếu split v1 cho phần train/val.
- Không sửa `split.csv` sau khi đã chạy thí nghiệm.
- Nếu cần thay đổi (thêm dữ liệu, đổi mapping), tạo `data/splits/v2/` và chạy lại
  toàn bộ thí nghiệm liên quan.
- Test set chỉ dùng đánh giá cuối; không dùng để chọn epoch, threshold hay cấu hình.

## Kiểm tra

Script tự kiểm tra:

- Mọi ảnh đều có split.
- Không group nào xuất hiện ở hai split.
- Tỷ lệ tổng bằng 1.
- Split chỉ chứa `train`/`val`/`test`.
- Báo lỗi nếu gặp nhãn gốc lạ (tránh mapping sai âm thầm).

## Split cho RQ3

Sinh bằng:

```bash
uv run python scripts/make_rq3_splits.py
```

Kết quả trong `data/splits/rq3/`:

| File | Train | Val | Test |
|---|---:|---:|---:|
| `rw_to_tn.csv` | RealWaste 2.163 | RealWaste 466 | **toàn bộ TrashNet 2.390** |
| `tn_to_rw.csv` | TrashNet 1.670 | TrashNet 360 | **toàn bộ RealWaste 3.092** |
| `pooled.csv` | 3.833 | 826 | 823 (test từng nguồn) |

- Không sửa split v1.
- Nếu một group chứa ảnh từ cả 2 nguồn, group đó bị loại khỏi cross-dataset split để chống leakage.
- Với P1/P2, test là toàn bộ nguồn còn lại vì nguồn đó không xuất hiện trong train.
- Với P3, train pooled bằng `pooled.csv`; đánh giá riêng từng nguồn trên split test.
