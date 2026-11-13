# Mapping nhãn

Dùng 5 lớp chung giữa RealWaste và TrashNet:

```text
cardboard, glass, metal, paper, plastic
```

Tên lưu trong code/CSV dùng chữ thường; tên hiển thị trong báo cáo viết hoa.
Nguồn sự thật: `src/data/labels.py`.

## Mapping

| Dataset | Nhãn gốc | Nhãn chuẩn | Số ảnh |
|---|---|---|---:|
| RealWaste | Cardboard | cardboard | 461 |
| RealWaste | Glass | glass | 420 |
| RealWaste | Metal | metal | 790 |
| RealWaste | Paper | paper | 500 |
| RealWaste | Plastic | plastic | 921 |
| **RealWaste** | | **5 lớp** | **3.092** |
| TrashNet | cardboard | cardboard | 403 |
| TrashNet | glass | glass | 501 |
| TrashNet | metal | metal | 410 |
| TrashNet | paper | paper | 594 |
| TrashNet | plastic | plastic | 482 |
| **TrashNet** | | **5 lớp** | **2.390** |
| **Tổng** | | | **5.482** |

## Lớp bị loại

| Dataset | Nhãn gốc | Số ảnh | Lý do |
|---|---|---:|---|
| RealWaste | Food Organics | 411 | không có lớp tương ứng ở TrashNet |
| RealWaste | Miscellaneous Trash | 495 | không đồng nhất ngữ nghĩa với `trash` của TrashNet |
| RealWaste | Textile Trash | 318 | không có lớp tương ứng ở TrashNet |
| RealWaste | Vegetation | 436 | không có lớp tương ứng ở TrashNet |
| TrashNet | trash | 137 | lớp gom nhóm không đồng nhất |

Tổng bị loại: **1.797 ảnh**. Không dùng các lớp này trong RQ1/RQ2.

## Quy tắc

- Không đổi mapping sau khi đã tạo split v1 (sẽ làm lệch kết quả).
- Nếu cần thay đổi, tạo split version mới và ghi rõ lý do.
- `canonical_label(dataset, raw_label)` trả `None` cho lớp bị loại và raise lỗi
  nếu gặp nhãn lạ (bảo vệ khi dataset có thêm lớp mới).
- Thứ tự lớp cố định: `CANONICAL_CLASSES`; luôn lưu `class_names` vào metrics/checkpoint.
