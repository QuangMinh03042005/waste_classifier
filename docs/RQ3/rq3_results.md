# Kết quả RQ3

Sinh tự động bởi `scripts/analyze_rq3.py` từ artifact của các run `completed`; test chỉ được đánh giá một lần mỗi run.

Tiêu chí: pooled cao hơn single-source ≥ 0.02 Macro-F1 trên **cả hai** nguồn test ngoài miền.

## Tổng hợp

| Protocol | Train | Test | n_seeds | n_test | Macro-F1 (mean ± std) | Accuracy | Balanced acc | Val Macro-F1 (in-domain) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pooled → từng nguồn | pooled | realwaste | 3 | 463 | 0.9057 ± 0.0042 | 0.9057 | 0.9052 | 0.8962 |
| TrashNet → RealWaste | trashnet | realwaste | 3 | 3092 | 0.4788 ± 0.0265 | 0.5246 | 0.4948 | 0.9023 |
| Pooled → từng nguồn | pooled | trashnet | 3 | 360 | 0.8889 ± 0.0031 | 0.8917 | 0.8866 | 0.8962 |
| RealWaste → TrashNet | realwaste | trashnet | 3 | 2390 | 0.2894 ± 0.0261 | 0.3032 | 0.3245 | 0.8987 |
| Pooled → từng nguồn | pooled | all | 3 | 823 | 0.8983 ± 0.0049 | 0.8996 | 0.8973 | 0.8962 |

## Lưu ý phương pháp

- **Single-source:** test là **toàn bộ** nguồn còn lại (TrashNet 2.390 hoặc RealWaste 3.092) — mô hình chưa từng thấy nguồn đó, nên đây là test ngoài miền thuần.
- **Pooled:** test là phần held-out của split v1 theo từng nguồn (TrashNet 360, RealWaste 463) — nguồn đó đã xuất hiện trong train, nên đây là test in-domain trên phần giữ lại.
- Vì vậy chênh lệch pooled − single-source phản ánh **cả hai** yếu tố: bổ sung dữ liệu và khác biệt bản chất/kích thước test set. Không nên đọc delta như mức cải thiện thuần túy của khả năng chuyển miền; cần nêu rõ khi báo cáo.

## Pooled so với single-source

| Test source | Single-source | Single (mean ± std) | Pooled (mean ± std) | Delta | Đạt ≥ 0.02 |
| --- | --- | --- | --- | --- | --- |
| realwaste | TrashNet → RealWaste | 0.4788 ± 0.0265 | 0.9057 ± 0.0042 | 0.4269 | có |
| trashnet | RealWaste → TrashNet | 0.2894 ± 0.0261 | 0.8889 ± 0.0031 | 0.5995 | có |

**Kết luận:** đạt tiêu chí đã đăng ký (delta ≥ 0.02) trên cả hai nguồn. Đọc kèm lưu ý phương pháp: pooled được test in-domain trên phần held-out, single-source được test ngoài miền trên toàn bộ nguồn còn lại.

## Per-class F1 trên test

| label | test_source | cardboard | glass | metal | paper | plastic |
| --- | --- | --- | --- | --- | --- | --- |
| Pooled → từng nguồn | all | 0.8973 | 0.8842 | 0.8736 | 0.9257 | 0.9109 |
| Pooled → từng nguồn | realwaste | 0.8801 | 0.9239 | 0.8871 | 0.9194 | 0.9180 |
| Pooled → từng nguồn | trashnet | 0.9167 | 0.8513 | 0.8474 | 0.9311 | 0.8977 |
| RealWaste → TrashNet | trashnet | 0.3141 | 0.1883 | 0.3847 | 0.3284 | 0.2314 |
| TrashNet → RealWaste | realwaste | 0.3313 | 0.3227 | 0.5986 | 0.5977 | 0.5438 |

## Sụp đổ dự đoán (domain shift)

Tỉ lệ nhãn dự đoán nhiều nhất trên test; tỉ lệ cao cho thấy mô hình dồn về một lớp khi đổi miền.

| Protocol | Test | Lớp dự đoán nhiều nhất | Tỉ lệ |
| --- | --- | --- | --- |
| RealWaste → TrashNet | trashnet | cardboard | 0.5820 |
| TrashNet → RealWaste | realwaste | plastic | 0.3110 |
| Pooled → từng nguồn | realwaste | plastic | 0.3070 |
| Pooled → từng nguồn | trashnet | paper | 0.2640 |

## Hình

- `test_macro_f1.png`
- `confusion_matrices.png`

