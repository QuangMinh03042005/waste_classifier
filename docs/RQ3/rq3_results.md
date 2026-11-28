# Kết quả RQ3

Sinh tự động bởi `scripts/analyze_rq3.py` từ artifact của các run `completed`; test chỉ được đánh giá một lần mỗi run.

Tiêu chí: pooled cao hơn single-source ≥ 0.02 Macro-F1 trên **cả hai** nguồn test ngoài miền.

## Tổng hợp

| Protocol | Train | Test | n_seeds | n_test | Macro-F1 (mean ± std) | CI95 (±) | Accuracy | Balanced acc | Val Macro-F1 (in-domain) | Gap val−test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pooled (TrashNet 100%) | pooled | realwaste | 3 | 463 | 0.9057 ± 0.0042 | 0.0104 | 0.9057 | 0.9052 | 0.8962 | -0.0095 |
| Mix: TrashNet 25% | realwaste+trashnet | realwaste | 3 | 463 | 0.8980 ± 0.0090 | 0.0223 | 0.9021 | 0.8987 | 0.8596 | -0.0384 |
| Mix: TrashNet 75% | realwaste+trashnet | realwaste | 3 | 463 | 0.8922 ± 0.0061 | 0.0151 | 0.8949 | 0.8911 | 0.8874 | -0.0049 |
| Mix: TrashNet 50% | realwaste+trashnet | realwaste | 3 | 463 | 0.8907 ± 0.0044 | 0.0109 | 0.8934 | 0.8903 | 0.8711 | -0.0196 |
| Mix: TrashNet 0% | realwaste | realwaste | 3 | 463 | 0.8803 ± 0.0418 | 0.1038 | 0.8790 | 0.8778 | 0.6645 | -0.2159 |
| TrashNet → RealWaste | trashnet | realwaste | 3 | 3092 | 0.4788 ± 0.0265 | 0.0657 | 0.5246 | 0.4948 | 0.9023 | 0.4235 |
| Pooled (TrashNet 100%) | pooled | trashnet | 3 | 360 | 0.8889 ± 0.0031 | 0.0077 | 0.8917 | 0.8866 | 0.8962 | 0.0074 |
| Mix: TrashNet 75% | realwaste+trashnet | trashnet | 3 | 360 | 0.8608 ± 0.0141 | 0.0350 | 0.8630 | 0.8567 | 0.8874 | 0.0266 |
| Mix: TrashNet 50% | realwaste+trashnet | trashnet | 3 | 360 | 0.8389 ± 0.0084 | 0.0207 | 0.8407 | 0.8357 | 0.8711 | 0.0322 |
| Mix: TrashNet 25% | realwaste+trashnet | trashnet | 3 | 360 | 0.7908 ± 0.0074 | 0.0184 | 0.7963 | 0.7893 | 0.8596 | 0.0688 |
| Mix: TrashNet 0% | realwaste | trashnet | 3 | 360 | 0.3706 ± 0.0593 | 0.1473 | 0.3833 | 0.3986 | 0.6645 | 0.2939 |
| RealWaste → TrashNet | realwaste | trashnet | 3 | 2390 | 0.2894 ± 0.0261 | 0.0649 | 0.3032 | 0.3245 | 0.8987 | 0.6093 |
| Pooled (TrashNet 100%) | pooled | all | 3 | 823 | 0.8983 ± 0.0049 | 0.0121 | 0.8996 | 0.8973 | 0.8962 | -0.0021 |
| Mix: TrashNet 75% | realwaste+trashnet | all | 3 | 823 | 0.8795 ± 0.0089 | 0.0222 | 0.8809 | 0.8789 | 0.8874 | 0.0078 |
| Mix: TrashNet 50% | realwaste+trashnet | all | 3 | 823 | 0.8688 ± 0.0018 | 0.0045 | 0.8704 | 0.8681 | 0.8711 | 0.0023 |
| Mix: TrashNet 25% | realwaste+trashnet | all | 3 | 823 | 0.8535 ± 0.0072 | 0.0178 | 0.8558 | 0.8549 | 0.8596 | 0.0061 |
| Mix: TrashNet 0% | realwaste | all | 3 | 823 | 0.6613 ± 0.0032 | 0.0080 | 0.6622 | 0.6635 | 0.6645 | 0.0032 |

## Lưu ý phương pháp

- **Single-source:** test là **toàn bộ** nguồn còn lại (TrashNet 2.390 hoặc RealWaste 3.092) — mô hình chưa từng thấy nguồn đó, nên đây là test ngoài miền thuần.
- **Pooled (100%) và các mức mix:** test là phần held-out của split v1 theo từng nguồn (TrashNet 360, RealWaste 463) — nguồn đó đã xuất hiện trong train ở mức tương ứng.
- So sánh công bằng dùng cách chấm single-source **trên cùng held-out test** (mục dưới): chỉ lọc predictions đã có, không đánh giá lại test.
- CI95 tính theo phân phối t với n=3 nên khá rộng. Hiệu số nhỏ hơn CI (ví dụ giữa các mức mix gần nhau) **không** nên đọc là khác biệt thực.
- Có nhiều biến thể × nhiều chỉ số nhưng không hiệu chỉnh multiple comparisons; các so sánh ngoài tiêu chí đăng ký chỉ mang tính exploratory.

## Pooled so với single-source

| Test source | Single-source | Single (mean ± std) | Pooled (mean ± std) | Delta | Đạt ≥ 0.02 |
| --- | --- | --- | --- | --- | --- |
| realwaste | TrashNet → RealWaste | 0.4788 ± 0.0265 | 0.9057 ± 0.0042 | 0.4269 | có |
| trashnet | RealWaste → TrashNet | 0.2894 ± 0.0261 | 0.8889 ± 0.0031 | 0.5995 | có |

### Trên cùng held-out test

| Test source | Single (held-out, mean ± std) | Pooled (mean ± std) | Delta held-out |
| --- | --- | --- | --- |
| realwaste | 0.4749 ± 0.0286 | 0.9057 ± 0.0042 | 0.4307 |
| trashnet | 0.3093 ± 0.0218 | 0.8889 ± 0.0031 | 0.5795 |

### Delta ghép cặp theo seed

Cùng seed nên khử được phần lớn phương sai giữa seed; dùng để kiểm tra delta có lớn hơn nhiễu hay không (n=3, không báo p-value).

| Test source | n cặp | Delta trung bình | SD | CI95 (±) | Delta từng seed |
| --- | --- | --- | --- | --- | --- |
| realwaste | 3 | 0.4307 | 0.0270 | 0.0672 | 123: +0.438, 2024: +0.401, 42: +0.454 |
| trashnet | 3 | 0.5795 | 0.0210 | 0.0521 | 123: +0.590, 2024: +0.593, 42: +0.555 |

**Kết luận:** đạt tiêu chí đã đăng ký (delta ≥ 0.02) trên cả hai nguồn. Delta trên cùng held-out test xác nhận kết luận không phải do khác kích thước test set.

## Tỉ lệ dữ liệu nguồn thứ hai (mix-ratio)

RealWaste giữ 100% train; cột là tỉ lệ group TrashNet train được thêm vào.

| TrashNet % | Macro-F1 realwaste | Macro-F1 trashnet |
| --- | --- | --- |
| 0 | 0.8803 ± 0.0418 | 0.3706 ± 0.0593 |
| 25 | 0.8980 ± 0.0090 | 0.7908 ± 0.0074 |
| 50 | 0.8907 ± 0.0044 | 0.8389 ± 0.0084 |
| 75 | 0.8922 ± 0.0061 | 0.8608 ± 0.0141 |
| 100 | 0.9057 ± 0.0042 | 0.8889 ± 0.0031 |

- Tăng 0% → 25%: **+0.4202** Macro-F1 trên TrashNet (tiêu chí ≥ +0.05).
- Mức 50%: **0.8389** trên TrashNet (tiêu chí ≥ 0.80).
- RealWaste giảm khi lên 50%: **-0.0104** (tiêu chí ≤ 0.03).
- Tiêu chí plan: **đạt**.

**Lưu ý:** val dùng cả hai nguồn ở mọi mức tỉ lệ. Với mức 0%, val TrashNet là ngoài miền nên epoch được chọn có thể chưa tối ưu cho RealWaste; đọc kèm cột Val Macro-F1 và Gap val−test trong bảng tổng hợp.

## Per-class F1 trên test

| label | test_source | cardboard | glass | metal | paper | plastic |
| --- | --- | --- | --- | --- | --- | --- |
| Mix: TrashNet 0% | all | 0.5324 | 0.6128 | 0.7261 | 0.7131 | 0.7221 |
| Mix: TrashNet 0% | realwaste | 0.8611 | 0.8950 | 0.8726 | 0.8992 | 0.8737 |
| Mix: TrashNet 0% | trashnet | 0.3642 | 0.2729 | 0.4414 | 0.4937 | 0.2806 |
| Mix: TrashNet 25% | all | 0.8439 | 0.8337 | 0.8465 | 0.8662 | 0.8771 |
| Mix: TrashNet 25% | realwaste | 0.8428 | 0.9320 | 0.9103 | 0.8819 | 0.9232 |
| Mix: TrashNet 25% | trashnet | 0.8451 | 0.7546 | 0.7120 | 0.8530 | 0.7893 |
| Mix: TrashNet 50% | all | 0.8704 | 0.8465 | 0.8594 | 0.8836 | 0.8841 |
| Mix: TrashNet 50% | realwaste | 0.8424 | 0.9214 | 0.8902 | 0.8875 | 0.9122 |
| Mix: TrashNet 50% | trashnet | 0.9017 | 0.7840 | 0.7956 | 0.8804 | 0.8328 |
| Mix: TrashNet 75% | all | 0.8690 | 0.8769 | 0.8668 | 0.8900 | 0.8948 |
| Mix: TrashNet 75% | realwaste | 0.8458 | 0.9239 | 0.8902 | 0.8880 | 0.9132 |
| Mix: TrashNet 75% | trashnet | 0.8945 | 0.8394 | 0.8188 | 0.8918 | 0.8594 |
| Pooled (TrashNet 100%) | all | 0.8973 | 0.8842 | 0.8736 | 0.9257 | 0.9109 |
| Pooled (TrashNet 100%) | realwaste | 0.8801 | 0.9239 | 0.8871 | 0.9194 | 0.9180 |
| Pooled (TrashNet 100%) | trashnet | 0.9167 | 0.8513 | 0.8474 | 0.9311 | 0.8977 |
| RealWaste → TrashNet | trashnet | 0.3141 | 0.1883 | 0.3847 | 0.3284 | 0.2314 |
| TrashNet → RealWaste | realwaste | 0.3313 | 0.3227 | 0.5986 | 0.5977 | 0.5438 |

## Sụp đổ dự đoán (domain shift)

Tỉ lệ nhãn dự đoán nhiều nhất trên test; tỉ lệ cao cho thấy mô hình dồn về một lớp khi đổi miền.

| Protocol | Test | Lớp dự đoán nhiều nhất | Tỉ lệ |
| --- | --- | --- | --- |
| RealWaste → TrashNet | trashnet | cardboard | 0.5820 |
| TrashNet → RealWaste | realwaste | plastic | 0.3110 |
| Pooled (TrashNet 100%) | realwaste | plastic | 0.3070 |
| Pooled (TrashNet 100%) | trashnet | paper | 0.2640 |
| Mix: TrashNet 0% | realwaste | plastic | 0.2900 |
| Mix: TrashNet 0% | trashnet | cardboard | 0.5360 |
| Mix: TrashNet 25% | realwaste | plastic | 0.2930 |
| Mix: TrashNet 25% | trashnet | paper | 0.2760 |
| Mix: TrashNet 50% | realwaste | plastic | 0.3000 |
| Mix: TrashNet 50% | trashnet | paper | 0.2700 |
| Mix: TrashNet 75% | realwaste | plastic | 0.3060 |
| Mix: TrashNet 75% | trashnet | paper | 0.2740 |

## Hình

- `test_macro_f1.png`
- `dose_response.png`
- `confusion_matrices.png`

