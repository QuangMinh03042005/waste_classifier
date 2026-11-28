# Kết quả RQ3

Sinh tự động bởi `scripts/analyze_rq3.py` từ artifact của các run `completed`; test chỉ được đánh giá một lần mỗi run.

Tiêu chí: pooled cao hơn single-source ≥ 0.02 Macro-F1 trên **cả hai** nguồn test ngoài miền.

## Tổng hợp

| Protocol | Train | Test | n_seeds | n_test | Macro-F1 (mean ± std) | CI95 (±) | Accuracy | Balanced acc | Val Macro-F1 (in-domain) | Gap val−test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mix: TrashNet 25% | realwaste+trashnet | realwaste | 3 | 463 | 0.9006 ± 0.0082 | 0.0205 | 0.9035 | 0.9016 | 0.8554 | -0.0453 |
| Pooled (TrashNet 100%) | pooled | realwaste | 5 | 463 | 0.8992 ± 0.0037 | 0.0045 | 0.9015 | 0.8999 | 0.8948 | -0.0044 |
| Mix: TrashNet 50% | realwaste+trashnet | realwaste | 3 | 463 | 0.8932 ± 0.0083 | 0.0206 | 0.8956 | 0.8931 | 0.8723 | -0.0209 |
| Mix: TrashNet 75% | realwaste+trashnet | realwaste | 3 | 463 | 0.8903 ± 0.0019 | 0.0046 | 0.8949 | 0.8890 | 0.8866 | -0.0037 |
| Mix: TrashNet 0% | realwaste | realwaste | 3 | 463 | 0.8697 ± 0.0357 | 0.0886 | 0.8690 | 0.8670 | 0.6662 | -0.2034 |
| TrashNet → RealWaste | trashnet | realwaste | 5 | 3092 | 0.4694 ± 0.0154 | 0.0191 | 0.5173 | 0.4924 | 0.9120 | 0.4426 |
| Pooled (TrashNet 100%) | pooled | trashnet | 5 | 360 | 0.8869 ± 0.0081 | 0.0100 | 0.8889 | 0.8850 | 0.8948 | 0.0078 |
| Mix: TrashNet 75% | realwaste+trashnet | trashnet | 3 | 360 | 0.8614 ± 0.0065 | 0.0161 | 0.8630 | 0.8568 | 0.8866 | 0.0252 |
| Mix: TrashNet 50% | realwaste+trashnet | trashnet | 3 | 360 | 0.8413 ± 0.0058 | 0.0145 | 0.8435 | 0.8382 | 0.8723 | 0.0310 |
| Mix: TrashNet 25% | realwaste+trashnet | trashnet | 3 | 360 | 0.7860 ± 0.0206 | 0.0513 | 0.7935 | 0.7845 | 0.8554 | 0.0694 |
| Mix: TrashNet 0% | realwaste | trashnet | 3 | 360 | 0.3826 ± 0.0518 | 0.1288 | 0.3917 | 0.4104 | 0.6662 | 0.2837 |
| RealWaste → TrashNet | realwaste | trashnet | 5 | 2390 | 0.2971 ± 0.0432 | 0.0537 | 0.3078 | 0.3297 | 0.8996 | 0.6025 |
| Pooled (TrashNet 100%) | pooled | all | 5 | 823 | 0.8945 ± 0.0059 | 0.0073 | 0.8960 | 0.8944 | 0.8948 | 0.0003 |
| Mix: TrashNet 75% | realwaste+trashnet | all | 3 | 823 | 0.8790 ± 0.0041 | 0.0102 | 0.8809 | 0.8781 | 0.8866 | 0.0076 |
| Mix: TrashNet 50% | realwaste+trashnet | all | 3 | 823 | 0.8714 ± 0.0063 | 0.0157 | 0.8728 | 0.8711 | 0.8723 | 0.0009 |
| Mix: TrashNet 25% | realwaste+trashnet | all | 3 | 823 | 0.8531 ± 0.0129 | 0.0320 | 0.8554 | 0.8546 | 0.8554 | 0.0023 |
| Mix: TrashNet 0% | realwaste | all | 3 | 823 | 0.6593 ± 0.0050 | 0.0125 | 0.6602 | 0.6625 | 0.6662 | 0.0069 |

## Môi trường và thời gian

| Protocol | n run | GPU | Torch | CUDA | Phút/run | Params |
| --- | --- | --- | --- | --- | --- | --- |
| RealWaste → TrashNet | 5 | NVIDIA GeForce RTX 5070 | 2.15.0+cu132 | 13.2 | 2.7 | 1,522,981 |
| TrashNet → RealWaste | 5 | NVIDIA GeForce RTX 5070 | 2.15.0+cu132 | 13.2 | 2.3 | 1,522,981 |
| Pooled (TrashNet 100%) | 5 | NVIDIA GeForce RTX 5070 | 2.15.0+cu132 | 13.2 | 5.0 | 1,522,981 |
| Mix: TrashNet 0% | 3 | NVIDIA GeForce RTX 5070 | 2.15.0+cu132 | 13.2 | 2.3 | 1,522,981 |
| Mix: TrashNet 25% | 3 | NVIDIA GeForce RTX 5070 | 2.15.0+cu132 | 13.2 | 4.3 | 1,522,981 |
| Mix: TrashNet 50% | 3 | NVIDIA GeForce RTX 5070 | 2.15.0+cu132 | 13.2 | 3.8 | 1,522,981 |
| Mix: TrashNet 75% | 3 | NVIDIA GeForce RTX 5070 | 2.15.0+cu132 | 13.2 | 3.9 | 1,522,981 |

## Lưu ý phương pháp

- **Single-source:** test là **toàn bộ** nguồn còn lại (TrashNet 2.390 hoặc RealWaste 3.092) — mô hình chưa từng thấy nguồn đó, nên đây là test ngoài miền thuần.
- **Pooled (100%) và các mức mix:** test là phần held-out của split v1 theo từng nguồn (TrashNet 360, RealWaste 463) — nguồn đó đã xuất hiện trong train ở mức tương ứng.
- So sánh công bằng dùng cách chấm single-source **trên cùng held-out test** (mục dưới): chỉ lọc predictions đã có, không đánh giá lại test.
- CI95 tính theo phân phối t với số seed thực tế của từng arm (3 cho mix-ratio, 5 cho 3 protocol chính) nên vẫn khá rộng. Hiệu số nhỏ hơn CI (ví dụ giữa các mức mix gần nhau) **không** nên đọc là khác biệt thực.
- Có nhiều biến thể × nhiều chỉ số nhưng không hiệu chỉnh multiple comparisons; các so sánh ngoài tiêu chí đăng ký chỉ mang tính exploratory.

## Pooled so với single-source

| Test source | Single-source | Single (mean ± std) | Pooled (mean ± std) | Delta | Đạt ≥ 0.02 |
| --- | --- | --- | --- | --- | --- |
| realwaste | TrashNet → RealWaste | 0.4694 ± 0.0154 | 0.8992 ± 0.0037 | 0.4298 | có |
| trashnet | RealWaste → TrashNet | 0.2971 ± 0.0432 | 0.8869 ± 0.0081 | 0.5898 | có |

### Trên cùng held-out test

| Test source | Single (held-out, mean ± std) | Pooled (mean ± std) | Delta held-out |
| --- | --- | --- | --- |
| realwaste | 0.4672 ± 0.0252 | 0.8992 ± 0.0037 | 0.4320 |
| trashnet | 0.3190 ± 0.0392 | 0.8869 ± 0.0081 | 0.5679 |

### Delta ghép cặp theo seed

Cùng seed nên khử được phần lớn phương sai giữa seed; dùng để kiểm tra delta có lớn hơn nhiễu hay không (n nhỏ, không báo p-value).

| Test source | n cặp | Delta trung bình | SD | CI95 (±) | Delta từng seed |
| --- | --- | --- | --- | --- | --- |
| realwaste | 5 | 0.4320 | 0.0283 | 0.0352 | 123: +0.421, 2024: +0.434, 3407: +0.431, 42: +0.398, 7: +0.476 |
| trashnet | 5 | 0.5679 | 0.0383 | 0.0476 | 123: +0.589, 2024: +0.593, 3407: +0.511, 42: +0.546, 7: +0.601 |

**Kết luận:** đạt tiêu chí đã đăng ký (delta ≥ 0.02) trên cả hai nguồn. Delta trên cùng held-out test xác nhận kết luận không phải do khác kích thước test set.

## Tỉ lệ dữ liệu nguồn thứ hai (mix-ratio)

RealWaste giữ 100% train; cột là tỉ lệ group TrashNet train được thêm vào.

| TrashNet % | Macro-F1 realwaste | Macro-F1 trashnet |
| --- | --- | --- |
| 0 | 0.8697 ± 0.0357 | 0.3826 ± 0.0518 |
| 25 | 0.9006 ± 0.0082 | 0.7860 ± 0.0206 |
| 50 | 0.8932 ± 0.0083 | 0.8413 ± 0.0058 |
| 75 | 0.8903 ± 0.0019 | 0.8614 ± 0.0065 |
| 100 | 0.8992 ± 0.0037 | 0.8869 ± 0.0081 |

- Tăng 0% → 25%: **+0.4034** Macro-F1 trên TrashNet (tiêu chí ≥ +0.05).
- Mức 50%: **0.8413** trên TrashNet (tiêu chí ≥ 0.80).
- RealWaste giảm khi lên 50%: **-0.0236** (tiêu chí ≤ 0.03).
- Tiêu chí plan: **đạt**.

**Lưu ý:** val dùng cả hai nguồn ở mọi mức tỉ lệ. Với mức 0%, val TrashNet là ngoài miền nên epoch được chọn có thể chưa tối ưu cho RealWaste; đọc kèm cột Val Macro-F1 và Gap val−test trong bảng tổng hợp.

## Per-class F1 trên test

| label | test_source | cardboard | glass | metal | paper | plastic |
| --- | --- | --- | --- | --- | --- | --- |
| Mix: TrashNet 0% | all | 0.5376 | 0.6246 | 0.7333 | 0.6865 | 0.7144 |
| Mix: TrashNet 0% | realwaste | 0.8496 | 0.8749 | 0.8637 | 0.8951 | 0.8650 |
| Mix: TrashNet 0% | trashnet | 0.3717 | 0.3433 | 0.4786 | 0.4313 | 0.2880 |
| Mix: TrashNet 25% | all | 0.8545 | 0.8247 | 0.8364 | 0.8713 | 0.8783 |
| Mix: TrashNet 25% | realwaste | 0.8627 | 0.9275 | 0.9077 | 0.8850 | 0.9203 |
| Mix: TrashNet 25% | trashnet | 0.8456 | 0.7435 | 0.6821 | 0.8597 | 0.7991 |
| Mix: TrashNet 50% | all | 0.8760 | 0.8477 | 0.8583 | 0.8888 | 0.8861 |
| Mix: TrashNet 50% | realwaste | 0.8490 | 0.9189 | 0.8917 | 0.8945 | 0.9120 |
| Mix: TrashNet 50% | trashnet | 0.9061 | 0.7889 | 0.7896 | 0.8841 | 0.8379 |
| Mix: TrashNet 75% | all | 0.8624 | 0.8785 | 0.8716 | 0.8840 | 0.8985 |
| Mix: TrashNet 75% | realwaste | 0.8319 | 0.9234 | 0.8926 | 0.8811 | 0.9224 |
| Mix: TrashNet 75% | trashnet | 0.8964 | 0.8425 | 0.8279 | 0.8865 | 0.8534 |
| Pooled (TrashNet 100%) | all | 0.8875 | 0.8819 | 0.8827 | 0.9101 | 0.9101 |
| Pooled (TrashNet 100%) | realwaste | 0.8667 | 0.9172 | 0.8957 | 0.8976 | 0.9186 |
| Pooled (TrashNet 100%) | trashnet | 0.9103 | 0.8529 | 0.8565 | 0.9210 | 0.8941 |
| RealWaste → TrashNet | trashnet | 0.3140 | 0.2188 | 0.3962 | 0.3256 | 0.2310 |
| TrashNet → RealWaste | realwaste | 0.2977 | 0.3493 | 0.6134 | 0.5588 | 0.5279 |

## Cặp nhầm phổ biến

Cộng dồn toàn bộ seed của từng arm; tỉ lệ tính trên số ảnh thật của lớp đó.

| Protocol | Test | Nhãn thật → đoán | Số ảnh | Tỉ lệ trên lớp thật |
| --- | --- | --- | --- | --- |
| RealWaste → TrashNet | trashnet | paper → cardboard | 2144 | 0.7220 |
| RealWaste → TrashNet | trashnet | plastic → cardboard | 1421 | 0.5900 |
| RealWaste → TrashNet | trashnet | glass → cardboard | 1275 | 0.5090 |
| TrashNet → RealWaste | realwaste | cardboard → paper | 1269 | 0.5510 |
| TrashNet → RealWaste | realwaste | plastic → paper | 995 | 0.2160 |
| TrashNet → RealWaste | realwaste | plastic → metal | 921 | 0.2000 |
| Pooled (TrashNet 100%) | realwaste | cardboard → paper | 37 | 0.1070 |
| Pooled (TrashNet 100%) | realwaste | metal → plastic | 34 | 0.0580 |
| Pooled (TrashNet 100%) | realwaste | plastic → metal | 33 | 0.0480 |
| Pooled (TrashNet 100%) | trashnet | metal → glass | 34 | 0.1100 |
| Pooled (TrashNet 100%) | trashnet | glass → plastic | 26 | 0.0690 |
| Pooled (TrashNet 100%) | trashnet | cardboard → paper | 23 | 0.0750 |
| Mix: TrashNet 0% | realwaste | plastic → metal | 46 | 0.1110 |
| Mix: TrashNet 0% | realwaste | glass → plastic | 22 | 0.1160 |
| Mix: TrashNet 0% | realwaste | cardboard → paper | 21 | 0.1010 |
| Mix: TrashNet 0% | trashnet | paper → cardboard | 168 | 0.6290 |
| Mix: TrashNet 0% | trashnet | plastic → cardboard | 106 | 0.4840 |
| Mix: TrashNet 0% | trashnet | glass → cardboard | 86 | 0.3820 |
| Mix: TrashNet 25% | realwaste | cardboard → paper | 23 | 0.1110 |
| Mix: TrashNet 25% | realwaste | metal → plastic | 23 | 0.0650 |
| Mix: TrashNet 25% | realwaste | paper → cardboard | 14 | 0.0620 |
| Mix: TrashNet 25% | trashnet | metal → glass | 36 | 0.1940 |
| Mix: TrashNet 25% | trashnet | plastic → glass | 27 | 0.1230 |
| Mix: TrashNet 25% | trashnet | cardboard → paper | 23 | 0.1260 |
| Mix: TrashNet 50% | realwaste | metal → plastic | 25 | 0.0710 |
| Mix: TrashNet 50% | realwaste | cardboard → paper | 24 | 0.1160 |
| Mix: TrashNet 50% | realwaste | plastic → metal | 22 | 0.0530 |
| Mix: TrashNet 50% | trashnet | glass → plastic | 25 | 0.1110 |
| Mix: TrashNet 50% | trashnet | metal → glass | 23 | 0.1240 |
| Mix: TrashNet 50% | trashnet | plastic → glass | 20 | 0.0910 |
| Mix: TrashNet 75% | realwaste | cardboard → paper | 28 | 0.1350 |
| Mix: TrashNet 75% | realwaste | metal → plastic | 22 | 0.0620 |
| Mix: TrashNet 75% | realwaste | plastic → metal | 17 | 0.0410 |
| Mix: TrashNet 75% | trashnet | cardboard → paper | 22 | 0.1200 |
| Mix: TrashNet 75% | trashnet | glass → plastic | 21 | 0.0930 |
| Mix: TrashNet 75% | trashnet | metal → glass | 21 | 0.1130 |

## Sụp đổ dự đoán (domain shift)

Tỉ lệ nhãn dự đoán nhiều nhất trên test; tỉ lệ cao cho thấy mô hình dồn về một lớp khi đổi miền.

| Protocol | Test | Lớp dự đoán nhiều nhất | Tỉ lệ |
| --- | --- | --- | --- |
| RealWaste → TrashNet | trashnet | cardboard | 0.6040 |
| TrashNet → RealWaste | realwaste | paper | 0.3290 |
| Pooled (TrashNet 100%) | realwaste | plastic | 0.2960 |
| Pooled (TrashNet 100%) | trashnet | paper | 0.2590 |
| Mix: TrashNet 0% | realwaste | plastic | 0.2880 |
| Mix: TrashNet 0% | trashnet | cardboard | 0.5270 |
| Mix: TrashNet 25% | realwaste | plastic | 0.2980 |
| Mix: TrashNet 25% | trashnet | paper | 0.2740 |
| Mix: TrashNet 50% | realwaste | plastic | 0.2990 |
| Mix: TrashNet 50% | trashnet | paper | 0.2720 |
| Mix: TrashNet 75% | realwaste | plastic | 0.3030 |
| Mix: TrashNet 75% | trashnet | paper | 0.2750 |

## Hình

- `test_macro_f1.png`
- `dose_response.png`
- `learning_curves_mix.png`
- `learning_curves_cross.png`
- `confusion_matrices.png`

