# Kết quả RQ1

Sinh tự động bởi `scripts/analyze_rq1.py`; test được đánh giá một lần cho mỗi phiên bản. Scratch có tái sử dụng test sau khi sửa recipe, xem `plans/rq1/scratch_final_v2.yaml`.

## Tổng hợp

| label | n_seeds | Macro-F1 mean | Macro-F1 std | Accuracy mean | Params |
| --- | --- | --- | --- | --- | --- |
| MobileNetV3_small pretrained | 5 | 0.8901 | 0.0114 | 0.8914 | 1522981.0000 |
| Custom CNN | 5 | 0.8870 | 0.0060 | 0.8868 | 1174501.0000 |
| MobileNetV3_small scratch (p20) | 5 | 0.6734 | 0.0144 | 0.6722 | 1522981.0000 |

## Theo seed

| label | seed | macro_f1 | accuracy | best_val_macro_f1 |
| --- | --- | --- | --- | --- |
| Custom CNN | 7 | 0.8880 | 0.8870 | 0.8664 |
| Custom CNN | 42 | 0.8906 | 0.8906 | 0.8809 |
| Custom CNN | 123 | 0.8763 | 0.8761 | 0.8607 |
| Custom CNN | 2024 | 0.8901 | 0.8906 | 0.8708 |
| Custom CNN | 3407 | 0.8899 | 0.8894 | 0.8728 |
| MobileNetV3_small pretrained | 7 | 0.8900 | 0.8906 | 0.8925 |
| MobileNetV3_small pretrained | 42 | 0.8833 | 0.8858 | 0.8809 |
| MobileNetV3_small pretrained | 123 | 0.8793 | 0.8797 | 0.8974 |
| MobileNetV3_small pretrained | 2024 | 0.9089 | 0.9101 | 0.9005 |
| MobileNetV3_small pretrained | 3407 | 0.8888 | 0.8906 | 0.8935 |
| MobileNetV3_small scratch (p20) | 7 | 0.6799 | 0.6780 | 0.6791 |
| MobileNetV3_small scratch (p20) | 42 | 0.6804 | 0.6829 | 0.7039 |
| MobileNetV3_small scratch (p20) | 123 | 0.6792 | 0.6780 | 0.6919 |
| MobileNetV3_small scratch (p20) | 2024 | 0.6477 | 0.6416 | 0.6898 |
| MobileNetV3_small scratch (p20) | 3407 | 0.6798 | 0.6804 | 0.7161 |

## Per-class F1

| label | class | f1_mean | f1_std | recall_mean |
| --- | --- | --- | --- | --- |
| Custom CNN | cardboard | 0.8862 | 0.0133 | 0.8738 |
| Custom CNN | glass | 0.8817 | 0.0112 | 0.9014 |
| Custom CNN | metal | 0.8661 | 0.0066 | 0.8522 |
| Custom CNN | paper | 0.9138 | 0.0138 | 0.9049 |
| Custom CNN | plastic | 0.8870 | 0.0129 | 0.9005 |
| MobileNetV3_small pretrained | cardboard | 0.8836 | 0.0214 | 0.8754 |
| MobileNetV3_small pretrained | glass | 0.8786 | 0.0067 | 0.8797 |
| MobileNetV3_small pretrained | metal | 0.8733 | 0.0134 | 0.8500 |
| MobileNetV3_small pretrained | paper | 0.9124 | 0.0133 | 0.9585 |
| MobileNetV3_small pretrained | plastic | 0.9024 | 0.0131 | 0.8919 |
| MobileNetV3_small scratch (p20) | cardboard | 0.6805 | 0.0340 | 0.6769 |
| MobileNetV3_small scratch (p20) | glass | 0.6805 | 0.0309 | 0.6841 |
| MobileNetV3_small scratch (p20) | metal | 0.6231 | 0.0355 | 0.6089 |
| MobileNetV3_small scratch (p20) | paper | 0.7164 | 0.0336 | 0.7293 |
| MobileNetV3_small scratch (p20) | plastic | 0.6666 | 0.0263 | 0.6711 |

## Lỗi phổ biến

| label | true_label | pred_label | count |
| --- | --- | --- | --- |
| Custom CNN | metal | plastic | 60.0000 |
| Custom CNN | plastic | metal | 44.0000 |
| Custom CNN | metal | glass | 40.0000 |
| Custom CNN | plastic | glass | 37.0000 |
| Custom CNN | cardboard | paper | 36.0000 |
| Custom CNN | [realwaste] | error_rate | 0.1011 |
| Custom CNN | [trashnet] | error_rate | 0.1289 |
| MobileNetV3_small pretrained | cardboard | paper | 64.0000 |
| MobileNetV3_small pretrained | glass | plastic | 47.0000 |
| MobileNetV3_small pretrained | plastic | metal | 47.0000 |
| MobileNetV3_small pretrained | metal | glass | 43.0000 |
| MobileNetV3_small pretrained | plastic | glass | 40.0000 |
| MobileNetV3_small pretrained | [realwaste] | error_rate | 0.1050 |
| MobileNetV3_small pretrained | [trashnet] | error_rate | 0.1133 |
| MobileNetV3_small scratch (p20) | metal | plastic | 151.0000 |
| MobileNetV3_small scratch (p20) | plastic | metal | 116.0000 |
| MobileNetV3_small scratch (p20) | glass | plastic | 94.0000 |
| MobileNetV3_small scratch (p20) | plastic | glass | 87.0000 |
| MobileNetV3_small scratch (p20) | cardboard | paper | 83.0000 |
| MobileNetV3_small scratch (p20) | [realwaste] | error_rate | 0.3274 |
| MobileNetV3_small scratch (p20) | [trashnet] | error_rate | 0.3283 |

## Hình

- `val_macro_f1_curves.png`
- `confusion_matrices.png`
- `test_macro_f1.png`

