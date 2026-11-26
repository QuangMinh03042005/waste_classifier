# Kết quả RQ1

Sinh tự động bởi `scripts/analyze_rq1.py`; test được đánh giá một lần cho mỗi phiên bản. Cả ba arm đều đã tái sử dụng test sau khi điều chỉnh protocol, xem các plan tương ứng.

## Tổng hợp

| label | n_seeds | Macro-F1 mean | Macro-F1 std | Accuracy mean | Params |
| --- | --- | --- | --- | --- | --- |
| MobileNetV3_small pretrained (p20) | 5 | 0.8945 | 0.0081 | 0.8955 | 1522981.0000 |
| Custom CNN (p20) | 5 | 0.8863 | 0.0057 | 0.8860 | 1174501.0000 |
| MobileNetV3_small scratch (p20) | 5 | 0.6734 | 0.0144 | 0.6722 | 1522981.0000 |

## Theo seed

| label | seed | macro_f1 | accuracy | best_val_macro_f1 |
| --- | --- | --- | --- | --- |
| Custom CNN (p20) | 7 | 0.8880 | 0.8870 | 0.8664 |
| Custom CNN (p20) | 42 | 0.8872 | 0.8870 | 0.8746 |
| Custom CNN (p20) | 123 | 0.8763 | 0.8761 | 0.8607 |
| Custom CNN (p20) | 2024 | 0.8901 | 0.8906 | 0.8708 |
| Custom CNN (p20) | 3407 | 0.8899 | 0.8894 | 0.8728 |
| MobileNetV3_small pretrained (p20) | 7 | 0.8900 | 0.8906 | 0.8925 |
| MobileNetV3_small pretrained (p20) | 42 | 0.9024 | 0.9028 | 0.8958 |
| MobileNetV3_small pretrained (p20) | 123 | 0.8835 | 0.8846 | 0.8975 |
| MobileNetV3_small pretrained (p20) | 2024 | 0.9021 | 0.9040 | 0.9028 |
| MobileNetV3_small pretrained (p20) | 3407 | 0.8944 | 0.8955 | 0.9027 |
| MobileNetV3_small scratch (p20) | 7 | 0.6799 | 0.6780 | 0.6791 |
| MobileNetV3_small scratch (p20) | 42 | 0.6804 | 0.6829 | 0.7039 |
| MobileNetV3_small scratch (p20) | 123 | 0.6792 | 0.6780 | 0.6919 |
| MobileNetV3_small scratch (p20) | 2024 | 0.6477 | 0.6416 | 0.6898 |
| MobileNetV3_small scratch (p20) | 3407 | 0.6798 | 0.6804 | 0.7161 |

## Per-class F1

| label | class | f1_mean | f1_std | recall_mean |
| --- | --- | --- | --- | --- |
| Custom CNN (p20) | cardboard | 0.8873 | 0.0139 | 0.8769 |
| Custom CNN (p20) | glass | 0.8796 | 0.0131 | 0.9000 |
| Custom CNN (p20) | metal | 0.8667 | 0.0071 | 0.8489 |
| Custom CNN (p20) | paper | 0.9125 | 0.0128 | 0.9024 |
| Custom CNN (p20) | plastic | 0.8855 | 0.0127 | 0.9014 |
| MobileNetV3_small pretrained (p20) | cardboard | 0.8914 | 0.0131 | 0.8600 |
| MobileNetV3_small pretrained (p20) | glass | 0.8862 | 0.0081 | 0.8971 |
| MobileNetV3_small pretrained (p20) | metal | 0.8788 | 0.0121 | 0.8533 |
| MobileNetV3_small pretrained (p20) | paper | 0.9104 | 0.0120 | 0.9707 |
| MobileNetV3_small pretrained (p20) | plastic | 0.9057 | 0.0117 | 0.8938 |
| MobileNetV3_small scratch (p20) | cardboard | 0.6805 | 0.0340 | 0.6769 |
| MobileNetV3_small scratch (p20) | glass | 0.6805 | 0.0309 | 0.6841 |
| MobileNetV3_small scratch (p20) | metal | 0.6231 | 0.0355 | 0.6089 |
| MobileNetV3_small scratch (p20) | paper | 0.7164 | 0.0336 | 0.7293 |
| MobileNetV3_small scratch (p20) | plastic | 0.6666 | 0.0263 | 0.6711 |

## Lỗi phổ biến

| label | true_label | pred_label | count |
| --- | --- | --- | --- |
| Custom CNN (p20) | metal | plastic | 62.0000 |
| Custom CNN (p20) | plastic | metal | 41.0000 |
| Custom CNN (p20) | metal | glass | 40.0000 |
| Custom CNN (p20) | plastic | glass | 38.0000 |
| Custom CNN (p20) | cardboard | paper | 35.0000 |
| Custom CNN (p20) | [realwaste] | error_rate | 0.1015 |
| Custom CNN (p20) | [trashnet] | error_rate | 0.1300 |
| MobileNetV3_small pretrained (p20) | cardboard | paper | 72.0000 |
| MobileNetV3_small pretrained (p20) | metal | glass | 47.0000 |
| MobileNetV3_small pretrained (p20) | plastic | metal | 41.0000 |
| MobileNetV3_small pretrained (p20) | metal | plastic | 40.0000 |
| MobileNetV3_small pretrained (p20) | plastic | glass | 39.0000 |
| MobileNetV3_small pretrained (p20) | [realwaste] | error_rate | 0.1045 |
| MobileNetV3_small pretrained (p20) | [trashnet] | error_rate | 0.1044 |
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

