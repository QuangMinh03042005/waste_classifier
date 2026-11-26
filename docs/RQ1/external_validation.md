# RQ1 – External validation trên GD

External test chỉ dùng để đánh giá sau khi khóa mô hình; không dùng để chọn model.

## Tổng hợp

| experiment | seeds | external_macro_f1_mean | external_macro_f1_std | external_accuracy_mean | in_domain_macro_f1_mean | drop_mean |
| --- | --- | --- | --- | --- | --- | --- |
| rq1_customcnn_p20 | 5 | 0.3718 | 0.0124 | 0.3704 | 0.8863 | 0.5145 |
| rq1_mobilenetv3s_full_p20 | 5 | 0.5285 | 0.0218 | 0.5417 | 0.8945 | 0.3660 |
| rq1_mobilenetv3s_scratch_p20 | 5 | 0.2834 | 0.0230 | 0.2919 | 0.6734 | 0.3900 |

## Từng seed

| run_id | experiment | seed | in_domain_macro_f1 | external_macro_f1 | external_accuracy | drop |
| --- | --- | --- | --- | --- | --- | --- |
| rq1_customcnn_p20_s123 | rq1_customcnn_p20 | 123 | 0.8763 | 0.3796 | 0.3747 | 0.4968 |
| rq1_customcnn_p20_s2024 | rq1_customcnn_p20 | 2024 | 0.8901 | 0.3807 | 0.3795 | 0.5094 |
| rq1_customcnn_p20_s3407 | rq1_customcnn_p20 | 3407 | 0.8899 | 0.3519 | 0.3494 | 0.5381 |
| rq1_customcnn_p20_s42 | rq1_customcnn_p20 | 42 | 0.8872 | 0.3796 | 0.3777 | 0.5076 |
| rq1_customcnn_p20_s7 | rq1_customcnn_p20 | 7 | 0.8880 | 0.3672 | 0.3706 | 0.5207 |
| rq1_mobilenetv3s_full_p20_s123 | rq1_mobilenetv3s_full_p20 | 123 | 0.8835 | 0.4954 | 0.5110 | 0.3881 |
| rq1_mobilenetv3s_full_p20_s2024 | rq1_mobilenetv3s_full_p20 | 2024 | 0.9021 | 0.5290 | 0.5427 | 0.3731 |
| rq1_mobilenetv3s_full_p20_s3407 | rq1_mobilenetv3s_full_p20 | 3407 | 0.8944 | 0.5281 | 0.5416 | 0.3664 |
| rq1_mobilenetv3s_full_p20_s42 | rq1_mobilenetv3s_full_p20 | 42 | 0.9024 | 0.5336 | 0.5464 | 0.3688 |
| rq1_mobilenetv3s_full_p20_s7 | rq1_mobilenetv3s_full_p20 | 7 | 0.8900 | 0.5564 | 0.5668 | 0.3337 |
| rq1_mobilenetv3s_scratch_p20_s123 | rq1_mobilenetv3s_scratch_p20 | 123 | 0.6792 | 0.3177 | 0.3250 | 0.3615 |
| rq1_mobilenetv3s_scratch_p20_s2024 | rq1_mobilenetv3s_scratch_p20 | 2024 | 0.6477 | 0.2536 | 0.2613 | 0.3941 |
| rq1_mobilenetv3s_scratch_p20_s3407 | rq1_mobilenetv3s_scratch_p20 | 3407 | 0.6798 | 0.2835 | 0.2938 | 0.3963 |
| rq1_mobilenetv3s_scratch_p20_s42 | rq1_mobilenetv3s_scratch_p20 | 42 | 0.6804 | 0.2770 | 0.2817 | 0.4034 |
| rq1_mobilenetv3s_scratch_p20_s7 | rq1_mobilenetv3s_scratch_p20 | 7 | 0.6799 | 0.2853 | 0.2975 | 0.3946 |
