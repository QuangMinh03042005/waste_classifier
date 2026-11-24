# RQ1 – External validation trên GD

External test chỉ dùng để đánh giá sau khi khóa mô hình; không dùng để chọn model.

## Tổng hợp

| experiment | seeds | external_macro_f1_mean | external_macro_f1_std | external_accuracy_mean | in_domain_macro_f1_mean | drop_mean |
| --- | --- | --- | --- | --- | --- | --- |
| rq1_customcnn | 5 | 0.3694 | 0.0117 | 0.3679 | 0.8870 | 0.5175 |
| rq1_mobilenetv3s_full | 5 | 0.5409 | 0.0097 | 0.5510 | 0.8901 | 0.3492 |
| rq1_mobilenetv3s_scratch_p20 | 5 | 0.2834 | 0.0230 | 0.2919 | 0.6734 | 0.3900 |

## Từng seed

| run_id | experiment | seed | in_domain_macro_f1 | external_macro_f1 | external_accuracy | drop |
| --- | --- | --- | --- | --- | --- | --- |
| rq1_customcnn_s123 | rq1_customcnn | 123 | 0.8763 | 0.3796 | 0.3747 | 0.4968 |
| rq1_customcnn_s2024 | rq1_customcnn | 2024 | 0.8901 | 0.3807 | 0.3795 | 0.5094 |
| rq1_customcnn_s3407 | rq1_customcnn | 3407 | 0.8899 | 0.3519 | 0.3494 | 0.5381 |
| rq1_customcnn_s42 | rq1_customcnn | 42 | 0.8906 | 0.3679 | 0.3654 | 0.5227 |
| rq1_customcnn_s7 | rq1_customcnn | 7 | 0.8880 | 0.3672 | 0.3706 | 0.5207 |
| rq1_mobilenetv3s_full_s123 | rq1_mobilenetv3s_full | 123 | 0.8793 | 0.5377 | 0.5516 | 0.3416 |
| rq1_mobilenetv3s_full_s2024 | rq1_mobilenetv3s_full | 2024 | 0.9089 | 0.5296 | 0.5381 | 0.3793 |
| rq1_mobilenetv3s_full_s3407 | rq1_mobilenetv3s_full | 3407 | 0.8888 | 0.5411 | 0.5491 | 0.3477 |
| rq1_mobilenetv3s_full_s42 | rq1_mobilenetv3s_full | 42 | 0.8833 | 0.5397 | 0.5493 | 0.3436 |
| rq1_mobilenetv3s_full_s7 | rq1_mobilenetv3s_full | 7 | 0.8900 | 0.5564 | 0.5668 | 0.3337 |
| rq1_mobilenetv3s_scratch_p20_s123 | rq1_mobilenetv3s_scratch_p20 | 123 | 0.6792 | 0.3177 | 0.3250 | 0.3615 |
| rq1_mobilenetv3s_scratch_p20_s2024 | rq1_mobilenetv3s_scratch_p20 | 2024 | 0.6477 | 0.2536 | 0.2613 | 0.3941 |
| rq1_mobilenetv3s_scratch_p20_s3407 | rq1_mobilenetv3s_scratch_p20 | 3407 | 0.6798 | 0.2835 | 0.2938 | 0.3963 |
| rq1_mobilenetv3s_scratch_p20_s42 | rq1_mobilenetv3s_scratch_p20 | 42 | 0.6804 | 0.2770 | 0.2817 | 0.4034 |
| rq1_mobilenetv3s_scratch_p20_s7 | rq1_mobilenetv3s_scratch_p20 | 7 | 0.6799 | 0.2853 | 0.2975 | 0.3946 |
