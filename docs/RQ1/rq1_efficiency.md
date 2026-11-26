# RQ1 – Efficiency

Sinh bởi `scripts/measure_efficiency.py`; FLOPs tính cho một ảnh 224×224.

| Run | Model | Params | GFLOPs | Latency B1 (ms) | Latency B32 (ms) | Images/s | VRAM (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| rq1_customcnn_p20_s123 | custom_cnn | 1,174,501 | 5.173 | 26.22 | 1039.72 | 30.8 |  |
| rq1_customcnn_p20_s2024 | custom_cnn | 1,174,501 | 5.173 | 27.71 | 1066.77 | 30.0 |  |
| rq1_customcnn_p20_s3407 | custom_cnn | 1,174,501 | 5.173 | 27.67 | 1067.54 | 30.0 |  |
| rq1_customcnn_p20_s42 | custom_cnn | 1,174,501 | 5.173 | 29.29 | 1057.61 | 30.3 |  |
| rq1_customcnn_p20_s7 | custom_cnn | 1,174,501 | 5.173 | 28.03 | 1080.61 | 29.6 |  |
| rq1_mobilenetv3s_full_p20_s123 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.67 | 118.97 | 269.0 |  |
| rq1_mobilenetv3s_full_p20_s2024 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.82 | 117.89 | 271.4 |  |
| rq1_mobilenetv3s_full_p20_s3407 | mobilenet_v3_small | 1,522,981 | 0.111 | 9.12 | 117.61 | 272.1 |  |
| rq1_mobilenetv3s_full_p20_s42 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.91 | 115.90 | 276.1 |  |
| rq1_mobilenetv3s_full_p20_s7 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.17 | 117.27 | 272.9 |  |
| rq1_mobilenetv3s_scratch_p20_s123 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.02 | 116.98 | 273.6 |  |
| rq1_mobilenetv3s_scratch_p20_s2024 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.13 | 117.48 | 272.4 |  |
| rq1_mobilenetv3s_scratch_p20_s3407 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.40 | 117.05 | 273.4 |  |
| rq1_mobilenetv3s_scratch_p20_s42 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.09 | 117.40 | 272.6 |  |
| rq1_mobilenetv3s_scratch_p20_s7 | mobilenet_v3_small | 1,522,981 | 0.111 | 9.54 | 117.32 | 272.8 |  |
