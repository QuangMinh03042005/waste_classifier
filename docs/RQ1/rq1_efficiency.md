# RQ1 – Efficiency

Sinh bởi `scripts/measure_efficiency.py`; FLOPs tính cho một ảnh 224×224.

| Run | Model | Params | GFLOPs | Latency B1 (ms) | Latency B32 (ms) | Images/s | VRAM (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| rq1_customcnn_s123 | custom_cnn | 1,174,501 | 5.173 | 37.12 | 1098.99 | 29.1 |  |
| rq1_customcnn_s2024 | custom_cnn | 1,174,501 | 5.173 | 28.60 | 1057.64 | 30.3 |  |
| rq1_customcnn_s3407 | custom_cnn | 1,174,501 | 5.173 | 28.50 | 1069.35 | 29.9 |  |
| rq1_customcnn_s42 | custom_cnn | 1,174,501 | 5.173 | 29.93 | 1089.50 | 29.4 |  |
| rq1_customcnn_s7 | custom_cnn | 1,174,501 | 5.173 | 27.67 | 1038.01 | 30.8 |  |
| rq1_mobilenetv3s_full_s123 | mobilenet_v3_small | 1,522,981 | 0.111 | 8.96 | 116.71 | 274.2 |  |
| rq1_mobilenetv3s_full_s2024 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.45 | 117.73 | 271.8 |  |
| rq1_mobilenetv3s_full_s3407 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.50 | 120.80 | 264.9 |  |
| rq1_mobilenetv3s_full_s42 | mobilenet_v3_small | 1,522,981 | 0.111 | 11.26 | 116.45 | 274.8 |  |
| rq1_mobilenetv3s_full_s7 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.68 | 115.23 | 277.7 |  |
