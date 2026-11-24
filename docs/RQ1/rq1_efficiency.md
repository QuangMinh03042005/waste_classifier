# RQ1 – Efficiency

Sinh bởi `scripts/measure_efficiency.py`; FLOPs tính cho một ảnh 224×224.

| Run | Model | Params | GFLOPs | Latency B1 (ms) | Latency B32 (ms) | Images/s | VRAM (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| rq1_customcnn_s123 | custom_cnn | 1,174,501 | 5.173 | 29.10 | 1143.94 | 28.0 |  |
| rq1_customcnn_s2024 | custom_cnn | 1,174,501 | 5.173 | 34.07 | 1129.96 | 28.3 |  |
| rq1_customcnn_s3407 | custom_cnn | 1,174,501 | 5.173 | 30.51 | 1143.30 | 28.0 |  |
| rq1_customcnn_s42 | custom_cnn | 1,174,501 | 5.173 | 32.40 | 1126.44 | 28.4 |  |
| rq1_customcnn_s7 | custom_cnn | 1,174,501 | 5.173 | 30.35 | 1095.10 | 29.2 |  |
| rq1_mobilenetv3s_full_s123 | mobilenet_v3_small | 1,522,981 | 0.111 | 11.03 | 127.62 | 250.7 |  |
| rq1_mobilenetv3s_full_s2024 | mobilenet_v3_small | 1,522,981 | 0.111 | 11.31 | 129.60 | 246.9 |  |
| rq1_mobilenetv3s_full_s3407 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.69 | 128.21 | 249.6 |  |
| rq1_mobilenetv3s_full_s42 | mobilenet_v3_small | 1,522,981 | 0.111 | 11.67 | 127.40 | 251.2 |  |
| rq1_mobilenetv3s_full_s7 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.84 | 133.49 | 239.7 |  |
| rq1_mobilenetv3s_scratch_p20_s123 | mobilenet_v3_small | 1,522,981 | 0.111 | 11.71 | 139.27 | 229.8 |  |
| rq1_mobilenetv3s_scratch_p20_s2024 | mobilenet_v3_small | 1,522,981 | 0.111 | 14.99 | 135.76 | 235.7 |  |
| rq1_mobilenetv3s_scratch_p20_s3407 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.99 | 144.11 | 222.0 |  |
| rq1_mobilenetv3s_scratch_p20_s42 | mobilenet_v3_small | 1,522,981 | 0.111 | 14.33 | 135.48 | 236.2 |  |
| rq1_mobilenetv3s_scratch_p20_s7 | mobilenet_v3_small | 1,522,981 | 0.111 | 13.72 | 135.79 | 235.7 |  |
