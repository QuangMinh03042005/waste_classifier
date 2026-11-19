# RQ1 – Efficiency

Sinh bởi `scripts/measure_efficiency.py`; FLOPs tính cho một ảnh 224×224.

| Run | Model | Params | GFLOPs | Latency B1 (ms) | Latency B32 (ms) | Images/s | VRAM (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| rq1_customcnn_d0_s42 | custom_cnn | 1,174,501 | 5.173 | 29.97 | 1080.44 | 29.6 |  |
| rq1_customcnn_r1_s42 | custom_cnn | 1,218,405 | 5.327 | 36.07 | 1304.94 | 24.5 |  |
| rq1_mobilenet_m1_full_ft_s42 | mobilenet_v3_small | 1,522,981 | 0.111 | 10.80 | 126.02 | 253.9 |  |
