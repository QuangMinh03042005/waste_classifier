# Waste Classification Using CNN

**Nghiên cứu ứng dụng Convolutional Neural Network (CNN) trong phân loại rác thải qua hình ảnh và đánh giá khả năng tổng quát hóa của mô hình.**

## 1. Giới thiệu đề tài

Phân loại rác thải là một bước quan trọng trong quy trình quản lý và tái chế chất thải. Tuy nhiên, việc phân loại thủ công thường tốn thời gian, phụ thuộc vào con người và khó mở rộng khi khối lượng rác thải lớn.

Với sự phát triển của Deep Learning và Computer Vision, các mô hình Convolutional Neural Network (CNN) có khả năng học đặc trưng trực tiếp từ hình ảnh và thực hiện phân loại tự động.

Dự án này nghiên cứu việc ứng dụng CNN vào bài toán phân loại rác thải thông qua hình ảnh. Bên cạnh việc xây dựng mô hình, nhóm tập trung đánh giá ảnh hưởng của kiến trúc mạng, pretrained weights, Data Augmentation, Class Weighting và khả năng tổng quát hóa khi mô hình được huấn luyện, kiểm thử trên các nguồn dữ liệu khác nhau.

RQ1 được thiết kế với ba nhánh: Custom CNN, MobileNetV3_small huấn luyện từ đầu (scratch) và MobileNetV3_small pretrained ImageNet. Cách này giúp tách ảnh hưởng của **kiến trúc** khỏi ảnh hưởng của **transfer learning**. Ngoài test in-domain, RQ1 còn đánh giá thêm trên bộ GD như một external test.

Dự án được triển khai bằng **Python và PyTorch**, quản lý môi trường bằng `uv`, có logging/checkpoint đầy đủ và cung cấp giao diện demo Streamlit sau khi huấn luyện.

## 2. Mục tiêu

- Xây dựng mô hình CNN phục vụ bài toán phân loại rác thải qua hình ảnh.
- So sánh hiệu quả giữa CNN tự xây dựng và MobileNetV3_small pretrained.
- Tách ảnh hưởng của kiến trúc và pretrained weights bằng nhánh MobileNetV3_small scratch.
- Đánh giá ảnh hưởng của Data Augmentation và Class Weighting đến hiệu quả phân loại.
- Nghiên cứu khả năng tổng quát hóa trên các nguồn dữ liệu khác nhau.
- Đánh giá thêm trên external test GD cho RQ1.
- Thực hiện thí nghiệm có kiểm soát, ghi log, phân tích lỗi và chi phí tính toán.
- Xây dựng ứng dụng demo cho phép người dùng tải ảnh và nhận dự đoán.

## 3. Research Questions

### RQ1 – Model Architecture

**Transfer learning với MobileNetV3_small có cải thiện hiệu quả phân loại rác thải so với Custom CNN không, và phần cải thiện đến từ kiến trúc hay từ pretrained weights?**

Nội dung thực hiện:

- Custom CNN baseline huấn luyện từ đầu.
- MobileNetV3_small scratch — cùng kiến trúc nhưng không pretrained.
- MobileNetV3_small pretrained ImageNet.
- Cùng split v1, input 224×224, không augmentation, không class weighting.
- 5 seed: 42, 123, 2024, 7, 2025.
- Chọn epoch và cấu hình trên validation; test chỉ đánh giá một lần.
- Báo cáo accuracy, balanced accuracy, Macro-F1, per-class F1, confusion matrix.
- Báo cáo params, GFLOPs, latency, VRAM.
- External validation trên GD sau khi khóa mô hình.

Tài liệu chi tiết: [docs/RQ1/rq1_overview.md](docs/RQ1/rq1_overview.md).

### RQ2 – Data Augmentation & Class Weighting

**Data Augmentation và Class Weighting ảnh hưởng thế nào đến hiệu quả phân loại rác thải?**

Nội dung thực hiện:

- Full model: augmentation + class weighting.
- Ablation: bỏ augmentation.
- Ablation: bỏ class weighting.
- Cùng split, cùng MobileNetV3_small, cùng ngân sách huấn luyện.
- Phân tích per-class recall, đặc biệt ở lớp thiểu số.
- Phân tích overfitting qua learning curves.

### RQ3 – Generalization

**Việc kết hợp dữ liệu từ nhiều nguồn có cải thiện khả năng tổng quát hóa của mô hình CNN trên các nguồn dữ liệu khác nhau không?**

Nội dung thực hiện:

- Nghiên cứu khác biệt giữa RealWaste và TrashNet.
- Cross-dataset evaluation: RealWaste → TrashNet và TrashNet → RealWaste.
- Huấn luyện trên dữ liệu kết hợp, kiểm thử riêng từng nguồn.
- Phân tích ảnh hưởng của bối cảnh, phân phối lớp và domain shift.

Ghi chú: GD hiện chỉ dùng làm **external test cho RQ1**, không tham gia train RQ3.

## 4. Dataset

### 4.1. Nguồn dữ liệu

| Dataset | Vai trò | Tổng ảnh gốc | Ảnh dùng cho 5 lớp | License |
|---|---|---:|---:|---|
| RealWaste | Train / validation / test | 4.752 | 3.092 | CC BY 4.0 |
| TrashNet | Train / validation / test | 2.527 | 2.390 | MIT |
| GD (Garbage Dataset) | **External test RQ1** | 13.348 | 4.806 sau dedup | MIT |

Nguồn:

- [RealWaste – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste)
- [TrashNet – GitHub Repository](https://github.com/garythung/trashnet)
- [GD – Kaggle](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

### 4.2. Các lớp phân loại

5 lớp chung đã xác nhận:

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic

Các lớp bị loại:

- RealWaste: Food Organics, Miscellaneous Trash, Textile Trash, Vegetation.
- TrashNet: trash.
- GD: biological, battery, trash, shoes, clothes.

Tổng 1.797 ảnh bị loại khỏi RealWaste + TrashNet; GD chỉ dùng external test.

### 4.3. Phân bố dữ liệu

| Lớp | RealWaste | TrashNet | Tổng |
|---|---:|---:|---:|
| Cardboard | 461 | 403 | 864 |
| Glass | 420 | 501 | 921 |
| Metal | 790 | 410 | 1.200 |
| Paper | 500 | 594 | 1.094 |
| Plastic | 921 | 482 | 1.403 |
| **Tổng** | **3.092** | **2.390** | **5.482** |

Split v1:

| Split | Số ảnh |
|---|---:|
| Train | 3.833 |
| Validation | 826 |
| Test | 823 |

- Tỷ lệ 70/15/15, seed 42.
- Stratified theo `(dataset, class)`.
- Chống leakage bằng pHash; 35 ảnh trùng được gom group, không tách giữa các split.

### 4.4. GD external test

- 7.010 ảnh GD mapping về 5 lớp.
- Loại 2.200 ảnh trùng với split v1.
- Loại 4 ảnh trùng nội bộ.
- Còn 4.806 ảnh external test.
- Không dùng GD để train hay chọn mô hình.

Chi tiết: [docs/RQ1/external_validation.md](docs/RQ1/external_validation.md).

### 4.5. Quy trình xử lý dữ liệu

- Kiểm tra cấu trúc, số lượng, license.
- Mapping nhãn về 5 lớp.
- Kiểm tra ảnh lỗi, ảnh mờ, ảnh quá sáng.
- Tính pHash, gom ảnh trùng.
- Chia train/validation/test theo group.
- Resize và normalize 224×224.
- Lưu split CSV để tái lập.
- EDA: [docs/EDA/eda_report.md](docs/EDA/eda_report.md).

## 5. Phương pháp thực hiện

### 5.1. Mô hình

- **Custom CNN**: baseline từ đầu, channels 32-64-128-256, convs/stage 1-2-2-2, dropout 0.3.
- **MobileNetV3_small scratch**: kiến trúc MobileNetV3_small, không pretrained.
- **MobileNetV3_small pretrained**: ImageNet `IMAGENET1K_V1`, fine-tune cho 5 lớp.

### 5.2. Kỹ thuật

- Convolutional Neural Network.
- Transfer learning, freeze/unfreeze.
- Data Augmentation.
- Class Weighting.
- Cross-dataset Evaluation.
- Ablation Study.
- Grad-CAM.
- Đo FLOPs, latency, VRAM.

### 5.3. Evaluation Metrics

- Accuracy.
- Balanced accuracy.
- Precision, Recall, Macro-F1.
- Per-class F1 và recall.
- Confusion Matrix.
- Params, GFLOPs, latency, throughput, VRAM.
- Mean ± std trên nhiều seed.

### 5.4. Experimental Protocol

- Cùng split, cùng preprocessing, cùng ngân sách cho các thí nghiệm so sánh.
- Ghi config, seed, git commit, environment.
- RQ1 dùng 5 seed; RQ2/RQ3 dùng 3 seed.
- Chọn epoch theo validation Macro-F1.
- Chỉ dùng test set cho đánh giá cuối.
- Không dùng test/external để chọn epoch, threshold hay cấu hình.
- Không ghi đè kết quả cũ.
- Plan thí nghiệm được commit trước khi chạy.

## 6. Phân công công việc

Nhóm gồm 3 thành viên. Mỗi thành viên phụ trách một Research Question, sử dụng chung data pipeline, training engine và logging.

| Thành viên | Research Question | Phạm vi phụ trách |
|---|---|---|
| Đinh Quang Minh (3123580024) | RQ1 – Model Architecture | Custom CNN, MobileNetV3_small scratch/pretrained, external GD, phân tích kết quả |
| Dương Tấn Cường (3123580004) | RQ2 – Data Augmentation & Class Weighting | Augmentation, class weighting, ablation study |
| Lê Thành Duy (3123580007) | RQ3 – Generalization | Cross-dataset evaluation, kết hợp nguồn, phân tích domain shift |

### Công việc chung

| Công việc | Trách nhiệm |
|---|---|
| Data pipeline, split, EDA | Cả nhóm |
| Training engine, logging, checkpoint | Cả nhóm |
| Plan thí nghiệm và ma trận RQ | Cả nhóm |
| Kiểm thử module dùng chung | Cả nhóm |
| Tích hợp kết quả, phân tích | Cả nhóm |
| Demo Streamlit | Cả nhóm |
| Báo cáo và thuyết trình | Cả nhóm |

## 7. Cấu trúc Repository

```text
waste-classifier/
│
├── configs/                 # Cấu hình thí nghiệm
│   └── rq1/                 # debug, selection, final
│
├── data/                    # Không commit dataset
│   ├── raw/                 # RealWaste, TrashNet
│   ├── external/            # GD external test
│   └── splits/              # v1, rq3, external
│
├── src/                     # Module dùng chung
│   ├── data/                # download, split, dataset, transform, EDA, external
│   ├── models/              # Custom CNN, MobileNet, factory
│   ├── training/            # runner, metrics, checkpoint, evaluator, Grad-CAM
│   └── utils/               # paths, config, logging, plan, export
│
├── research/                # Entrypoint theo RQ
│   ├── rq1/train.py
│   └── evaluate.py
│
├── plans/                   # Plan pre-registration
│   ├── rq1/
│   ├── rq2/
│   └── rq3/
│
├── experiments/             # Kết quả chạy (không commit)
│   └── rq1/
│
├── notebooks/
│   ├── shared/              # EDA
│   └── rq1/                 # Phân tích RQ1
│
├── scripts/                 # make_splits, run_batch, export, analysis
├── tests/                   # pytest
├── docs/                    # Tài liệu, EDA, RQ1
├── app/                     # Streamlit demo
│
├── pyproject.toml
├── uv.lock
├── requirements-colab.txt
├── AGENTS.md
└── README.md
```

## 8. Git Workflow

| Branch | Mục đích |
|---|---|
| `main` | Phiên bản ổn định |
| `feature/shared-pipeline` | Module dùng chung |
| `feature/rq1-model-comparison` | Triển khai RQ1 |
| `feature/rq2-ablation` | Triển khai RQ2 |
| `feature/rq3-generalization` | Triển khai RQ3 |

### Quy tắc làm việc

1. Không push trực tiếp vào `main`.
2. Mỗi thành viên làm việc trên branch phụ trách.
3. Thay đổi tích hợp vào `main` qua Pull Request.
4. Thay đổi code dùng chung phải được thông báo và review.
5. Không commit dataset, checkpoint lớn, cache.
6. Mỗi thí nghiệm lưu config và kết quả riêng.
7. Không sửa hoặc ghi đè kết quả cũ.
8. Commit message tiếng Việt dạng `<scope>: <mô tả>`; scope: `feat`, `fix`, `data`, `docs`, `test`, `chore`.

## 9. Công nghệ sử dụng

| Công nghệ | Mục đích |
|---|---|
| Python 3.12 | Ngôn ngữ lập trình |
| PyTorch | Huấn luyện và suy luận |
| Torchvision | MobileNetV3, transform, pretrained weights |
| NumPy | Xử lý số, pHash |
| Pandas | Tổng hợp kết quả, EDA |
| Scikit-learn | Metrics, confusion matrix |
| Matplotlib | Learning curves, confusion matrix, Grad-CAM |
| uv | Quản lý môi trường và dependency |
| pytest, ruff | Kiểm thử, lint/format |
| Jupyter | EDA notebook |
| Streamlit | Demo |
| Google Colab / GPU riêng | Huấn luyện |
| Git / GitHub | Quản lý mã nguồn |

## 10. Expected Deliverables

- Source code hoàn chỉnh.
- Custom CNN, MobileNetV3_small scratch và pretrained đã huấn luyện.
- Config, plan và kết quả thực nghiệm.
- Bảng so sánh và phân tích 3 RQ.
- Confusion matrix, learning curves, Grad-CAM.
- External validation trên GD.
- Báo cáo EDA và nhận xét dữ liệu.
- Ứng dụng demo Streamlit.
- Báo cáo nghiên cứu và slide.
- README hướng dẫn cài đặt và tái lập.

## 11. Cài đặt và tải dữ liệu

Yêu cầu: Python 3.12 và [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                             # cài môi trường
uv run python -m src.data.download --dataset all    # RealWaste + TrashNet
uv run python scripts/make_splits.py                # tạo data/splits/v1
uv run python scripts/run_eda.py                    # EDA
uv run python scripts/prepare_external_gd.py        # external GD cho RQ1
uv run pytest                                       # test
uv run ruff check .                                 # lint
```

Dữ liệu vào `data/raw/`, GD external vào `data/external/`; không commit. Chi tiết:

- Dataset: [docs/datasets.md](docs/datasets.md)
- Mapping nhãn: [docs/label_mapping.md](docs/label_mapping.md)
- Split: [docs/splits.md](docs/splits.md)
- Data pipeline: [docs/data_pipeline.md](docs/data_pipeline.md)
- EDA: [docs/EDA/eda_report.md](docs/EDA/eda_report.md), [notebooks/shared/eda.ipynb](notebooks/shared/eda.ipynb)
- External validation RQ1: [docs/RQ1/external_validation.md](docs/RQ1/external_validation.md)
- Training: [docs/training.md](docs/training.md), [docs/models.md](docs/models.md)
- RQ1: [docs/RQ1/rq1_overview.md](docs/RQ1/rq1_overview.md)

Training trên GPU: xem [docs/colab.md](docs/colab.md). Kế hoạch thí nghiệm:
[docs/experiment_plans.md](docs/experiment_plans.md). Logging:
[docs/logging.md](docs/logging.md).

## 12. Project Status

**Status:** In Progress

Đã hoàn thành:

- [x] Xác nhận dataset, mapping 5 lớp.
- [x] Data pipeline, split v1, kiểm tra leakage.
- [x] EDA và báo cáo nhận xét dữ liệu.
- [x] Custom CNN, MobileNetV3_small scratch/pretrained.
- [x] Training engine, logging, checkpoint, resume.
- [x] Chọn và debug cấu hình Custom CNN trên validation.
- [x] External test GD đã chuẩn bị (4.806 ảnh sau dedup).

Đang thực hiện:

- [ ] Xác nhận MobileNetV3_small ở 50 epoch.
- [ ] RQ1 final: 3 arm × 5 seed.
- [ ] External validation GD trên checkpoint final RQ1.

Chưa bắt đầu:

- [ ] RQ2: full, no-augmentation, no-class-weighting.
- [ ] RQ3: cross-dataset và pooled.
- [ ] Phân tích tổng hợp, hình, kết luận.
- [ ] Demo Streamlit.
- [ ] Báo cáo và slide.

---

**Project:** Waste Classification Using CNN  
**Course:** Deep Learning  
**Team size:** 3 members  
**Language:** Python  
**Framework:** PyTorch
