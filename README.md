# Waste Classification Using CNN

**Nghiên cứu ứng dụng Convolutional Neural Network (CNN) trong phân loại rác thải qua hình ảnh và đánh giá khả năng tổng quát hóa của mô hình.**

## 1. Giới thiệu đề tài

Phân loại rác thải là một bước quan trọng trong quy trình quản lý và tái chế chất thải. Tuy nhiên, việc phân loại thủ công thường tốn thời gian, phụ thuộc vào con người và khó mở rộng khi khối lượng rác thải lớn.

Với sự phát triển của Deep Learning và Computer Vision, các mô hình Convolutional Neural Network (CNN) có khả năng học đặc trưng trực tiếp từ hình ảnh và thực hiện phân loại tự động.

Dự án này nghiên cứu việc ứng dụng CNN vào bài toán phân loại rác thải thông qua hình ảnh. Bên cạnh việc xây dựng mô hình, nhóm tập trung đánh giá ảnh hưởng của kiến trúc mạng, kỹ thuật Data Augmentation và khả năng tổng quát hóa khi mô hình được huấn luyện, kiểm thử trên các nguồn dữ liệu khác nhau.

Dự án được triển khai bằng **Python và PyTorch**, đồng thời cung cấp giao diện demo để thử nghiệm mô hình sau khi huấn luyện.

## 2. Mục tiêu

- Xây dựng mô hình CNN phục vụ bài toán phân loại rác thải qua hình ảnh.
- So sánh hiệu quả giữa CNN tự xây dựng và mô hình sử dụng Transfer Learning.
- Đánh giá ảnh hưởng của Data Augmentation và Class Weighting đến hiệu quả phân loại.
- Nghiên cứu khả năng tổng quát hóa của mô hình trên các nguồn dữ liệu khác nhau.
- Thực hiện các thí nghiệm có kiểm soát, ghi nhận kết quả và phân tích sai số.
- Xây dựng ứng dụng demo cho phép người dùng tải ảnh và nhận kết quả dự đoán.

## 3. Research Questions

Dự án tập trung vào ba câu hỏi nghiên cứu chính.

### RQ1 – Model Architecture

**Transfer Learning với MobileNetV3 có cải thiện hiệu quả phân loại rác thải so với CNN tự xây dựng không?**

Nội dung thực hiện:

- Xây dựng mô hình Custom CNN làm baseline.
- Sử dụng MobileNetV3 pretrained với ImageNet.
- So sánh hiệu quả hai mô hình trên cùng điều kiện thực nghiệm.
- Phân tích hiệu quả phân loại và chi phí tính toán.

### RQ2 – Data Augmentation

**Data Augmentation có cải thiện hiệu quả phân loại rác thải bằng MobileNetV3 không?**

Nội dung thực hiện:

- Xây dựng cấu hình Full Model.
- Thực hiện Ablation Study để đánh giá ảnh hưởng của từng thành phần.
- So sánh mô hình có và không có Data Augmentation.
- Đánh giá ảnh hưởng của Class Weighting.
- Phân tích khả năng tổng quát hóa và hiện tượng Overfitting.

### RQ3 – Generalization

**Việc kết hợp dữ liệu từ nhiều nguồn có cải thiện khả năng tổng quát hóa của mô hình CNN trên các nguồn dữ liệu khác nhau không?**

Nội dung thực hiện:

- Nghiên cứu sự khác biệt giữa các nguồn dữ liệu.
- Thực hiện Cross-dataset Evaluation.
- Huấn luyện trên một nguồn dữ liệu và kiểm thử trên nguồn khác.
- Đánh giá hiệu quả khi kết hợp nhiều nguồn dữ liệu.
- Phân tích ảnh hưởng của sự khác biệt về bối cảnh và phân phối dữ liệu.

## 4. Dataset

Các dataset dự kiến sử dụng:

| Dataset   | Số lượng ảnh | Đặc điểm                                     |
| --------- | -----------: | -------------------------------------------- |
| RealWaste |        4.752 | Hình ảnh rác thải trong môi trường thực tế   |
| TrashNet  |        2.527 | Hình ảnh rác thải với nền tương đối đơn giản |

Nguồn dữ liệu:

- [RealWaste – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste)
- [TrashNet – GitHub Repository](https://github.com/garythung/trashnet)

### Các lớp phân loại dự kiến

Nhóm dự kiến sử dụng 5 lớp chung giữa hai dataset:

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic

Danh sách lớp chính thức sẽ được xác nhận sau khi kiểm tra dữ liệu, mapping nhãn và loại bỏ những trường hợp không tương thích.

### Quy trình xử lý dữ liệu

- Kiểm tra chất lượng và cấu trúc dataset.
- Chuẩn hóa tên và mapping nhãn.
- Kiểm tra ảnh trùng lặp.
- Resize và normalize hình ảnh.
- Chia dữ liệu thành Train, Validation và Test.
- Lưu lại thông tin split để đảm bảo khả năng tái lập thí nghiệm.

## 5. Phương pháp thực hiện

### 5.1. Mô hình

- Custom CNN: mô hình baseline do nhóm xây dựng.
- MobileNetV3: mô hình pretrained sử dụng Transfer Learning.

### 5.2. Kỹ thuật

- Convolutional Neural Network.
- Transfer Learning.
- Data Augmentation.
- Class Weighting.
- Cross-dataset Evaluation.
- Ablation Study.

### 5.3. Evaluation Metrics

Các chỉ số đánh giá dự kiến:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

Ngoài ra, nhóm có thể ghi nhận thời gian huấn luyện, thời gian suy luận và số lượng tham số của mô hình.

### 5.4. Experimental Protocol

Để đảm bảo tính khách quan và khả năng tái lập:

- Sử dụng cùng quy trình đánh giá cho các thí nghiệm cần so sánh.
- Ghi lại cấu hình, random seed và phiên bản dữ liệu.
- Thực hiện nhiều lần chạy với các seed khác nhau cho những thí nghiệm quan trọng.
- Báo cáo kết quả trung bình và độ lệch chuẩn khi có nhiều lần chạy.
- Chỉ sử dụng Test Set cho đánh giá cuối cùng.
- Không ghi đè kết quả của những lần chạy trước.

## 6. Phân công công việc

Nhóm gồm 3 thành viên. Mỗi thành viên phụ trách một Research Question, đồng thời sử dụng chung hệ thống xử lý dữ liệu và các module kỹ thuật nền tảng.

| Thành viên                                 | Research Question        | Phạm vi phụ trách                                                             |
| ------------------------------------------ | ------------------------ | ----------------------------------------------------------------------------- |
| Thành viên 1: Đinh Quang Minh (3123580024) | RQ1 – Model Architecture | Custom CNN, MobileNetV3, so sánh kiến trúc và phân tích kết quả               |
| Thành viên 2: Dương Tấn Cường (3123580004) | RQ2 – Data Augmentation  | Augmentation, Class Weighting, Ablation Study và phân tích kết quả            |
| Thành viên 3: Lê Thành Duy (3123580007)    | RQ3 – Generalization     | Cross-dataset Evaluation, kết hợp dữ liệu và phân tích khả năng tổng quát hóa |

### Công việc chung

| Công việc                              | Trách nhiệm |
| -------------------------------------- | ----------- |
| Xây dựng Data Pipeline                 | Cả nhóm     |
| Thống nhất preprocessing và data split | Cả nhóm     |
| Xây dựng Training Engine và Evaluation | Cả nhóm     |
| Kiểm thử các module dùng chung         | Cả nhóm     |
| Tích hợp kết quả nghiên cứu            | Cả nhóm     |
| Hoàn thiện báo cáo và thuyết trình     | Cả nhóm     |
| Xây dựng ứng dụng demo                 | Cả nhóm     |

## 7. Cấu trúc Repository

```text
waste-classification-cnn/
│
├── configs/                 # Cấu hình thí nghiệm
│   ├── rq1/
│   ├── rq2/
│   └── rq3/
│
├── data/                    # Dữ liệu (không commit dataset)
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/                     # Các module dùng chung
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
│
├── research/                # Triển khai từng RQ
│   ├── rq1/
│   ├── rq2/
│   └── rq3/
│
├── experiments/             # Kết quả thực nghiệm
│   ├── rq1/
│   ├── rq2/
│   └── rq3/
│
├── app/                     # Ứng dụng demo
│
├── requirements.txt
└── README.md
```

## 8. Git Workflow

Dự án sử dụng một repository chung và Git Branch để phân chia công việc.

| Branch                         | Mục đích                         |
| ------------------------------ | -------------------------------- |
| `main`                         | Phiên bản code ổn định           |
| `feature/shared-pipeline`      | Phát triển các module dùng chung |
| `feature/rq1-model-comparison` | Triển khai RQ1                   |
| `feature/rq2-ablation`         | Triển khai RQ2                   |
| `feature/rq3-generalization`   | Triển khai RQ3                   |

### Quy tắc làm việc

1. Không push trực tiếp vào `main`.
2. Mỗi thành viên làm việc trên branch phụ trách.
3. Mọi thay đổi cần tích hợp vào `main` thông qua Pull Request.
4. Thay đổi code dùng chung phải được thông báo và review.
5. Không commit dataset, checkpoint lớn và file cache lên Git.
6. Các thí nghiệm phải lưu cấu hình và kết quả riêng biệt.
7. Không chỉnh sửa hoặc ghi đè kết quả thực nghiệm cũ.

## 9. Công nghệ sử dụng

| Công nghệ            | Mục đích                                     |
| -------------------- | -------------------------------------------- |
| Python               | Ngôn ngữ lập trình                           |
| PyTorch              | Xây dựng và huấn luyện mô hình Deep Learning |
| Torchvision          | Model pretrained và xử lý ảnh                |
| NumPy                | Xử lý dữ liệu số                             |
| Pandas               | Quản lý kết quả thực nghiệm                  |
| Matplotlib / Seaborn | Trực quan hóa kết quả                        |
| Scikit-learn         | Metrics và đánh giá mô hình                  |
| Streamlit            | Xây dựng ứng dụng demo                       |
| Git / GitHub         | Quản lý mã nguồn và cộng tác                 |

## 10. Expected Deliverables

- Source code hoàn chỉnh.
- Các mô hình CNN đã huấn luyện.
- Bộ cấu hình và kết quả thực nghiệm.
- Bảng so sánh và phân tích kết quả của ba RQ.
- Confusion Matrix và các biểu đồ đánh giá.
- Ứng dụng demo phân loại rác thải.
- Báo cáo nghiên cứu và slide thuyết trình.
- README hướng dẫn cài đặt và chạy project.

## 11. Cài đặt và tải dữ liệu

Yêu cầu: Python 3.12 và [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                             # cài môi trường
uv run python -m src.data.download --dataset all    # tải RealWaste + TrashNet
uv run python scripts/make_splits.py                # tạo data/splits/v1
uv run pytest                                       # chạy test
uv run ruff check .                                 # lint
```

Dữ liệu được tải vào `data/raw/` và không commit. Chi tiết nguồn, giấy phép, cấu trúc
thư mục và các tùy chọn tải: [docs/datasets.md](docs/datasets.md). Mapping nhãn và
split: [docs/label_mapping.md](docs/label_mapping.md),
[docs/splits.md](docs/splits.md).

Training trên GPU: xem [docs/colab.md](docs/colab.md). Kế hoạch thí nghiệm:
[docs/experiment_plans.md](docs/experiment_plans.md). Log và schema kết quả:
[docs/logging.md](docs/logging.md).

## 12. Project Status

**Status:** In Progress

Các công việc tiếp theo:

- [ ] Kiểm tra và xác nhận dataset.
- [ ] Hoàn thiện Data Pipeline.
- [ ] Thống nhất preprocessing và data split.
- [ ] Xây dựng baseline.
- [ ] Triển khai các thí nghiệm RQ1, RQ2 và RQ3.
- [ ] Phân tích và tổng hợp kết quả.
- [ ] Xây dựng ứng dụng demo.
- [ ] Hoàn thiện báo cáo và thuyết trình.

---

**Project:** Waste Classification Using CNN  
**Course:** Deep Learning  
**Team size:** 3 members  
**Language:** Python  
**Framework:** PyTorch
