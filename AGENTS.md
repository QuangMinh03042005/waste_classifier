# AGENTS.md

Hướng dẫn cho AI agent và thành viên khi làm việc trong repo này. Đọc kỹ trước khi sửa code.

## 1. Bối cảnh dự án

- Bài toán: phân loại rác thải qua ảnh, 5 lớp chung (Cardboard, Glass, Metal, Paper, Plastic).
- Dataset: RealWaste (4.752 ảnh) + TrashNet (2.527 ảnh). Không commit dataset.
- Framework: Python + PyTorch. Demo: Streamlit.
- Ba RQ, mỗi RQ do một thành viên phụ trách:
  - RQ1 – Kiến trúc: Custom CNN (baseline) vs MobileNetV3_small (transfer learning).
  - RQ2 – Data Augmentation & Class Weighting: ablation study.
  - RQ3 – Generalization: cross-dataset, kết hợp nhiều nguồn dữ liệu.

## 2. Môi trường & lệnh

Dùng `uv`. `pyproject.toml` là nguồn sự thật duy nhất cho dependencies và tool config.

```bash
uv sync                          # cài môi trường
uv run pytest                    # chạy test
uv run ruff check .              # lint
uv run ruff format .             # format
uv run python -m research.rq1.train --config configs/rq1/<config>.yaml
uv run streamlit run app/main.py # demo
```

- Lệnh train/eval cụ thể: xem README. Nếu README chưa có, **không tự bịa** — ghi TBD và hỏi.
- Mọi entrypoint nhận `--config`, không nhận hyperparameter qua CLI rời rạc.

## 3. Cấu trúc & quyền sửa

```text
configs/rq{1,2,3}/   # config thí nghiệm
data/                # raw, processed, splits (không commit)
src/                 # module dùng chung: data, models, training, utils
research/rq{1,2,3}/  # code đặc thù từng RQ
experiments/rq{1,2,3}/ # kết quả chạy (không commit)
notebooks/rq{1,2,3}/ # notebook theo từng RQ (chỉ commit bản sạch)
app/                 # Streamlit demo
tests/               # pytest
docs/                # tài liệu, ma trận thực nghiệm, docs/notebooks/rqX/ chứa PDF
```

- `src/` là code dùng chung: sửa phải báo nhóm, chạy test, và không phá API cũ.
- `research/rqX/` chỉ phụ trách đúng RQ đó. Không import chéo giữa các `research/rqX/`; chỉ import từ `src/`.
- `data/`, `experiments/`: **chỉ đọc**. Chỉ script mới được tạo run/thư mục mới, không sửa run cũ.
- Kết quả/số liệu đã ghi trong `experiments/` là bất biến.

## 4. Quy ước code

- Type hint cho public function; docstring bằng tiếng Việt ngắn gọn.
- Không hardcode đường dẫn tuyệt đối; dùng `pathlib.Path` + config.
- Config-driven: đổi thí nghiệm bằng YAML, không sửa code.
- Tách rõ: dataset/dataloader, model, train loop, evaluate, inference.
- Seed mặc định 42; seed chính thức cho thí nghiệm chính: 42, 123, 2024.
- Tên: `snake_case` cho file/hàm/biến, `PascalCase` cho class.
- Không thêm dependency mới khi chưa hỏi.

### Notebook

- Notebook được dùng để chạy EDA, huấn luyện/đánh giá thực nghiệm và phân tích kết quả. Toàn bộ logic vẫn nằm trong `src/`; notebook chỉ cấu hình, gọi hàm và hiển thị kết quả.
- `src/` phải viết dạng thư viện (hàm/class có thể import); notebook chỉ `import` và gọi. Không copy-paste code, không định nghĩa lại model, train loop hay dataset trong notebook.
- Huấn luyện từ notebook phải dùng cùng config/entrypoint như CLI và ghi kết quả vào `experiments/rqX/<exp_id>_s<seed>/` với đầy đủ artifact như quy định. Không để kết quả chỉ tồn tại trong output notebook.
- Logic mới phát sinh trong notebook mà có thể tái sử dụng → chuyển vào `src/` kèm test, notebook gọi lại.
- Notebook phải chạy được từ trên xuống dưới; seed và hyperparameter lấy từ config, không hardcode.
- Mỗi RQ có thư mục riêng: `notebooks/rqX/`. Tên file theo mục đích (vd: `notebooks/rq3/eda_cross_dataset.ipynb`, `notebooks/rq1/train_compare.ipynb`).
- Quy trình commit notebook đã chạy:
  1. Xuất PDF để lưu kết quả: `uv run jupyter nbconvert --to pdf notebooks/rqX/<file>.ipynb --output-dir docs/notebooks/rqX/`
  2. Xóa toàn bộ output: `uv run jupyter nbconvert --clear-output --inplace notebooks/rqX/<file>.ipynb`
  3. Commit cả file `.ipynb` đã sạch và PDF trong `docs/notebooks/rqX/`.
- Không commit notebook còn output, ảnh nhúng, checkpoint, dataset hay cache (`**/.ipynb_checkpoints/`).
- Nếu thiếu toolchain xuất PDF (LaTeX), dừng lại và báo nhóm — không commit notebook khi chưa có PDF.

## 5. Luật dữ liệu (bất biến)

- Không commit dataset, split, checkpoint, cache (`data/`, `experiments/` nằm trong `.gitignore`).
- Label mapping và split do pipeline sinh ra, lưu ở `data/splits/` kèm version.
- Split một lần, dùng lại cho mọi thí nghiệm so sánh.
- Kiểm tra ảnh trùng lặp trước khi split; không để dữ liệu liên quan nằm ở 2 split.
- **Test set chỉ dùng cho đánh giá cuối.** Cấm dùng test để chọn epoch, threshold, hyperparameter hay biến thể ablation.

## 6. Luật thực nghiệm (bất biến)

- Mỗi lần chạy tạo thư mục mới: `experiments/rqX/<exp_id>_s<seed>/`.
- `<exp_id>` dạng `rq1_customcnn`, `rq2_mobilenetv3s_aug`, `rq2_mobilenetv3s_noaug_nocw`.
- Mỗi run lưu: bản copy config, seed, git commit hash, `metrics.json`, log, checkpoint, confusion matrix, learning curves.
- Mỗi thí nghiệm chỉ thay đổi **một** biến; ghi rõ các biến giữ cố định.
- Cập nhật `docs/experiment_matrix.md` trước khi chạy thí nghiệm mới.
- Không ghi đè hay xóa run cũ. Chạy lại thì tạo run mới.
- Thí nghiệm chính báo cáo mean ± std trên ≥3 seed.

## 7. Đánh giá & artifact

- Metric chuẩn: Accuracy, Precision, Recall, Macro-F1, Confusion Matrix.
- Ghi thêm khi liên quan: training time, inference latency, số tham số.
- Artifact mỗi run: `metrics.json`, `confusion_matrix.png`, `loss_curve.png`, `predictions.csv`.
- Không chọn lọc, ẩn hoặc bỏ kết quả bất lợi.

## 8. Git

- Nhánh: `main`, `feature/shared-pipeline`, `feature/rq1-model-comparison`, `feature/rq2-ablation`, `feature/rq3-generalization`.
- Không push trực tiếp vào `main`; mọi thay đổi qua PR.
- Commit message: `<scope>: <mô tả>` (vd: `rq1: add MobileNetV3_small baseline config`).
- Trước khi commit: `uv run ruff check .` và `uv run pytest`; notebook phải sạch output và đã có PDF tương ứng.

## 9. Trung thực học thuật

- Trích dẫn RealWaste, TrashNet, pretrained weights, thư viện và phần có AI hỗ trợ.
- Không trình bày mô hình pretrained/có sẵn như đóng góp của nhóm.
- Không chỉnh sửa kết quả để che giấu thí nghiệm bất lợi.

## 10. Việc agent được và không được làm

- Được: sửa `src/`, `research/`, `configs/`, `app/`, `tests/`, `notebooks/`, `docs/`.
- Không tự chạy training dài/tốn GPU khi chưa được yêu cầu.
- Không xóa/ghi đè `experiments/`, `data/splits/`, kết quả cũ.
- Không đổi interface của `src/` mà không chạy test và ghi chú lại.
- Khi kết thúc task: chạy lint + test liên quan, cập nhật README/docs nếu đổi lệnh hoặc hành vi.

## 11. Definition of Done

- [ ] Code chạy được với config mẫu.
- [ ] Có test cho logic mới trong `src/`.
- [ ] Run mới đúng naming và đủ artifact.
- [ ] Không commit file bị cấm.
- [ ] Notebook commit ở trạng thái sạch output, kèm PDF trong `docs/notebooks/rqX/`.
- [ ] README/docs đã cập nhật nếu cần.

## 12. Quyết định đã chốt

- 5 lớp: Cardboard, Glass, Metal, Paper, Plastic. Loại `trash` (TrashNet) và Food Organics, Miscellaneous Trash, Textile Trash, Vegetation (RealWaste).
- Mapping nhãn + số lượng ảnh ghi tại `docs/label_mapping.md`; EDA xác nhận trước khi đóng băng label space.
- RQ1: Custom CNN from scratch (baseline) vs MobileNetV3_small pretrained ImageNet.
- Input 224×224, normalize mean/std ImageNet.
- Train: `RandomResizedCrop(224, scale=(0.7, 1.0))`; val/test: `Resize(256)` + `CenterCrop(224)`.
- Split 70/15/15, stratified theo (dataset, class), gộp nhóm ảnh trùng (pHash distance ≤ 5), seed 42; lưu bất biến tại `data/splits/v1/split.csv` (`path, dataset, label, split, group_id`).
- RQ3: train từng nguồn → test toàn bộ nguồn còn lại; train pooled → báo test riêng theo từng nguồn.
- Chọn model/epoch theo Macro-F1 trên validation; early stopping patience 7–10; test chỉ dùng một lần cuối.
- Seed 42, 123, 2024; báo mean ± std.
- Class weighting: `w_c = N / (K * n_c)`, clip [0.5, 3.0]; chỉ dùng một trong {weighted loss, WeightedRandomSampler}.
- Fine-tune MobileNetV3_small: freeze backbone 3 epoch, mở 2 block cuối với lr/10.
- Training budget: tối đa 30 epoch, cùng ngân sách cho mọi mô hình.
- Demo: temperature scaling fit trên validation; confidence threshold chọn trên validation và đóng băng trước khi đánh giá test.
- Determinism: seed cố định, `cudnn.benchmark=False`; ghi chú giới hạn phi tất định trên GPU.

## 13. TBD

- Xác nhận mapping/count sau EDA; điều chỉnh split ratio nếu có lớp < 100 ảnh/split.
- Có dùng 4 lớp RealWaste còn lại làm OOD test hay không.
