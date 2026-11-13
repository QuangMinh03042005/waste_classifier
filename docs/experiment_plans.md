# Kế hoạch thí nghiệm (pre-registration)

Trước mỗi lần chạy, kế hoạch phải được viết, commit và dùng làm đầu vào cho training.
Mục đích: chứng minh giả thuyết/tiêu chí được đặt **trước** khi xem kết quả, đúng yêu
cầu của đề tài.

## Vị trí

```text
plans/
├── rq1/model_comparison.yaml
├── rq2/augmentation_ablation.yaml
└── rq3/cross_dataset_generalization.yaml
```

Plan được **commit vào git**. Mỗi plan mô tả một RQ gồm baseline và các biến thể.

## Trường bắt buộc

| Trường | Ý nghĩa |
|---|---|
| `schema_version` | Phiên bản schema, hiện tại `"1.0"` |
| `plan_id` | ID duy nhất, chỉ chữ/số/`._-` |
| `rq` | `rq1`, `rq2`, `rq3` |
| `title`, `author`, `created_at` | Thông tin plan |
| `status` | `active`, `completed`, `aborted` |
| `question` | Câu hỏi nghiên cứu |
| `hypothesis` | Giả thuyết kiểm chứng được |
| `expected_result` | Kết quả kỳ vọng |
| `failure_plan` | Xử lý nếu giả thuyết sai |
| `baseline` | `{experiment, description}` |
| `variants` | Danh sách `{experiment, change, fixed[]}` |
| `comparison` | `{metric, secondary[], test_policy, success_criterion}` |
| `data` | `{train_datasets, test_dataset, splits_version, class_names}` |
| `seeds` | Danh sách seed, không trùng |
| `budget` | Tùy chọn: epoch, batch size, thời gian ước tính |

Quy tắc ablation: mỗi variant chỉ thay **một** biến; các biến còn lại ghi trong `fixed`.

## Quy trình

1. Tạo/sửa plan trong `plans/rqX/`.
2. Commit plan.
3. Sinh lại ma trận:

   ```bash
   uv run python scripts/experiment_matrix.py
   ```

4. Chạy training kèm plan:

   ```bash
   python -m research.rq1.train \
     --config configs/rq1/rq1_mobilenetv3s_full.yaml \
     --plan plans/rq1/model_comparison.yaml
   ```

5. Run copy plan vào `experiments/rq1/<run_id>/plan.yaml` và ghi hash vào `run.json`.

## Kiểm tra trước khi chạy

`src/training/run.py::start_run` từ chối chạy nếu:

- Thiếu `--plan`.
- `plan.status != "active"`.
- `rq` của plan khác config.
- `config["experiment"]` không nằm trong baseline/variants.
- Seed không có trong `plan.seeds`.
- Resume với plan đã bị sửa (hash khác lần trước).

Chỉ khi debug mới dùng `allow_unplanned=True`; run đó được đánh dấu
`"plan": {"status": "unplanned"}` và không dùng cho bảng kết quả chính.

## Snapshot trong run

```json
"plan": {
  "plan_id": "rq1_model_comparison",
  "rq": "rq1",
  "status": "active",
  "sha256": "…",
  "path": "plans/rq1/model_comparison.yaml",
  "snapshot": "plan.yaml"
}
```

Nhờ đó có thể đối chiếu plan đã commit với plan thực tế dùng khi chạy.

## Ma trận thí nghiệm

`docs/experiment_matrix.md` được sinh tự động, gồm:

- Câu hỏi, giả thuyết, kết quả kỳ vọng, xử lý nếu sai.
- Metric và test policy.
- Bảng experiment: baseline/variant, thay đổi, biến cố định, seeds, status.

Không sửa tay file này; sửa plan rồi chạy lại script.

## Khi kết thúc

- Plan giữ `active` trong lúc chạy.
- Sau khi hoàn tất RQ, đổi `status: completed` (hoặc `aborted`) và commit.
- Không sửa giả thuyết/tiêu chí sau khi đã xem test; nếu cần điều chỉnh, tạo plan mới.
