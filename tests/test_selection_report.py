"""Test báo cáo chọn cấu hình."""

from __future__ import annotations

from scripts.selection_report import build_markdown, load_candidates


def test_build_markdown_without_runs(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "a.yaml").write_text(
        "experiment: rq1_customcnn_cand_a\n"
        "model: {channels: [8, 16, 32, 64], convs_per_stage: [1, 1, 1, 1], dropout: 0.3}\n"
        "training: {epochs: 12}\n",
        encoding="utf-8",
    )
    candidates = load_candidates(config_dir)
    text = build_markdown(candidates, [])
    assert "Chưa có run completed" in text
    assert "rq1_customcnn_cand_a" not in text
    assert "cand_a" in text


def test_build_markdown_with_runs() -> None:
    candidates = {
        "rq1_customcnn_cand_a": {
            "config": "a.yaml",
            "channels": [8, 16, 32, 64],
            "convs_per_stage": [1, 1, 1, 1],
            "dropout": 0.3,
            "epochs": 12,
        }
    }
    runs = [
        {
            "experiment": "rq1_customcnn_cand_a",
            "seed": 42,
            "val_macro_f1": 0.5,
            "val_loss": 1.0,
            "best_epoch": 3,
            "params": 1000,
            "duration_sec": 10.0,
            "git_commit": "abc",
            "run_id": "rq1_customcnn_cand_a_s42",
        }
    ]
    text = build_markdown(candidates, runs)
    assert "## Kết quả" in text
    assert "0.5000" in text
    assert "Tổng hợp theo ứng viên" in text
