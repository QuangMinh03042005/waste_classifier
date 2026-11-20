"""Test tổng hợp kết quả RQ3 từ artifact giả lập."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_rq3 import (  # noqa: E402
    _run_rows,
    aggregate,
    compare_pooled,
    compute_metrics,
    load_runs,
    prediction_shift,
)

LABELS = ["cardboard", "glass", "metal", "paper", "plastic"]


def _make_run(
    root: Path,
    experiment: str,
    seed: int,
    samples: list[tuple[str, str, str]],
) -> Path:
    run_dir = root / "rq3" / f"{experiment}_s{seed}"
    (run_dir / "predictions").mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "seed": seed,
                "model": {"params": 1_500_000, "name": "mobilenet_v3_small"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"best_val": {"epoch": 3, "macro_f1": 0.9}}), encoding="utf-8"
    )
    rows = [
        {
            "path": f"raw/{dataset}/{index}.jpg",
            "dataset": dataset,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": int(true_label == pred_label),
        }
        for index, (dataset, true_label, pred_label) in enumerate(samples)
    ]
    pd.DataFrame(rows).to_csv(run_dir / "predictions" / "test.csv", index=False)
    return run_dir


def test_compute_metrics_khop_sklearn() -> None:
    predictions = pd.DataFrame(
        {
            "true_label": ["cardboard", "glass", "metal", "metal"],
            "pred_label": ["cardboard", "glass", "metal", "paper"],
        }
    )
    metrics = compute_metrics(predictions)
    assert metrics["n"] == 4
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert 0 < metrics["macro_f1"] < 1
    assert metrics["per_class"]["metal"]["recall"] == pytest.approx(0.5)
    assert len(metrics["confusion_matrix"]) == 5


def test_aggregate_va_so_sanh_pooled(tmp_path: Path) -> None:
    samples_single = [("trashnet", label, label) for label in LABELS for _ in range(2)]
    samples_pooled = [
        ("trashnet", "glass", "cardboard"),
        ("trashnet", "glass", "glass"),
        ("realwaste", "metal", "metal"),
        ("realwaste", "metal", "metal"),
    ]
    for seed in (42, 123):
        _make_run(tmp_path, "rq3_train_realwaste_test_trashnet", seed, samples_single)
        _make_run(tmp_path, "rq3_train_pooled_test_each", seed, samples_pooled)

    data = load_runs(tmp_path)
    assert len(data["rq3_train_realwaste_test_trashnet"]) == 2

    rows = _run_rows(data)
    agg = aggregate(rows)

    pooled_all = agg[
        (agg["experiment"] == "rq3_train_pooled_test_each") & (agg["test_source"] == "all")
    ].iloc[0]
    # 2/4 dự đoán đúng trên toàn bộ test của pooled.
    assert pooled_all["accuracy_mean"] == pytest.approx(0.75)
    assert pooled_all["n_seeds"] == 2

    comparison = compare_pooled(agg)
    trashnet_row = comparison[comparison["test_source"] == "trashnet"].iloc[0]
    assert trashnet_row["single_macro_f1"] == pytest.approx(1.0)
    assert trashnet_row["delta"] < 0
    assert not trashnet_row["meets_criterion"]


def test_prediction_shift_phat_hien_sup_do(tmp_path: Path) -> None:
    samples = [("trashnet", label, "cardboard") for label in LABELS]
    _make_run(tmp_path, "rq3_train_realwaste_test_trashnet", 42, samples)
    shift = prediction_shift(load_runs(tmp_path))
    assert shift.iloc[0]["top_predicted"] == "cardboard"
    assert shift.iloc[0]["top_share"] == pytest.approx(1.0)
