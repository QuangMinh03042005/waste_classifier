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
    ci95,
    compare_pooled,
    compute_metrics,
    configure_names,
    confusion_pairs,
    domain_gap,
    environment_table,
    learning_curves,
    load_runs,
    paired_delta,
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
    history = []
    for epoch in range(1, 4):
        history.append({"phase": "train", "epoch": epoch, "loss": 1.0 - epoch * 0.2})
        history.append(
            {
                "phase": "val",
                "epoch": epoch,
                "loss": 1.1 - epoch * 0.2,
                "macro_f1": 0.5 + epoch * 0.1,
            }
        )
    (run_dir / "history.jsonl").write_text(
        "\n".join(json.dumps(record) for record in history) + "\n", encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "torch": {
                    "version": "2.15.0+cu132",
                    "cuda_version": "13.2",
                    "device_name": "NVIDIA GeForce RTX 5070",
                }
            }
        ),
        encoding="utf-8",
    )
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

    gaps = domain_gap(agg)
    assert set(gaps["test_source"]) == {"realwaste", "trashnet"}
    single_trashnet = gaps[gaps["test_source"] == "trashnet"].set_index("label")
    # Val 0.9, test 1.0 -> gap âm; pooled thấp hơn single-source trên cùng nguồn.
    assert (
        single_trashnet.loc["Pooled (TrashNet 100%)", "gap"]
        > single_trashnet.loc["RealWaste → TrashNet", "gap"]
    )


def test_ci95_theo_phan_phoi_t() -> None:
    assert ci95(0.01, 3) == pytest.approx(4.303 * 0.01 / 3**0.5)
    assert ci95(0.01, 2) == pytest.approx(12.706 * 0.01 / 2**0.5)
    assert ci95(0.01, 1) is None
    assert ci95(None, 3) is None
    assert ci95(0.01, 30) is None


def test_paired_delta_theo_seed(tmp_path: Path) -> None:
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

    rows = _run_rows(load_runs(tmp_path))
    paired = paired_delta(rows)
    assert len(paired) == 1

    row = paired.iloc[0]
    assert row["test_source"] == "trashnet"
    assert row["n_pairs"] == 2
    # Single-source hoàn hảo (F1=1) nên delta pooled - single phải âm.
    assert row["delta_mean"] < 0
    assert row["delta_ci95"] is not None
    assert "42:" in row["seeds"] and "123:" in row["seeds"]


def test_prediction_shift_phat_hien_sup_do(tmp_path: Path) -> None:
    samples = [("trashnet", label, "cardboard") for label in LABELS]
    _make_run(tmp_path, "rq3_train_realwaste_test_trashnet", 42, samples)
    shift = prediction_shift(load_runs(tmp_path))
    assert shift.iloc[0]["top_predicted"] == "cardboard"
    assert shift.iloc[0]["top_share"] == pytest.approx(1.0)


def test_confusion_pairs_dem_loi(tmp_path: Path) -> None:
    samples = [
        ("trashnet", "glass", "cardboard"),
        ("trashnet", "glass", "glass"),
        ("trashnet", "glass", "cardboard"),
    ]
    _make_run(tmp_path, "rq3_train_realwaste_test_trashnet", 42, samples)

    pairs = confusion_pairs(load_runs(tmp_path))
    first = pairs.iloc[0]
    assert first["true_label"] == "glass"
    assert first["pred_label"] == "cardboard"
    assert first["count"] == 2
    assert first["share_of_true"] == pytest.approx(2 / 3)


def test_environment_table_doc_gpu_va_thoi_gian(tmp_path: Path) -> None:
    _make_run(
        tmp_path,
        "rq3_mix_tn25",
        42,
        [("trashnet", "glass", "glass"), ("realwaste", "metal", "metal")],
    )
    table = environment_table(load_runs(tmp_path))
    assert len(table) == 1
    row = table.iloc[0]
    assert row["gpu"] == "NVIDIA GeForce RTX 5070"
    assert row["torch"] == "2.15.0+cu132"
    assert row["cuda"] == "13.2"
    assert row["params"] == 1_500_000
    assert row["minutes_per_run"] >= 0


def test_learning_curves_tao_hinh(tmp_path: Path) -> None:
    _make_run(tmp_path, "rq3_mix_tn00", 42, [("trashnet", "glass", "glass")])
    figures = tmp_path / "figures"
    figures.mkdir()

    files = learning_curves(
        load_runs(tmp_path), figures, ["rq3_mix_tn00"], "learning_curves_mix.png"
    )
    assert files == ["learning_curves_mix.png"]
    assert (figures / "learning_curves_mix.png").stat().st_size > 0


def test_configure_names_e50(tmp_path: Path) -> None:
    import analyze_rq3 as module

    try:
        configure_names("_e50")
        assert "rq3_e50_mix_tn25" in module.PROTOCOLS
        assert module.SINGLE_SOURCE["trashnet"] == "rq3_e50_train_realwaste_test_trashnet"

        _make_run(tmp_path, "rq3_e50_mix_tn25", 42, [("trashnet", "glass", "glass")])
        data = load_runs(tmp_path)
        assert "rq3_e50_mix_tn25" in data
        assert data["rq3_e50_mix_tn25"][0]["history"]
    finally:
        configure_names("")
        assert "rq3_e50_mix_tn25" not in module.PROTOCOLS
