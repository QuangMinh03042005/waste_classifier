"""Test gom bộ file RQ3 để upload Drive."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_rq3_bundle import build_bundle  # noqa: E402


def _fake_run(experiments_root: Path, name: str = "rq3_mix_tn25_s42") -> Path:
    run_dir = experiments_root / "rq3" / name
    (run_dir / "predictions").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "experiment": "rq3_mix_tn25",
                "seed": 42,
                "status": "completed",
                "duration_sec": 211.0,
                "dataset": {"train": 2577, "val": 826, "test": 823},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_val": {"epoch": 8, "macro_f1": 0.86},
                "test": {"macro_f1": 0.85},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "predictions" / "test.csv").write_text(
        "path,dataset,true_label,pred_label,correct\n"
        "raw/trashnet/a.jpg,trashnet,glass,glass,1\n",
        encoding="utf-8",
    )
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"best")
    (run_dir / "checkpoints" / "last.pt").write_bytes(b"last")
    return run_dir


def test_bundle_mac_dinh_khong_kem_checkpoint(tmp_path: Path) -> None:
    experiments_root = tmp_path / "exp"
    _fake_run(experiments_root)

    info = build_bundle(
        output_dir=tmp_path / "dist",
        experiments_root=experiments_root,
        checkpoints="none",
    )

    assert info["zip"].is_file()
    assert info["runs"] == 1
    with zipfile.ZipFile(info["zip"]) as archive:
        names = archive.namelist()
    assert any(name.endswith("INDEX.md") for name in names)
    assert any(name.endswith("runs/rq3/rq3_mix_tn25_s42/run.json") for name in names)
    assert any(name.endswith("predictions/test.csv") for name in names)
    assert not any(name.endswith(".pt") for name in names)
    assert (info["zip"].with_name(info["zip"].name + ".sha256")).is_file()


def test_bundle_kem_best_checkpoint(tmp_path: Path) -> None:
    experiments_root = tmp_path / "exp"
    _fake_run(experiments_root)

    info = build_bundle(
        output_dir=tmp_path / "dist",
        experiments_root=experiments_root,
        checkpoints="best",
    )

    with zipfile.ZipFile(info["zip"]) as archive:
        names = archive.namelist()
    assert any(name.endswith("checkpoints/best.pt") for name in names)
    assert not any(name.endswith("checkpoints/last.pt") for name in names)


def test_index_lieu_co_so_lieu_run(tmp_path: Path) -> None:
    experiments_root = tmp_path / "exp"
    _fake_run(experiments_root)

    info = build_bundle(
        output_dir=tmp_path / "dist",
        experiments_root=experiments_root,
        checkpoints="none",
    )

    index = (info["directory"] / "INDEX.md").read_text(encoding="utf-8")
    assert "rq3_mix_tn25_s42" in index
    assert "0.8500" in index
    assert "2577/826/823" in index
