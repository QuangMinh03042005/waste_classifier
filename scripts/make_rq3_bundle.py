"""Gom bộ file RQ3 để upload Google Drive theo dõi tiến độ.

Cách dùng:
    uv run python scripts/make_rq3_bundle.py
    uv run python scripts/make_rq3_bundle.py --checkpoints best

Sinh ra `dist/rq3_upload_<UTC>/` (thư mục để kéo lên Drive) và file `.zip` đi kèm.
Mặc định **không** kèm checkpoint để bộ file nhẹ; dùng `--checkpoints best` nếu cần.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import git_info  # noqa: E402
from src.utils.paths import experiments_root as default_experiments_root  # noqa: E402
from src.utils.paths import project_root  # noqa: E402

REPO_FOLDERS = (
    "docs/RQ3",
    "docs/notebooks/rq3",
    "notebooks/rq3",
    "configs/rq3",
    "plans/rq3",
)
REPO_FILES = (
    "data/splits/rq3/mix_summary.json",
    "data/splits/rq3/mix_tn_00.csv",
    "data/splits/rq3/mix_tn_25.csv",
    "data/splits/rq3/mix_tn_50.csv",
    "data/splits/rq3/mix_tn_75.csv",
    "data/splits/rq3/pooled.csv",
    "data/splits/rq3/rw_to_tn.csv",
    "data/splits/rq3/summary.json",
    "data/splits/rq3/tn_to_rw.csv",
    "research/rq3/__init__.py",
    "research/rq3/train.py",
    "scripts/analyze_rq3.py",
    "scripts/make_rq3_mix_splits.py",
    "scripts/track_rq3.py",
    "src/data/rq3_splits.py",
    "tests/test_analyze_rq3.py",
    "tests/test_rq3_configs.py",
    "tests/test_rq3_mix_splits.py",
    "tests/test_rq3_split_integrity.py",
    "tests/test_track_rq3.py",
)
RUN_FILES = (
    "run.json",
    "config.yaml",
    "plan.yaml",
    "environment.json",
    "history.jsonl",
    "metrics.json",
    "predictions/test.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _run_rows(experiments_root: Path, checkpoints: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_file in sorted((experiments_root / "rq3").glob("*/run.json")):
        run = json.loads(run_file.read_text(encoding="utf-8"))
        metrics_file = run_file.with_name("metrics.json")
        metrics = (
            json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.is_file() else {}
        )
        rows.append(
            {
                "run_dir": run_file.parent,
                "experiment": run.get("experiment"),
                "seed": run.get("seed"),
                "status": run.get("status"),
                "val_f1": (metrics.get("best_val") or {}).get("macro_f1"),
                "test_f1": (metrics.get("test") or {}).get("macro_f1"),
                "dataset": run.get("dataset") or {},
                "duration": run.get("duration_sec"),
                "checkpoints": checkpoints,
            }
        )
    return rows


def _index_markdown(
    created_at: str,
    git: dict[str, Any],
    rows: list[dict[str, Any]],
    checkpoints: str,
) -> str:
    lines = [
        "# RQ3 – Bộ file theo dõi tiến độ",
        "",
        f"- Tạo lúc: `{created_at}`",
        f"- Commit: `{(git.get('commit') or '')[:12]}` (nhánh `{git.get('branch')}`)",
        f"- Checkpoint kèm theo: `{checkpoints}`",
        f"- Số run: **{len(rows)}**",
        "",
        "## Đọc gì trước",
        "",
        "1. `docs/RQ3/rq3_results.md` – bảng số liệu chính (mean ± std, CI95, mix-ratio).",
        "2. `docs/RQ3/rq3_discussion.md` – thảo luận, giới hạn, kết luận.",
        "3. `docs/RQ3/rq3_tracking.xlsx` – tracking 21 run (4 sheet).",
        "4. `docs/notebooks/rq3/analyze_rq3.pdf` – notebook đã chạy.",
        "5. `docs/RQ3/rq3_runbook.md` – cách chạy lại.",
        "",
        "## Cấu trúc",
        "",
        "```text",
        "docs/RQ3/            báo cáo, hình, Excel tracking",
        "docs/notebooks/rq3/  PDF notebook",
        "notebooks/rq3/       notebook sạch output",
        "configs/rq3/         7 cấu hình thí nghiệm",
        "plans/rq3/           2 plan đã đăng ký",
        "data/splits/rq3/     split CSV đã dùng (bất biến)",
        "research/, scripts/, src/, tests/  code liên quan RQ3",
        "runs/rq3/<run>/      run.json, metrics.json, history, predictions,",
        "                     environment.json (checkpoint tuỳ chọn)",
        "```",
        "",
        "## Danh sách run",
        "",
        "| Run | Trạng thái | Val Macro-F1 | Test Macro-F1 | Train/Val/Test | Thời gian (s) |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        dataset = row["dataset"]
        sizes = "/".join(
            str(dataset.get(key)) if dataset.get(key) is not None else "-"
            for key in ("train", "val", "test")
        )
        val = row["val_f1"]
        test = row["test_f1"]
        lines.append(
            "| {run} | {status} | {val} | {test} | {sizes} | {duration} |".format(
                run=row["run_dir"].name,
                status=row["status"],
                val=f"{val:.4f}" if isinstance(val, float) else "-",
                test=f"{test:.4f}" if isinstance(test, float) else "-",
                sizes=sizes,
                duration=round(row["duration"] or 0),
            )
        )
    lines.extend(
        [
            "",
            "## Lưu ý",
            "",
            "- Test chỉ được chấm một lần mỗi run; số liệu trong `metrics.json` là nguồn gốc.",
            "- `predictions/test.csv` cho phép tính lại metric và confusion matrix.",
            "- Nếu cần checkpoint để resume/chạy lại đánh giá: tạo lại bộ này với "
            "`--checkpoints best` (hoặc dùng zip đầy đủ trong `experiments/exports/`).",
            "",
        ]
    )
    return "\n".join(lines)


def build_bundle(
    *,
    output_dir: Path,
    experiments_root: Path,
    checkpoints: str = "none",
    include_runs: bool = True,
) -> dict[str, Any]:
    """Tạo thư mục + zip bộ file RQ3; trả về thông tin bundle."""
    root = project_root()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = output_dir / f"rq3_upload_{stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for folder in REPO_FOLDERS:
        source = root / folder
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if path.is_file():
                _copy(path, bundle_dir / path.relative_to(root))
                copied += 1

    for name in REPO_FILES:
        source = root / name
        if source.is_file():
            _copy(source, bundle_dir / name)
            copied += 1

    rows = _run_rows(experiments_root, checkpoints) if include_runs else []
    for row in rows:
        run_dir = row["run_dir"]
        for name in RUN_FILES:
            source = run_dir / name
            if source.is_file():
                _copy(source, bundle_dir / "runs" / "rq3" / run_dir.name / name)
                copied += 1
        if checkpoints in {"best", "all"}:
            for name in ("best.pt", "last.pt") if checkpoints == "all" else ("best.pt",):
                source = run_dir / "checkpoints" / name
                if source.is_file():
                    _copy(
                        source,
                        bundle_dir / "runs" / "rq3" / run_dir.name / "checkpoints" / name,
                    )
                    copied += 1

    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    index = _index_markdown(created_at, git_info(root), rows, checkpoints)
    (bundle_dir / "INDEX.md").write_text(index + "\n", encoding="utf-8")
    copied += 1

    zip_path = bundle_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle_dir.parent))

    digest = _sha256(zip_path)
    checksum = zip_path.with_name(zip_path.name + ".sha256")
    checksum.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    return {
        "directory": bundle_dir,
        "zip": zip_path,
        "files": copied,
        "runs": len(rows),
        "bytes": zip_path.stat().st_size,
        "sha256": digest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gom bộ file RQ3 để upload Drive.")
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("--experiments-root", type=Path, default=default_experiments_root())
    parser.add_argument(
        "--checkpoints",
        choices=["none", "best", "all"],
        default="none",
        help="none: bộ nhẹ để theo dõi; best/all: kèm checkpoint (nặng hơn nhiều).",
    )
    parser.add_argument("--no-runs", action="store_true", help="Chỉ gom tài liệu/code.")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    info = build_bundle(
        output_dir=args.output_dir,
        experiments_root=args.experiments_root.resolve(),
        checkpoints=args.checkpoints,
        include_runs=not args.no_runs,
    )
    print(f"Thư mục: {info['directory']}")
    print(f"Zip: {info['zip']} ({info['bytes'] / 1024**2:.1f} MiB)")
    print(f"File: {info['files']} | Run: {info['runs']} | Checkpoint: {args.checkpoints}")
    print(f"sha256: {info['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
