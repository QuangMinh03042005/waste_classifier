"""Test sinh split mix-ratio cho RQ3."""

from __future__ import annotations

import pytest

from src.data.rq3_splits import _subsample_groups, make_mix_split
from src.data.split import SplitError, SplitRow


def _row(path: str, dataset: str, label: str, split: str, group: str) -> SplitRow:
    return SplitRow(
        path=path,
        dataset=dataset,
        label=label,
        raw_label=label,
        split=split,
        group_id=group,
    )


def _rows() -> list[SplitRow]:
    rows: list[SplitRow] = []
    for label in ("cardboard", "glass"):
        for split in ("train", "val", "test"):
            for dataset in ("realwaste", "trashnet"):
                for index in range(4):
                    rows.append(
                        _row(
                            f"raw/{dataset}/{label}_{split}_{index}.jpg",
                            dataset,
                            label,
                            split,
                            f"{dataset}-{label}-{split}-{index}",
                        )
                    )
    return rows


def test_share_0_khong_dung_trashnet_train() -> None:
    split = make_mix_split(_rows(), trashnet_share=0.0)
    assert all(row.dataset == "realwaste" for row in split if row.split == "train")
    assert sum(row.split == "val" for row in split) == 16
    assert sum(row.split == "test" for row in split) == 16


def test_share_05_giu_nguyen_group_va_val_test() -> None:
    rows = _rows()
    split = make_mix_split(rows, trashnet_share=0.5, seed=42)

    trashnet_train = [row for row in split if row.dataset == "trashnet" and row.split == "train"]
    assert 0 < len(trashnet_train) < 8
    assert len({row.group_id for row in trashnet_train}) == len(trashnet_train)

    for split_name, expected in (("val", 16), ("test", 16)):
        kept = [row for row in split if row.split == split_name]
        assert len(kept) == expected
        assert {row.path for row in kept} == {row.path for row in rows if row.split == split_name}


def test_share_khong_hop_le() -> None:
    with pytest.raises(SplitError):
        _subsample_groups(_rows(), share=1.5, seed=42)


def test_subsample_tat_dinh() -> None:
    rows = [row for row in _rows() if row.dataset == "trashnet" and row.split == "train"]
    first = _subsample_groups(rows, share=0.5, seed=42)
    second = _subsample_groups(rows, share=0.5, seed=42)
    assert [row.path for row in first] == [row.path for row in second]


def test_share_nho_nhat_lay_mot_group_moi_lop() -> None:
    rows = [row for row in _rows() if row.dataset == "trashnet" and row.split == "train"]
    selected = _subsample_groups(rows, share=0.01, seed=42)
    labels = {row.label for row in selected}
    assert labels == {"cardboard", "glass"}
