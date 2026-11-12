"""Metadata của các dataset dùng trong dự án.

Thông tin ở đây là nguồn sự thật duy nhất cho URL tải, cấu trúc thư mục,
số lượng ảnh mong đợi và giấy phép. Không hardcode các giá trị này ở nơi khác.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Mô tả một dataset cần tải về."""

    name: str
    url: str
    archive_name: str
    root_in_archive: str
    expected_counts: dict[str, int]
    license: str
    citation: str
    homepage: str
    sha256: str | None = None

    @property
    def expected_total(self) -> int:
        return sum(self.expected_counts.values())


REALWASTE = DatasetSpec(
    name="realwaste",
    url="https://archive.ics.uci.edu/static/public/908/realwaste.zip",
    archive_name="realwaste.zip",
    root_in_archive="realwaste-main/RealWaste",
    expected_counts={
        "Cardboard": 461,
        "Food Organics": 411,
        "Glass": 420,
        "Metal": 790,
        "Miscellaneous Trash": 495,
        "Paper": 500,
        "Plastic": 921,
        "Textile Trash": 318,
        "Vegetation": 436,
    },
    license="CC BY 4.0",
    citation=(
        "Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste [Dataset]. "
        "UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G"
    ),
    homepage="https://archive.ics.uci.edu/dataset/908/realwaste",
)

TRASHNET = DatasetSpec(
    name="trashnet",
    url=(
        "https://raw.githubusercontent.com/garythung/trashnet/master/"
        "data/dataset-resized.zip"
    ),
    archive_name="trashnet-dataset-resized.zip",
    root_in_archive="dataset-resized",
    expected_counts={
        "cardboard": 403,
        "glass": 501,
        "metal": 410,
        "paper": 594,
        "plastic": 482,
        "trash": 137,
    },
    license="MIT",
    citation=(
        "Thung, G., & Yang, M. (2016). Classification of Trash for Recyclability "
        "Status. Stanford University. https://github.com/garythung/trashnet"
    ),
    homepage="https://github.com/garythung/trashnet",
)

DATASETS: dict[str, DatasetSpec] = {
    REALWASTE.name: REALWASTE,
    TRASHNET.name: TRASHNET,
}
