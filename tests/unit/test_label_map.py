import json
from pathlib import Path

import pytest

from signvision.models import LabelMap


def _write_labels(tmp_path, labels) -> Path:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    return path


def test_from_path_loads_labels(tmp_path) -> None:
    label_map = LabelMap.from_path(_write_labels(tmp_path, ["A", "B", "C"]))

    assert label_map.labels == ["A", "B", "C"]
    assert len(label_map) == 3


def test_from_path_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        LabelMap.from_path(tmp_path / "missing.json")


def test_from_path_invalid_json(tmp_path) -> None:
    path = tmp_path / "labels.json"
    path.write_text("not-json")

    with pytest.raises(json.JSONDecodeError):
        LabelMap.from_path(path)


def test_from_path_not_a_list(tmp_path) -> None:
    with pytest.raises(ValueError, match="must contain a list"):
        LabelMap.from_path(_write_labels(tmp_path, {"A": 0}))


def test_empty_labels_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        LabelMap.from_path(_write_labels(tmp_path, []))


def test_non_string_label_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        LabelMap.from_path(_write_labels(tmp_path, ["A", 2]))


def test_empty_string_label_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        LabelMap.from_path(_write_labels(tmp_path, ["A", ""]))


def test_label_for_index(tmp_path) -> None:
    label_map = LabelMap.from_path(_write_labels(tmp_path, ["A", "B"]))

    assert label_map.label_for_index(0) == "A"
    assert label_map.label_for_index(1) == "B"


def test_label_for_index_out_of_range(tmp_path) -> None:
    label_map = LabelMap.from_path(_write_labels(tmp_path, ["A"]))

    with pytest.raises(IndexError):
        label_map.label_for_index(1)

    with pytest.raises(IndexError):
        label_map.label_for_index(-1)
