import numpy as np
import pytest

from signvision.training.dataset import LandmarkDataset, load_dataset, save_dataset


def _dataset(
    samples: int = 4,
    labels: list[str] | None = None,
) -> LandmarkDataset:
    label_names = labels or ["A", "B"]

    return LandmarkDataset(
        vectors=np.ones((samples, 21, 3), dtype=np.float32),
        labels=np.zeros(samples, dtype=np.int32),
        label_names=label_names,
    )


def test_dataset_rejects_invalid_vectors_shape() -> None:
    with pytest.raises(ValueError):
        LandmarkDataset(
            vectors=np.zeros((4, 21), dtype=np.float32),
            labels=np.zeros(4, dtype=np.int32),
            label_names=["A"],
        )


def test_dataset_rejects_mismatched_labels_length() -> None:
    with pytest.raises(ValueError):
        LandmarkDataset(
            vectors=np.zeros((4, 21, 3), dtype=np.float32),
            labels=np.zeros(3, dtype=np.int32),
            label_names=["A"],
        )


def test_dataset_rejects_empty_label_names() -> None:
    with pytest.raises(ValueError):
        LandmarkDataset(
            vectors=np.zeros((4, 21, 3), dtype=np.float32),
            labels=np.zeros(4, dtype=np.int32),
            label_names=[],
        )


def test_dataset_rejects_unknown_label_index() -> None:
    with pytest.raises(ValueError):
        LandmarkDataset(
            vectors=np.zeros((2, 21, 3), dtype=np.float32),
            labels=np.asarray([5, 5], dtype=np.int32),
            label_names=["A"],
        )


def test_save_and_load_round_trip(tmp_path) -> None:
    dataset = _dataset(samples=3, labels=["A", "B", "C"])
    path = tmp_path / "dataset.npz"

    save_dataset(dataset, path)
    loaded = load_dataset(path)

    assert loaded.label_names == ["A", "B", "C"]
    assert loaded.vectors.shape == (3, 21, 3)
    assert loaded.vectors.dtype == np.float32
    assert np.array_equal(loaded.vectors, dataset.vectors)
    assert np.array_equal(loaded.labels, dataset.labels)
