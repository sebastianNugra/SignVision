"""
Dataset

Responsibility: Persist and load labeled hand landmark samples
for the model training pipeline.
"""

from dataclasses import dataclass

import numpy as np

from signvision.config.paths import DATASET_FILE

LANDMARK_SHAPE = (21, 3)


@dataclass(frozen=True)
class LandmarkDataset:
    """A labeled collection of normalized hand landmark samples."""

    vectors: np.ndarray
    labels: np.ndarray
    label_names: list[str]

    def __post_init__(self) -> None:
        if self.vectors.ndim != 3 or self.vectors.shape[1:] != LANDMARK_SHAPE:
            raise ValueError(
                f"vectors must have shape (n, 21, 3), got {self.vectors.shape}"
            )

        if self.labels.ndim != 1 or len(self.labels) != len(self.vectors):
            raise ValueError("labels must be a 1D array matching the number of vectors")

        if not self.label_names:
            raise ValueError("label_names must contain at least one label")

        if set(self.labels.tolist()) - set(range(len(self.label_names))):
            raise ValueError("labels reference unknown label names")


def save_dataset(dataset: LandmarkDataset, path: str | None = None) -> None:
    """Persist the landmark dataset to disk as a compressed array file."""
    output_path = str(path or DATASET_FILE)

    np.savez_compressed(
        output_path,
        vectors=dataset.vectors,
        labels=dataset.labels,
        label_names=np.asarray(dataset.label_names, dtype=str),
    )


def load_dataset(path: str | None = None) -> LandmarkDataset:
    """Load the landmark dataset stored by save_dataset."""
    input_path = str(path or DATASET_FILE)

    with np.load(input_path) as data:
        label_names = [str(name) for name in data["label_names"]]

        return LandmarkDataset(
            vectors=data["vectors"].astype(np.float32, copy=False),
            labels=data["labels"].astype(np.int32, copy=False),
            label_names=label_names,
        )
