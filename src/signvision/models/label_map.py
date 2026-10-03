"""
LabelMap

Responsibility: Manage the mapping between model output indices and
human-readable sign language label names.
"""

import json
from pathlib import Path

from signvision.config.paths import GESTURE_MODEL_LABELS_PATH


class LabelMap:
    """
    Maps model output indices to human-readable gesture labels.

    Labels are stored on disk as a JSON list, written by the trainer
    next to the exported model (e.g. ``gesture_model_labels.json``).
    """

    def __init__(self, labels: list[str]) -> None:
        if not labels:
            raise ValueError("labels must contain at least one label")

        for label in labels:
            if not isinstance(label, str) or not label:
                raise ValueError("labels must be non-empty strings")

        self._labels = list(labels)

    @classmethod
    def from_path(
        cls,
        path: str | Path = GESTURE_MODEL_LABELS_PATH,
    ) -> "LabelMap":
        """Load a label mapping from a JSON file."""
        raw = json.loads(Path(path).read_text())

        if not isinstance(raw, list):
            raise ValueError(f"labels file must contain a list: {path}")

        return cls(raw)

    @property
    def labels(self) -> list[str]:
        """The gesture labels in model output order."""
        return list(self._labels)

    def label_for_index(self, index: int) -> str:
        """Return the label for the given model output index."""
        if index < 0 or index >= len(self._labels):
            raise IndexError(
                f"label index {index} out of range (0-{len(self._labels) - 1})"
            )

        return self._labels[index]

    def __len__(self) -> int:
        return len(self._labels)
