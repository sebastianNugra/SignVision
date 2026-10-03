from pathlib import Path

import numpy as np
import pytest

from signvision.models import GestureClassifier, LabelMap, ModelLoader
from signvision.training import LandmarkTrainer
from signvision.training.dataset import LandmarkDataset


def _dataset() -> LandmarkDataset:
    vectors = np.concatenate(
        [
            np.full((6, 21, 3), 1.0, dtype=np.float32),
            np.full((6, 21, 3), -1.0, dtype=np.float32),
        ]
    )
    labels = np.array([0] * 6 + [1] * 6, dtype=np.int32)

    return LandmarkDataset(
        vectors=vectors,
        labels=labels,
        label_names=["A", "B"],
    )


def _trained_artifacts(tmp_path) -> tuple[Path, Path]:
    trainer = LandmarkTrainer(epochs=30, seed=0)
    model = trainer.train(_dataset())

    model_dir = tmp_path / "gesture_model"
    trainer.save(model, model_dir, ["A", "B"])

    return model_dir, tmp_path / "gesture_model_labels.json"


def _classifier(model_dir: Path, labels_path: Path) -> GestureClassifier:
    classifier = GestureClassifier(
        model_loader=ModelLoader(model_dir),
        label_map=LabelMap.from_path(labels_path),
    )
    classifier.load()

    return classifier


def test_classify_known_gestures(tmp_path) -> None:
    model_dir, labels_path = _trained_artifacts(tmp_path)
    classifier = _classifier(model_dir, labels_path)

    label, confidence = classifier.classify(np.full((21, 3), 1.0, dtype=np.float32))

    assert label == "A"
    assert confidence >= 0.9

    label, confidence = classifier.classify(np.full((21, 3), -1.0, dtype=np.float32))

    assert label == "B"
    assert confidence >= 0.9


def test_classify_before_load_raises() -> None:
    classifier = GestureClassifier(
        model_loader=ModelLoader(),
        label_map=LabelMap(["A", "B"]),
    )

    with pytest.raises(RuntimeError, match="not loaded"):
        classifier.classify(np.zeros((21, 3), dtype=np.float32))
