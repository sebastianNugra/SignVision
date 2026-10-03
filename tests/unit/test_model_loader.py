from pathlib import Path

import numpy as np
import pytest

from signvision.models import ModelLoader
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


def _trained_model_dir(tmp_path) -> Path:
    trainer = LandmarkTrainer(epochs=30, seed=0)
    model = trainer.train(_dataset())

    model_dir = tmp_path / "gesture_model"
    trainer.save(model, model_dir, ["A", "B"])

    return model_dir


def test_load_and_predict(tmp_path) -> None:
    loader = ModelLoader(_trained_model_dir(tmp_path))
    loader.load()

    probabilities = loader.predict(np.full((21, 3), 1.0, dtype=np.float32))

    assert probabilities.shape == (2,)
    assert probabilities.argmax() == 0
    assert np.isclose(probabilities.sum(), 1.0)


def test_load_missing_directory(tmp_path) -> None:
    loader = ModelLoader(tmp_path / "missing_model")

    with pytest.raises(RuntimeError, match="not found"):
        loader.load()


def test_predict_before_load_raises() -> None:
    loader = ModelLoader()

    with pytest.raises(RuntimeError, match="not loaded"):
        loader.predict(np.zeros((21, 3), dtype=np.float32))
