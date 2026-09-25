import json

import numpy as np
import tensorflow as tf

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


def test_build_model_has_expected_shapes() -> None:
    model = LandmarkTrainer().build_model(3)

    assert model.input_shape == (None, 21, 3)
    assert model.output_shape == (None, 3)


def test_train_then_predict_separated_classes() -> None:
    trainer = LandmarkTrainer(epochs=30, seed=0)
    model = trainer.train(_dataset())

    assert trainer.predict(model, np.full((21, 3), 1.0))[0] >= 0.9
    assert trainer.predict(model, np.full((21, 3), -1.0))[1] >= 0.9


def test_save_exports_saved_model_and_labels(tmp_path) -> None:
    trainer = LandmarkTrainer(epochs=30, seed=0)
    model = trainer.train(_dataset())

    model_dir = tmp_path / "gesture_model"
    trainer.save(model, model_dir, ["A", "B"])

    assert (model_dir / "saved_model.pb").exists()
    assert (model_dir / "variables").is_dir()

    labels_path = tmp_path / "gesture_model_labels.json"
    assert labels_path.exists()
    assert json.loads(labels_path.read_text()) == ["A", "B"]


def test_saved_model_serves_inference(tmp_path) -> None:
    trainer = LandmarkTrainer(epochs=30, seed=0)
    model = trainer.train(_dataset())

    model_dir = tmp_path / "gesture_model"
    trainer.save(model, model_dir, ["A", "B"])

    loaded = tf.saved_model.load(str(model_dir))
    signature = loaded.signatures["serve"]
    output = signature(tf.constant(np.full((1, 21, 3), 1.0, dtype=np.float32)))

    probabilities = next(iter(output.values())).numpy()[0]
    assert probabilities.argmax() == 0
