"""
Trainer

Responsibility: Train a TensorFlow model on landmark samples and
export it as a SavedModel for gesture classification.
"""

import json
import shutil
from pathlib import Path

import keras
import numpy as np

from signvision.training.dataset import LandmarkDataset


class LandmarkTrainer:
    """Trains and exports a landmark classification model."""

    def __init__(
        self,
        epochs: int = 30,
        validation_split: float = 0.2,
        dropout: float = 0.3,
        seed: int = 42,
    ) -> None:
        """Initialize the trainer with the given hyperparameters."""
        self._epochs = epochs
        self._validation_split = validation_split
        self._dropout = dropout
        self._seed = seed

    def build_model(self, num_classes: int) -> keras.Model:
        """Build a compiled classification model for the landmark features."""
        keras.utils.set_random_seed(self._seed)

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(21, 3)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(self._dropout),
                keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train(self, dataset: LandmarkDataset) -> keras.Model:
        """Train the model on the landmark dataset and return it."""
        model = self.build_model(len(dataset.label_names))

        model.fit(
            dataset.vectors,
            dataset.labels,
            epochs=self._epochs,
            validation_split=self._validation_split,
            verbose=0,
        )

        return model

    def save(
        self,
        model: keras.Model,
        output_dir: str | Path,
        label_names: list[str],
    ) -> None:
        """Export the trained model as a SavedModel with its label mapping."""
        output_path = Path(output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            shutil.rmtree(output_path)

        model.export(str(output_path))

        labels_path = output_path.parent / f"{output_path.name}_labels.json"
        labels_path.write_text(json.dumps(label_names, indent=2))

    def predict(self, model: keras.Model, vector: np.ndarray) -> np.ndarray:
        """Return the class probability distribution for a landmark vector."""
        probabilities: np.ndarray = np.asarray(
            model.predict(vector[np.newaxis, ...], verbose=0),
            dtype=np.float32,
        )[0]

        return probabilities
