"""
ModelLoader

Responsibility: Load a trained gesture classification SavedModel from
disk and run inference on landmark feature vectors.
"""

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from signvision.config.paths import GESTURE_MODEL_DIR


class ModelLoader:
    """
    Loads a TensorFlow SavedModel and runs its ``serve`` signature.

    The exported model expects a feature vector shaped ``(21, 3)`` and
    returns the class probability distribution.
    """

    def __init__(self, model_dir: str | Path = GESTURE_MODEL_DIR) -> None:
        self._model_dir = Path(model_dir)
        self._model: Any | None = None
        self._signature: Any | None = None

    def load(self) -> None:
        """Load the model from disk and resolve its ``serve`` signature."""
        if not self._model_dir.exists():
            raise RuntimeError(
                f"Model directory not found: {self._model_dir}. "
                "Train a model with 'python scripts/train.py' first."
            )

        loaded = tf.saved_model.load(str(self._model_dir))
        self._signature = loaded.signatures["serve"]
        self._model = loaded

    def predict(self, vector: np.ndarray) -> np.ndarray:
        """Return the class probability distribution for a landmark vector."""
        if self._signature is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")

        tensor = tf.constant(vector[np.newaxis, ...], dtype=tf.float32)
        output = self._signature(tensor)
        raw = next(iter(output.values())).numpy()

        probabilities: np.ndarray = np.asarray(raw, dtype=np.float32)[0]

        return probabilities
