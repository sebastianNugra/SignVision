"""
GestureClassifier

Responsibility: Classify hand landmark feature vectors into sign
language gestures using a trained model and its label map.
"""

import numpy as np

from signvision.models.label_map import LabelMap
from signvision.models.model_loader import ModelLoader


class GestureClassifier:
    """
    Optionally loads the trained model and label map, then maps landmark
    vectors to gesture labels with a confidence score.
    """

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        label_map: LabelMap | None = None,
    ) -> None:
        self._model_loader = model_loader or ModelLoader()
        self._label_map = label_map or LabelMap.from_path()

    def load(self) -> None:
        """Load the underlying model so predictions can run."""
        self._model_loader.load()

    def classify(self, vector: np.ndarray) -> tuple[str, float]:
        """Return ``(label, confidence)`` for a landmark feature vector."""
        probabilities = self._model_loader.predict(vector)

        label_index = int(np.argmax(probabilities))
        confidence = float(probabilities[label_index])

        return self._label_map.label_for_index(label_index), confidence
