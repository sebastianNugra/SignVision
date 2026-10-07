"""
TranslationService

Responsibility: Coordinate the full pipeline from landmark input to
gesture classification and result delivery.

The service receives normalized hand landmark vectors (typically
extracted from live camera frames) and turns confidently detected
gesture poses into a sequence of words.
"""

from dataclasses import dataclass

import numpy as np

from signvision.models import GestureClassifier


@dataclass(frozen=True)
class GestureResult:
    """A confirmed gesture and its classification confidence."""

    label: str
    confidence: float


class TranslationService:
    """
    Turns a stream of landmark vectors into a sentence of gestures.

    A word is only confirmed when the same gesture is detected with
    sufficient confidence for ``confirm_frames`` consecutive frames,
    which avoids spurious single-frame classifications.
    """

    def __init__(
        self,
        classifier: GestureClassifier,
        min_confidence: float = 0.7,
        confirm_frames: int = 3,
    ) -> None:
        """
        Initialize the translation service.

        Args:
            classifier: Classifies landmark vectors into gesture labels.
            min_confidence: Minimum confidence to consider a detection.
            confirm_frames: Consecutive frames needed to confirm a word.
        """
        if not 0.0 < min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0, 1]")

        if confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")

        self._classifier = classifier
        self._min_confidence = min_confidence
        self._confirm_frames = confirm_frames

        self._streak_label: str | None = None
        self._streak_count = 0
        self._words: list[str] = []

    @property
    def words(self) -> list[str]:
        """The gesture words confirmed so far."""
        return list(self._words)

    @property
    def text(self) -> str:
        """The current translation as a space-separated sentence."""
        return " ".join(self._words)

    def process(self, vector: np.ndarray) -> GestureResult | None:
        """
        Classify a landmark vector and confirm a new gesture when ready.

        Returns a ``GestureResult`` only on the frame where a gesture
        first meets the confidence and hold requirements.
        """
        label, confidence = self._classifier.classify(vector)

        if confidence < self._min_confidence:
            self._reset_streak()
            return None

        if label == self._streak_label:
            self._streak_count += 1
        else:
            self._streak_label = label
            self._streak_count = 1

        if self._streak_count < self._confirm_frames:
            return None

        if self._words and self._words[-1] == label:
            return None

        self._words.append(label)
        return GestureResult(label, confidence)

    def reset(self) -> None:
        """Clear the current translation and confirmation state."""
        self._words = []
        self._reset_streak()

    def _reset_streak(self) -> None:
        self._streak_label = None
        self._streak_count = 0
