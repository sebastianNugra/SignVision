import mediapipe as mp
import numpy as np


class HandDetector:
    """Handles hand detection."""

    def __init__(self) -> None:
        self._hands = mp.solutions.hands.Hands()

    def detect(self, frame: np.ndarray) -> object:
        """Detects hands in a frame."""
        return self._hands.process(frame)
