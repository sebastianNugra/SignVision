"""
HandDetector

Responsibility: Detect hands in video frames using MediaPipe and manage
the lifecycle of the hand detection resources.
"""

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import core, vision

from signvision.config.paths import HAND_LANDMARKER_MODEL_PATH


class HandDetector:
    """Detects hands in video frames using the MediaPipe HandLandmarker."""

    def __init__(
        self,
        model_path: str | Path = HAND_LANDMARKER_MODEL_PATH,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize the hand detector with the given configuration."""
        self._model_path = Path(model_path)
        self._num_hands = num_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_hand_presence_confidence = min_hand_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence

        self._landmarker: vision.HandLandmarker | None = None
        self._timestamp_ms = 0

    def open(self) -> None:
        """Initialize the MediaPipe hand landmarker with the configured model."""
        options = vision.HandLandmarkerOptions(
            base_options=core.base_options.BaseOptions(
                model_asset_path=str(self._model_path)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self._num_hands,
            min_hand_detection_confidence=self._min_detection_confidence,
            min_hand_presence_confidence=self._min_hand_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )

        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open hand detector with model {self._model_path}."
            ) from exc

        self._timestamp_ms = 0

    def detect(self, frame: np.ndarray) -> vision.HandLandmarkerResult:
        """Detect hands in an RGB video frame and return the detection result."""
        if self._landmarker is None:
            raise RuntimeError("Hand detector is not open.")

        self._timestamp_ms += 33
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        return self._landmarker.detect_for_video(image, self._timestamp_ms)

    def close(self) -> None:
        """Release the MediaPipe hand detection resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            self._timestamp_ms = 0
