"""
LandmarkExtractor

Responsibility: Extract and normalize hand landmark data
from MediaPipe detection results into usable feature vectors.
"""

import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import landmark

WRIST_INDEX = 0
MIDDLE_FINGER_MCP_INDEX = 9


class LandmarkExtractor:
    """Extracts normalized hand landmark feature vectors from MediaPipe results."""

    def extract(self, result: vision.HandLandmarkerResult) -> list[np.ndarray]:
        """Return one (21, 3) normalized feature vector per detected hand."""
        return [self._normalize(hand) for hand in result.hand_landmarks]

    def _normalize(
        self, hand_landmarks: list[landmark.NormalizedLandmark]
    ) -> np.ndarray:
        normalized = np.array(
            [[point.x, point.y, point.z] for point in hand_landmarks],
            dtype=np.float64,
        )

        wrist = normalized[WRIST_INDEX]

        reference = float(
            np.linalg.norm(normalized[MIDDLE_FINGER_MCP_INDEX, :2] - wrist[:2])
        )

        if not reference > 0:
            raise ValueError(
                "Cannot normalize hand landmarks with zero reference scale."
            )

        feature_vector: np.ndarray = (normalized - wrist) / reference

        return feature_vector
