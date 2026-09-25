"""
DataCollector

Responsibility: Capture labeled hand landmark samples from the live
camera feed for building the training dataset.
"""

import time
from collections.abc import Callable

import cv2
import numpy as np

from signvision.camera import Camera
from signvision.vision import HandDetector, LandmarkExtractor


class DataCollector:
    """Collects normalized hand landmark samples from a live camera feed."""

    def __init__(
        self,
        camera: Camera,
        hand_detector: HandDetector,
        landmark_extractor: LandmarkExtractor,
        step_delay: float = 0.05,
    ) -> None:
        """Initialize the data collector with its pipeline components."""
        self._camera = camera
        self._hand_detector = hand_detector
        self._landmark_extractor = landmark_extractor
        self._step_delay = step_delay

    def collect(
        self,
        num_samples: int,
        on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> list[np.ndarray]:
        """Collect up to num_samples landmark vectors from the camera."""
        samples: list[np.ndarray] = []

        while len(samples) < num_samples:
            frame = self._camera.read()

            if on_frame is not None:
                on_frame(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._hand_detector.detect(rgb_frame)
            hand_vectors = self._landmark_extractor.extract(result)

            if hand_vectors:
                samples.append(hand_vectors[0])

            time.sleep(self._step_delay)

        return samples
