"""
Manage the lifecycle of the application's video source.

Responsible for opening the camera, capturing frames and releasing
the associated resources.
"""

import cv2
import numpy as np


class Camera:
    def __init__(self, device_index: int) -> None:
        """Initialize a camera with the specified device index."""

        self._device_index = device_index
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the connection to the camera."""

        self._capture = cv2.VideoCapture(self._device_index)

        if self._capture is None or not self._capture.isOpened():
            if self._capture is not None:
                self._capture.release()
            self._capture = None
            raise RuntimeError(f"Failed to open camera with index {self._device_index}")

    def read(self) -> np.ndarray:
        """Capture and return a frame from the camera."""

        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError(f"Camera with index {self._device_index} is not open.")

        success, frame = self._capture.read()

        if not success:
            raise RuntimeError(
                f"Failed to capture frame from camera with index {self._device_index}."
            )

        return frame

    def close(self) -> None:
        """Release the camera and its associated resources."""

        if self._capture is not None:
            self._capture.release()
            self._capture = None
