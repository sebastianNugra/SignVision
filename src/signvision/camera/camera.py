"""
Manage the lifecycle of the application's video source.

Responsible for opening the camera, capturing frames and releasing
the associated resources.
"""

import cv2


class Camera:
    def __init__(self, device_index: int) -> None:
        """Creates a new Camera instance."""
        self._device_index = device_index
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        # Initialize the camera connection
        self._capture = cv2.VideoCapture(self._device_index)

    def read(self) -> None:
        # Capture a frame from the camera
        pass

    def close(self) -> None:
        # Release the camera resources
        pass
