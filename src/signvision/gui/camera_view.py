"""
CameraView

Responsibility: Display the live camera feed within the GUI.
"""

import customtkinter
import cv2
import numpy as np
from PIL import Image


class CameraView(customtkinter.CTkLabel):
    """Renders BGR camera frames as a CustomTkinter image."""

    def __init__(
        self,
        master: customtkinter.CTk,
        width: int = 640,
        height: int = 480,
    ) -> None:
        super().__init__(
            master,
            text="Camera preview",
            width=width,
            height=height,
            corner_radius=8,
            fg_color="gray15",
        )

        self._width = width
        self._height = height
        self._image: customtkinter.CTkImage | None = None

    def update_frame(self, frame: np.ndarray) -> None:
        """Display the given BGR camera frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb).resize(
            (self._width, self._height),
            Image.Resampling.LANCZOS,
        )

        self._image = customtkinter.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=(self._width, self._height),
        )
        self.configure(image=self._image, text="")

    def clear(self) -> None:
        """Remove the current frame and show a placeholder message."""
        self._image = None
        self.configure(image=self._image, text="Camera preview")
