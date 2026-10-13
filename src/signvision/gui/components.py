"""
Components

Responsibility: Define reusable GUI widgets and custom components
shared across the interface.
"""

import customtkinter


class StatusBar(customtkinter.CTkFrame):
    """Displays a single status message at the bottom of a window."""

    def __init__(
        self,
        master: customtkinter.CTk,
        text: str = "Ready",
    ) -> None:
        super().__init__(master)

        self._status_var = customtkinter.StringVar(value=text)
        label = customtkinter.CTkLabel(
            self,
            textvariable=self._status_var,
            anchor="w",
        )
        label.pack(fill="x", padx=8, pady=4)

    def set_status(self, message: str) -> None:
        """Update the displayed status message."""
        self._status_var.set(message)
