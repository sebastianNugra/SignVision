"""
ResultsPanel

Responsibility: Display recognized gestures and translation
results to the user.
"""

from collections.abc import Callable

import customtkinter


class ResultsPanel(customtkinter.CTkFrame):
    """
    Shows the current translation and last recognized word, and exposes
    the pipeline controls (start/stop, reset and speak toggle).
    """

    def __init__(
        self,
        master: customtkinter.CTk,
        *,
        on_toggle: Callable[[], None],
        on_reset: Callable[[], None],
        on_speak_change: Callable[[], None],
    ) -> None:
        super().__init__(master, width=260)

        title = customtkinter.CTkLabel(
            self,
            text="Translation",
            font=customtkinter.CTkFont(size=18, weight="bold"),
        )
        title.pack(padx=12, pady=(12, 4), anchor="w")

        self._translation_var = customtkinter.StringVar(value="")
        translation_label = customtkinter.CTkLabel(
            self,
            textvariable=self._translation_var,
            wraplength=220,
            justify="left",
            anchor="nw",
        )
        translation_label.pack(padx=12, pady=4, fill="x")

        word_heading = customtkinter.CTkLabel(
            self,
            text="Last word",
            font=customtkinter.CTkFont(size=14, weight="bold"),
        )
        word_heading.pack(padx=12, pady=(12, 4), anchor="w")

        self._word_var = customtkinter.StringVar(value="")
        word_label = customtkinter.CTkLabel(
            self,
            textvariable=self._word_var,
            anchor="w",
        )
        word_label.pack(padx=12, pady=4, fill="x")

        self.start_button = customtkinter.CTkButton(
            self,
            text="Start",
            command=on_toggle,
        )
        self.start_button.pack(padx=12, pady=(16, 8), fill="x")

        self.reset_button = customtkinter.CTkButton(
            self,
            text="Reset",
            command=on_reset,
            fg_color="transparent",
            border_width=1,
        )
        self.reset_button.pack(padx=12, pady=8, fill="x")

        self._speak_var = customtkinter.BooleanVar(value=True)
        self.speak_checkbox = customtkinter.CTkCheckBox(
            self,
            text="Speak aloud",
            variable=self._speak_var,
            command=on_speak_change,
        )
        self.speak_checkbox.pack(padx=12, pady=8, anchor="w")

    def set_translation(self, text: str) -> None:
        """Update the displayed translation sentence."""
        self._translation_var.set(text)

    def set_last_word(self, word: str) -> None:
        """Update the displayed last recognized word."""
        self._word_var.set(word)

    def set_running(self, running: bool) -> None:
        """Update the start/stop button label to reflect the state."""
        self.start_button.configure(text="Stop" if running else "Start")

    @property
    def speak_enabled(self) -> bool:
        """Whether the text-to-speech output is enabled."""
        return bool(self._speak_var.get())
