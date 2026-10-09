"""
TextToSpeech

Responsibility: Convert recognized sign language text output into
spoken audio using pyttsx3.
"""

import pyttsx3


class TextToSpeech:
    """
    Converts text into spoken audio through a pyttsx3 engine.

    The engine is created lazily on first use and can be injected for
    testing or to reuse a shared engine across the application.
    """

    def __init__(self, engine: pyttsx3.Engine | None = None) -> None:
        self._engine = engine

    def speak(self, text: str) -> None:
        """Speak the given text, blocking until it finishes."""
        if not text.strip():
            raise ValueError("Cannot speak empty text")

        engine = self._engine or pyttsx3.init()
        self._engine = engine

        engine.say(text)
        engine.runAndWait()

    def stop(self) -> None:
        """Stop any pending speech on the underlying engine."""
        if self._engine is not None:
            self._engine.stop()

    def shutdown(self) -> None:
        """Stop outstanding speech and release the engine resources."""
        self.stop()
