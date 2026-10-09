import pytest

from signvision.services import TextToSpeech


class _FakeEngine:
    def __init__(self) -> None:
        self.said: list[str] = []
        self.run_times = 0
        self.stop_times = 0

    def say(self, text: str) -> None:
        self.said.append(text)

    def runAndWait(self) -> None:
        self.run_times += 1

    def stop(self) -> None:
        self.stop_times += 1


def test_speak_uses_engine() -> None:
    engine = _FakeEngine()
    text_to_speech = TextToSpeech(engine)

    text_to_speech.speak("Hola")

    assert engine.said == ["Hola"]
    assert engine.run_times == 1


def test_speak_empty_text_raises() -> None:
    text_to_speech = TextToSpeech(_FakeEngine())

    with pytest.raises(ValueError, match="empty text"):
        text_to_speech.speak("")


def test_speak_empty_whitespace_raises() -> None:
    text_to_speech = TextToSpeech(_FakeEngine())

    with pytest.raises(ValueError, match="empty text"):
        text_to_speech.speak("   ")


def test_shutdown_stops_engine() -> None:
    engine = _FakeEngine()
    text_to_speech = TextToSpeech(engine)

    text_to_speech.shutdown()

    assert engine.stop_times == 1


def test_stop_releases_when_no_engine() -> None:
    text_to_speech = TextToSpeech()

    text_to_speech.stop()
    text_to_speech.shutdown()
