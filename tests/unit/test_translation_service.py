import numpy as np
import pytest

from signvision.services import GestureResult, TranslationService


class _FakeClassifier:
    """Replies with a scripted sequence of (label, confidence) pairs."""

    def __init__(self, responses: list[tuple[str, float]]) -> None:
        self._responses = responses
        self._index = 0

    def classify(self, vector) -> tuple[str, float]:
        label, confidence = self._responses[self._index]
        self._index += 1
        return label, confidence


def _service(responses: list[tuple[str, float]], **kwargs) -> TranslationService:
    return TranslationService(_FakeClassifier(responses), **kwargs)


def _vector() -> np.ndarray:
    return np.zeros((21, 3), dtype=np.float32)


def test_word_confirmed_after_hold_frames() -> None:
    service = _service([("A", 0.99)] * 6)

    assert service.process(_vector()) is None
    assert service.process(_vector()) is None

    result = service.process(_vector())

    assert isinstance(result, GestureResult)
    assert result.label == "A"
    assert result.confidence == 0.99
    assert service.text == "A"


def test_below_threshold_never_confirms() -> None:
    service = _service([("A", 0.5)] * 6)

    for _ in range(6):
        assert service.process(_vector()) is None

    assert service.text == ""


def test_consecutive_words_appended() -> None:
    service = _service([("A", 0.99)] * 3 + [("B", 0.99)] * 3)

    assert service.process(_vector()) is None
    assert service.process(_vector()) is None

    first = service.process(_vector())
    assert first is not None
    assert first.label == "A"

    assert service.process(_vector()) is None
    assert service.process(_vector()) is None

    second = service.process(_vector())
    assert second is not None
    assert second.label == "B"
    assert service.text == "A B"


def test_continued_hold_not_duplicated() -> None:
    service = _service([("A", 0.99)] * 6)

    assert service.process(_vector()) is None
    assert service.process(_vector()) is None
    assert service.process(_vector()) is not None
    assert service.process(_vector()) is None
    assert service.process(_vector()) is None
    assert service.process(_vector()) is None

    assert service.text == "A"


def test_low_confidence_resets_streak() -> None:
    service = _service(
        [("A", 0.99), ("A", 0.99), ("A", 0.4), ("A", 0.99), ("A", 0.99), ("A", 0.99)]
    )

    for _ in range(6):
        service.process(_vector())

    assert service.words == ["A"]


def test_reset_clears_translation() -> None:
    service = _service([("A", 0.99)] * 3 + [("B", 0.99)] * 3)

    service.process(_vector())
    service.process(_vector())
    service.process(_vector())
    service.process(_vector())
    service.process(_vector())
    service.process(_vector())

    assert service.text == "A B"

    service.reset()

    assert service.text == ""
    assert service.words == []


def test_min_confidence_validated() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        TranslationService(_FakeClassifier([]), min_confidence=0.0)

    with pytest.raises(ValueError, match="min_confidence"):
        TranslationService(_FakeClassifier([]), min_confidence=1.2)


def test_confirm_frames_validated() -> None:
    with pytest.raises(ValueError, match="confirm_frames"):
        TranslationService(_FakeClassifier([]), confirm_frames=0)
