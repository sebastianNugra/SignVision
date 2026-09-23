from pathlib import Path

import numpy as np
import pytest

from signvision.vision import HandDetector


def test_hand_detector_initial_state() -> None:
    detector = HandDetector()

    assert detector._landmarker is None
    assert detector._num_hands == 2
    assert detector._min_detection_confidence == 0.5
    assert detector._min_hand_presence_confidence == 0.5
    assert detector._min_tracking_confidence == 0.5


def test_hand_detector_custom_configuration() -> None:
    detector = HandDetector(
        num_hands=1,
        min_detection_confidence=0.7,
        min_hand_presence_confidence=0.8,
        min_tracking_confidence=0.9,
    )

    assert detector._num_hands == 1
    assert detector._min_detection_confidence == 0.7
    assert detector._min_hand_presence_confidence == 0.8
    assert detector._min_tracking_confidence == 0.9


def test_hand_detector_detect_raises_when_not_open() -> None:
    detector = HandDetector()

    with pytest.raises(RuntimeError):
        detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))


def test_hand_detector_open_raises_when_model_missing(tmp_path: Path) -> None:
    detector = HandDetector(model_path=tmp_path / "missing.task")

    with pytest.raises(RuntimeError):
        detector.open()


def test_hand_detector_close_is_idempotent() -> None:
    detector = HandDetector()
    detector.close()
    detector.close()

    assert detector._landmarker is None
