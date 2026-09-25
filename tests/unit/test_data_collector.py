import numpy as np
import pytest

from signvision.training.data_collector import DataCollector


class FakeCamera:
    def __init__(self, fail_after: int | None = None) -> None:
        self._reads = 0
        self._fail_after = fail_after

    def read(self) -> np.ndarray:
        self._reads += 1

        if self._fail_after is not None and self._reads > self._fail_after:
            raise RuntimeError("camera failure")

        return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


class FakeHandDetector:
    def __init__(self, hands_sequence: list[bool]) -> None:
        self._sequence = hands_sequence
        self._calls = 0

    def detect(self, frame: np.ndarray) -> object:
        index = min(self._calls, len(self._sequence) - 1)
        self._calls += 1

        return object() if self._sequence[index] else []

    def close(self) -> None:
        pass


class FakeLandmarkExtractor:
    def extract(self, result: object) -> list[np.ndarray]:
        if isinstance(result, list):
            return []

        return [np.ones((21, 3), dtype=np.float32)]


def _collector(
    hands: list[bool],
    fail_after: int | None = None,
) -> tuple[DataCollector, FakeCamera]:
    camera = FakeCamera(fail_after=fail_after)
    collector = DataCollector(
        camera=camera,
        hand_detector=FakeHandDetector(hands),
        landmark_extractor=FakeLandmarkExtractor(),
        step_delay=0.0,
    )

    return collector, camera


def test_collect_returns_requested_number_of_samples() -> None:
    collector, camera = _collector([True])

    samples = collector.collect(5)

    assert len(samples) == 5
    assert all(sample.shape == (21, 3) for sample in samples)
    assert camera._reads == 5


def test_collect_skips_frames_without_hand() -> None:
    collector, camera = _collector([False, False, True, True, True])

    samples = collector.collect(3)

    assert len(samples) == 3
    assert camera._reads == 5


def test_collect_propagates_camera_failure() -> None:
    collector, camera = _collector([True], fail_after=3)

    with pytest.raises(RuntimeError):
        collector.collect(10)

    assert camera._reads == 4


def test_collect_invokes_on_frame_callback() -> None:
    collector, camera = _collector([True])
    seen = []

    collector.collect(2, on_frame=lambda frame: seen.append(frame))

    assert len(seen) == 2
    assert camera._reads == 2
    assert all(frame.shape == (480, 640, 3) for frame in seen)
