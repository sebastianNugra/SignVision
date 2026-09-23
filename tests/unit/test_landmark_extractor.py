import numpy as np
import pytest
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import landmark

from signvision.vision import LandmarkExtractor

WRIST = (0.5, 0.5, 0.0)
MIDDLE_FINGER_MCP = (0.5, 0.7, 0.0)


def _make_hand(
    mcp_point: tuple[float, float, float] = MIDDLE_FINGER_MCP,
) -> list[landmark.NormalizedLandmark]:
    points = [WRIST] * 21
    points[9] = mcp_point

    return [
        landmark.NormalizedLandmark(x=point[0], y=point[1], z=point[2])
        for point in points
    ]


def _result(
    hands: list[list[landmark.NormalizedLandmark]],
) -> vision.HandLandmarkerResult:
    """Build a HandLandmarkerResult with the given per-hand landmarks."""

    return vision.HandLandmarkerResult(
        handedness=[],
        hand_landmarks=hands,
        hand_world_landmarks=[],
    )


def test_extract_returns_empty_list_for_no_hands() -> None:
    assert LandmarkExtractor().extract(_result([])) == []


def test_extract_returns_one_normalized_vector_per_hand() -> None:
    features = LandmarkExtractor().extract(_result([_make_hand(), _make_hand()]))

    assert len(features) == 2
    assert features[0].shape == (21, 3)


def test_normalize_relocates_and_scales_landmarks() -> None:
    feature = LandmarkExtractor().extract(_result([_make_hand()]))[0]

    assert feature[0] == pytest.approx([0.0, 0.0, 0.0])
    assert feature[9] == pytest.approx([0.0, 1.0, 0.0])
    assert feature[10] == pytest.approx([0.0, 0.0, 0.0])


def test_normalize_preserves_relative_depth() -> None:
    mcp_with_depth = (0.5, 0.7, 0.1)
    feature = LandmarkExtractor().extract(_result([_make_hand(mcp_with_depth)]))[0]

    assert feature[9] == pytest.approx([0.0, 1.0, 0.5])


def test_normalize_raises_on_zero_reference_scale() -> None:
    with pytest.raises(ValueError):
        LandmarkExtractor().extract(_result([_make_hand(WRIST)]))


def test_normalize_returns_numeric_float_array() -> None:
    feature = LandmarkExtractor().extract(_result([_make_hand()]))[0]

    assert feature.dtype == np.float64
