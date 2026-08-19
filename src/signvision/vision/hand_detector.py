import mediapipe as mp


class HandDetector:
    """Responsible for managing hand detection."""

    def __init__(self) -> None:
        self._hands = mp.solutions.hands.Hands()
