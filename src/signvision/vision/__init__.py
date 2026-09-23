"""
Vision Module

Responsibility: Process video frames using MediaPipe for hand landmark
detection and extraction.
"""

from .hand_detector import HandDetector

__all__ = ["HandDetector"]
