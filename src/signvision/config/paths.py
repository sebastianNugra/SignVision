"""
Application paths configuration.

Defines filesystem locations used by SignVision.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOGS_DIR = PROJECT_ROOT / "logs"

LOG_FILE = LOGS_DIR / "signvision.log"

TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"

HAND_LANDMARKER_MODEL_PATH = TRAINED_MODELS_DIR / "hand_landmarker.task"
