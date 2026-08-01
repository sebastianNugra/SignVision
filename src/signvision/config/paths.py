"""
Application paths configuration.

Defines filesystem locations used by SignVision.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOGS_DIR = PROJECT_ROOT / "logs"

LOG_FILE = LOGS_DIR / "signvision.log"
