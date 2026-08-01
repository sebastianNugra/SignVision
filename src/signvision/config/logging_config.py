"""
Logging configuration for SignVision.

Provides the functionality to configure the application's logging system.
"""

import logging

from signvision.config.paths import LOG_FILE, LOGS_DIR
from signvision.config.settings import LOG_LEVEL


def configure_logging() -> None:
    """Configures the application's logging system."""
    log_level = getattr(logging, LOG_LEVEL)

    LOGS_DIR.mkdir(exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()

    if logger.handlers:
        return

    logger.setLevel(log_level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
