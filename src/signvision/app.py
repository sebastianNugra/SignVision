import logging

from signvision.config.logging_config import configure_logging


class Application:
    """Represents the main SignVision application."""

    def __init__(self) -> None:
        """Creates a new Application instance."""
        self._initialized = False
        self._running = False

    def initialize(self) -> None:
        """Initializes the application resources."""
        if self._initialized:
            raise RuntimeError("Application is already initialized.")

        configure_logging()

        self._initialized = True

    def run(self) -> None:
        """Runs the SignVision application."""
        if not self._initialized:
            raise RuntimeError("Application is not initialized.")

        if self._running:
            raise RuntimeError("Application is already running")

        self._running = True

        try:
            logging.info("SignVision iniciado.")
            # Bucle principal de la app.
        finally:
            self._running = False

    def shutdown(self) -> None:
        """Shuts down the SignVision application."""
        if self._running:
            self._running = False

        if self._initialized:
            self._initialized = False

        logging.info("SignVision shutdown completed.")
