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

        self._initialized = True

    def run(self) -> None:
        """Runs the SignVision application."""
        if not self._initialized:
            raise RuntimeError("Application is not initialized.")

        if self._running:
            raise RuntimeError("Application is already running")

        self._running = True

        try:
            print("SignVision iniciado.")
            # Bucle principal de la app.
        finally:
            self._running = False

    def shutdown(self) -> None:
        """Shuts down the SignVision application."""
        if not self._initialized:
            raise RuntimeError("Application is not initialized.")

        if self._running:
            raise RuntimeError("Cannot shut down while the application is running.")

        self._initialized = False
