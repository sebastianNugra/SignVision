"""
MainWindow

Responsibility: Define the main application window layout
and top-level GUI structure.
"""

import logging
import queue
import threading
from pathlib import Path

import customtkinter
import cv2
import numpy as np

from signvision.camera import Camera
from signvision.config.paths import GESTURE_MODEL_DIR
from signvision.gui.camera_view import CameraView
from signvision.gui.components import StatusBar
from signvision.gui.results_panel import ResultsPanel
from signvision.models import GestureClassifier, LabelMap, ModelLoader
from signvision.services import TextToSpeech, TranslationService
from signvision.vision import HandDetector, LandmarkExtractor

_LOG = logging.getLogger(__name__)


class MainWindow(customtkinter.CTk):
    """Main application window coordinating the live translation pipeline."""

    def __init__(
        self,
        camera_index: int = 0,
        model_dir: str | Path = GESTURE_MODEL_DIR,
    ) -> None:
        super().__init__()

        self.title("SignVision AI")
        self.geometry("980x640")

        self._camera_index = camera_index
        self._model_dir = Path(model_dir)

        self._stop_event = threading.Event()
        self._pipeline_thread: threading.Thread | None = None
        self._service: TranslationService | None = None
        self._updates: queue.Queue[tuple[np.ndarray | None, str, str, str]] = (
            queue.Queue()
        )
        self._text_to_speech = TextToSpeech()

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_updates()

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.camera_view = CameraView(self)
        self.camera_view.grid(
            row=0,
            column=0,
            padx=(12, 6),
            pady=12,
            sticky="nsew",
        )

        self.results_panel = ResultsPanel(
            self,
            on_toggle=self._on_toggle,
            on_reset=self._on_reset,
            on_speak_change=self._on_speak_change,
        )
        self.results_panel.grid(row=0, column=1, padx=(6, 12), pady=12, sticky="ns")

        self.status_bar = StatusBar(self)
        self.status_bar.grid(
            row=1,
            column=0,
            columnspan=2,
            padx=12,
            pady=(0, 8),
            sticky="ew",
        )

    def _on_toggle(self) -> None:
        if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def _start_pipeline(self) -> None:
        self._stop_event = threading.Event()
        self._pipeline_thread = threading.Thread(
            target=self._pipeline_loop,
            daemon=True,
        )
        self._pipeline_thread.start()

        self.results_panel.set_running(True)
        self.status_bar.set_status("Pipeline running")

    def _stop_pipeline(self) -> None:
        self._stop_event.set()

        if self._pipeline_thread is not None:
            self._pipeline_thread.join(timeout=5)
        self._pipeline_thread = None

        self.results_panel.set_running(False)
        self.status_bar.set_status("Pipeline stopped")

    def _on_reset(self) -> None:
        if self._service is not None:
            self._service.reset()
        self.results_panel.set_translation("")
        self.results_panel.set_last_word("")

    def _on_speak_change(self) -> None:
        state = "enabled" if self.results_panel.speak_enabled else "disabled"
        self.status_bar.set_status(f"Text-to-speech {state}")

    def _pipeline_loop(self) -> None:
        camera = Camera(self._camera_index)
        detector = HandDetector()
        extractor = LandmarkExtractor()

        labels_path = self._model_dir.parent / f"{self._model_dir.name}_labels.json"
        classifier = GestureClassifier(
            model_loader=ModelLoader(self._model_dir),
            label_map=LabelMap.from_path(labels_path),
        )
        service = TranslationService(classifier)
        self._service = service

        try:
            classifier.load()
            camera.open()
            detector.open()
        except (
            Exception
        ) as exc:  # noqa: BLE001 - pipeline failure must not kill the app
            _LOG.exception("Failed to start the translation pipeline: %s", exc)
            self._updates.put((None, "", "", f"Error: {exc}"))
            return

        try:
            while not self._stop_event.is_set():
                frame = camera.read()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                vectors = extractor.extract(detector.detect(rgb))

                word = ""
                if vectors:
                    result = service.process(vectors[0])
                    if result is not None:
                        word = result.label
                        if self.results_panel.speak_enabled:
                            self._text_to_speech.speak(result.label)

                status = f"Last word: {word}" if word else "Listening..."
                self._updates.put((frame, service.text, word, status))
        except (
            Exception
        ) as exc:  # noqa: BLE001 - pipeline failure must not kill the app
            _LOG.exception("Translation pipeline failed: %s", exc)
            self._updates.put((None, service.text, "", f"Error: {exc}"))
        finally:
            detector.close()
            camera.close()

    def _poll_updates(self) -> None:
        try:
            while True:
                frame, text, word, status = self._updates.get_nowait()

                if frame is not None:
                    self.camera_view.update_frame(frame)

                self.results_panel.set_translation(text)

                if word:
                    self.results_panel.set_last_word(word)

                if status:
                    self.status_bar.set_status(status)
        except queue.Empty:
            pass

        thread = self._pipeline_thread
        if thread is not None and not thread.is_alive():
            self._pipeline_thread = None
            self.results_panel.set_running(False)

        self.after(16, self._poll_updates)

    def _on_close(self) -> None:
        self._stop_event.set()

        if self._pipeline_thread is not None:
            self._pipeline_thread.join(timeout=5)
        self._pipeline_thread = None

        self._text_to_speech.shutdown()
        self.destroy()
