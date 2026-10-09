"""
CLI tool to translate live sign gestures into text.

Usage:
    python scripts/translate_live.py [--camera INDEX] [--model-dir PATH] [--speak]

Hold a pose to confirm a word. Press R to reset, ESC to exit.
"""

import argparse
import sys
from pathlib import Path

import cv2

from signvision.camera import Camera
from signvision.config.paths import GESTURE_MODEL_DIR
from signvision.models import GestureClassifier, LabelMap, ModelLoader
from signvision.services import TextToSpeech, TranslationService
from signvision.vision import HandDetector, LandmarkExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate live sign gestures to text")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (e.g. 1 if 0 is unavailable)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=GESTURE_MODEL_DIR,
        help="Trained gesture model directory",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak each confirmed word aloud",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    labels_path = args.model_dir.parent / f"{args.model_dir.name}_labels.json"
    label_map = LabelMap.from_path(labels_path)

    classifier = GestureClassifier(
        model_loader=ModelLoader(args.model_dir),
        label_map=label_map,
    )
    classifier.load()

    service = TranslationService(classifier)

    text_to_speech = TextToSpeech() if args.speak else None

    camera = Camera(args.camera)
    camera.open()
    hand_detector = HandDetector()
    hand_detector.open()
    extractor = LandmarkExtractor()

    print("Translating gestures. Hold a pose to confirm a word.")
    print("Press R to reset, ESC to exit.")
    print(f"Labels: {', '.join(label_map.labels)}")

    try:
        while True:
            frame = camera.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vectors = extractor.extract(hand_detector.detect(rgb))

            if vectors:
                result = service.process(vectors[0])
                if result is not None:
                    print(
                        f"Word: {result.label} ({result.confidence:.2f}) "
                        f"-> {service.text}"
                    )
                    if text_to_speech is not None:
                        text_to_speech.speak(result.label)

            cv2.putText(
                frame,
                service.text or "No words yet",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("SignVision - Live Translation", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
                service.reset()
                print("Translation reset.")
    except KeyboardInterrupt:
        pass
    finally:
        if text_to_speech is not None:
            text_to_speech.shutdown()
        hand_detector.close()
        camera.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
