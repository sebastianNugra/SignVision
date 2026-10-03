"""
CLI tool to classify hand gestures live from the camera.

Usage:
    python scripts/classify_live.py [--camera INDEX] [--model-dir PATH]

Press ESC in the preview window or Ctrl+C in the terminal to exit.
"""

import argparse
import sys
from pathlib import Path

import cv2

from signvision.camera import Camera
from signvision.config.paths import GESTURE_MODEL_DIR
from signvision.models import GestureClassifier, LabelMap, ModelLoader
from signvision.vision import HandDetector, LandmarkExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify hand gestures live from the camera"
    )
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

    camera = Camera(args.camera)
    camera.open()
    hand_detector = HandDetector()
    hand_detector.open()
    extractor = LandmarkExtractor()

    print("Classifying gestures. Press ESC to exit.")
    print(f"Labels: {', '.join(label_map.labels)}")

    try:
        while True:
            frame = camera.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for vector in extractor.extract(hand_detector.detect(rgb)):
                label, confidence = classifier.classify(vector)
                text = f"{label}: {confidence:.2f}"
                print(text)
                cv2.putText(
                    frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("SignVision - Live Classification", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        hand_detector.close()
        camera.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
