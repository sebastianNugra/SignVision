"""
CLI tool to collect landmark samples for one gesture from the live camera.

Usage:
    python scripts/collect_data.py <gesture> <num_samples> [--output PATH]

Samples are appended to the dataset file (default: trained_models/
sign_language_dataset.npz).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from signvision.camera import Camera
from signvision.config.paths import DATASET_FILE
from signvision.training import (
    DataCollector,
    LandmarkDataset,
    load_dataset,
    save_dataset,
)
from signvision.vision import HandDetector, LandmarkExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect landmark samples for a gesture"
    )
    parser.add_argument("gesture", type=str, help="Gesture label to record")
    parser.add_argument("num_samples", type=int, help="Number of samples to capture")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_FILE,
        help="Dataset file path",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    camera = Camera(0)
    camera.open()
    hand_detector = HandDetector()
    hand_detector.open()
    collector = DataCollector(camera, hand_detector, LandmarkExtractor())

    print(
        f"Recording {args.num_samples} samples for "
        f"'{args.gesture}'. Hold the pose..."
    )

    try:
        vectors = collector.collect(args.num_samples)
    finally:
        hand_detector.close()
        camera.close()

    vector_array = np.stack(vectors)

    existing: LandmarkDataset | None = None

    if args.output.exists():
        existing = load_dataset(args.output)
        label_names = list(existing.label_names)
        label = label_names.index(args.gesture) if args.gesture in label_names else None
        if label is None:
            label = len(label_names)
            label_names.append(args.gesture)
        labels = np.concatenate(
            [existing.labels, np.full(len(vectors), label, dtype=np.int32)]
        )
        vectors_all = np.concatenate([existing.vectors, vector_array])
    else:
        label_names = [args.gesture]
        labels = np.zeros(len(vectors), dtype=np.int32)
        vectors_all = vector_array

    dataset = LandmarkDataset(
        vectors=vectors_all,
        labels=labels,
        label_names=label_names,
    )
    save_dataset(dataset, args.output)

    print(f"Captured {len(vectors)} samples for '{args.gesture}'.")
    print(
        f"Dataset now has {len(dataset.vectors)} samples "
        f"across {len(label_names)} labels."
    )
    print(f"Saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
