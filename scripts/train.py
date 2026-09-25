"""
CLI tool to train a landmark classification model and export it as a SavedModel.

Usage:
    python scripts/train.py [--dataset PATH] [--output DIR] [--epochs N]
"""

import argparse
import sys
from pathlib import Path

from signvision.config.paths import DATASET_FILE, GESTURE_MODEL_DIR
from signvision.training import LandmarkDataset, LandmarkTrainer, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the gesture classification model"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_FILE,
        help="Dataset file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=GESTURE_MODEL_DIR,
        help="Model output dir",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}. Run scripts/collect_data.py first.")
        return 1

    dataset: LandmarkDataset = load_dataset(args.dataset)

    if len(dataset.label_names) < 2:
        print("At least two gestures are required to train a model.")
        return 1

    print(
        f"Loaded {len(dataset.vectors)} samples across "
        f"{len(dataset.label_names)} labels."
    )
    print(f"Labels: {', '.join(dataset.label_names)}")

    trainer = LandmarkTrainer(epochs=args.epochs)
    model = trainer.train(dataset)
    trainer.save(model, args.output, dataset.label_names)

    print(f"Model exported to {args.output}")
    print(f"Labels written to {args.output.name}_labels.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
