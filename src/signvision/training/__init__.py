"""
Training Module

Responsibility: Manage landmark data collection, dataset persistence,
and AI model training.
"""

from .data_collector import DataCollector
from .dataset import LandmarkDataset, load_dataset, save_dataset
from .trainer import LandmarkTrainer

__all__ = [
    "DataCollector",
    "LandmarkDataset",
    "LandmarkTrainer",
    "load_dataset",
    "save_dataset",
]
