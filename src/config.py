"""Configuration constants for the Image-Classification project.

Contains dataset parameters, class names, default paths, and training-related constants.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

# Image dimensions for CIFAR-10
IMG_HEIGHT: int = 32
IMG_WIDTH: int = 32
IMG_CHANNELS: int = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# CIFAR-10 class names
CLASS_NAMES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
NUM_CLASSES: int = len(CLASS_NAMES)

# Project directories
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
SAVED_MODEL_DIR: Path = PROJECT_ROOT / "saved_model"
IMAGES_DIR: Path = PROJECT_ROOT / "images"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Training defaults
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_EPOCHS: int = 20
RANDOM_SEED: int = 42

# File formats
MODEL_FILENAME: str = "cifar10_classifier.keras"

# Ensure directories exist (safe to call at import time)
for _p in (DATASET_DIR, SAVED_MODEL_DIR, IMAGES_DIR, NOTEBOOKS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

