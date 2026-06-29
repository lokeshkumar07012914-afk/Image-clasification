"""Image preprocessing helpers.

Includes utilities to load a single image from disk, preprocess it for the model,
and visualize or save images when needed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf

from . import config

LOGGER = logging.getLogger(__name__)


def load_image(path: Path, target_size: Tuple[int, int] | None = None) -> np.ndarray:
    """
    Load an image from a file and return as a numpy array scaled to [0, 1].

    Args:
        path: Path to the image file.
        target_size: (height, width). If None, uses config.IMG_HEIGHT/WIDTH.

    Returns:
        image: np.ndarray shape (H, W, C), dtype float32
    """
    try:
        if target_size is None:
            target_size = (config.IMG_HEIGHT, config.IMG_WIDTH)

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Read file and decode with TensorFlow to ensure consistent channels handling
        img_bytes = tf.io.read_file(str(path))
        img = tf.image.decode_image(img_bytes, channels=config.IMG_CHANNELS, expand_animations=False)
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0
        arr = img.numpy()
        LOGGER.debug("Loaded image %s with shape %s", path, arr.shape)
        return arr
    except Exception:
        LOGGER.exception("Failed to load image: %s", path)
        raise


def preprocess_for_model(img: np.ndarray) -> np.ndarray:
    """
    Given an image array (H, W, C) scaled in [0,1], prepare a batch of shape (1, H, W, C).

    Args:
        img: np.ndarray image with dtype float32 in [0,1].

    Returns:
        batch: np.ndarray shape (1, H, W, C)
    """
    try:
        if not isinstance(img, np.ndarray):
            raise TypeError("img must be a numpy array")
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        batch = np.expand_dims(img, axis=0)
        return batch
    except Exception:
        LOGGER.exception("Failed to preprocess image for model")
        raise


def decode_predictions(scores: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Decode a list/array of scores (probabilities) into human-readable class names and confidences.

    Args:
        scores: Iterable of floats of length config.NUM_CLASSES representing probabilities.
        top_k: How many top classes to return.

    Returns:
        List of tuples (class_name, probability) sorted by probability desc.
    """
    try:
        probs = np.asarray(scores, dtype=float)
        if probs.ndim != 1 or probs.size != config.NUM_CLASSES:
            raise ValueError("scores must be a 1D array with length NUM_CLASSES")

        idx = np.argsort(probs)[::-1][:top_k]
        results = [(config.CLASS_NAMES[i], float(probs[i])) for i in idx]
        return results
    except Exception:
        LOGGER.exception("Failed to decode predictions")
        raise
