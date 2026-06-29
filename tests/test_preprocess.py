"""Pytest unit tests for preprocessing utilities."""
from __future__ import annotations

import numpy as np
from src import config
from src.preprocess import preprocess_for_model


def test_preprocess_for_model_shape_and_dtype() -> None:
    # Create a dummy image with expected shape
    img = np.random.rand(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS).astype("float32")
    batch = preprocess_for_model(img)
    assert batch.shape == (1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    assert batch.dtype == "float32"
