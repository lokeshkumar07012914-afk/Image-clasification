"""src.predict

Utilities to load a saved model and predict a single local image.

Functions:
- predict_image: Load model, preprocess image, run prediction, and print/show confidence scores.
- _format_probabilities: helper to format all class probabilities as a list of dicts.

This module is designed for both programmatic use and CLI invocation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf

from . import config, preprocess, utils

LOGGER = logging.getLogger(__name__)


def _format_probabilities(probs: np.ndarray) -> List[Dict[str, float]]:
    """
    Convert a 1D array of probabilities into a list of dicts with class name and probability.

    Args:
        probs: 1D numpy array of length NUM_CLASSES with probabilities summing to ~1.

    Returns:
        List of {"class": class_name, "probability": prob} sorted by probability desc.
    """
    try:
        probs = np.asarray(probs).ravel()
        if probs.size != config.NUM_CLASSES:
            raise ValueError("Probability array length does not match number of classes")

        idx = np.argsort(probs)[::-1]
        return [{"class": config.CLASS_NAMES[i], "probability": float(probs[i])} for i in idx]
    except Exception:
        LOGGER.exception("Failed to format probabilities")
        raise


def predict_image(model_path: Path, image_path: Path, top_k: int = 5) -> Dict[str, object]:
    """
    Load a saved model and run prediction on a single local image.

    Args:
        model_path: Path to the saved .keras model file.
        image_path: Path to the local image to predict.
        top_k: Number of top predictions to return.

    Returns:
        A dictionary with keys:
            - predicted_class: str
            - predicted_index: int
            - confidence: float
            - top_k: list of (class, prob)
            - all_probabilities: list of {"class", "probability"}
    """
    utils.configure_logging()
    try:
        model_path = Path(model_path)
        image_path = Path(image_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        LOGGER.info("Loading model from %s", model_path)
        model = tf.keras.models.load_model(str(model_path))

        LOGGER.info("Loading and preprocessing image %s", image_path)
        img = preprocess.load_image(image_path, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH))
        batch = preprocess.preprocess_for_model(img)

        LOGGER.info("Running prediction")
        preds = model.predict(batch, verbose=0)
        probs = preds.ravel()
        predicted_idx = int(np.argmax(probs))
        predicted_class = config.CLASS_NAMES[predicted_idx]
        confidence = float(probs[predicted_idx])

        all_probs = _format_probabilities(probs)
        topk = [(p["class"], p["probability"]) for p in all_probs[:top_k]]

        # Print friendly output
        LOGGER.info("Predicted: %s (index=%d) with confidence=%.4f", predicted_class, predicted_idx, confidence)
        print(f"Predicted class: {predicted_class} (index={predicted_idx})")
        print(f"Confidence: {confidence:.4%}")
        print()
        print(f"Top {top_k} predictions:")
        for cls, score in topk:
            print(f"  - {cls}: {score:.4%}")

        return {
            "predicted_class": predicted_class,
            "predicted_index": predicted_idx,
            "confidence": confidence,
            "top_k": topk,
            "all_probabilities": all_probs,
        }

    except Exception:
        LOGGER.exception("Prediction failed")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict a single image using a saved CIFAR-10 model.")
    parser.add_argument("model_path", type=str, help="Path to the saved .keras model")
    parser.add_argument("image_path", type=str, help="Path to the image file to predict")
    parser.add_argument("--top-k", type=int, default=5, help="Show top-K predictions")
    args = parser.parse_args()

    try:
        predict_image(Path(args.model_path), Path(args.image_path), top_k=args.top_k)
    except Exception as exc:
        print(f"Prediction failed: {exc}")
        raise
