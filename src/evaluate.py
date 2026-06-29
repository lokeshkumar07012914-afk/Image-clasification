"""Evaluation utilities for the CIFAR-10 classifier.

Functions:
- evaluate_model: Load a saved model (.keras), run predictions on CIFAR-10 test set,
  compute accuracy, precision, recall, f1-score, classification report, and confusion matrix.
- _plot_confusion_matrix: internal helper to draw and save the confusion matrix heatmap.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from . import config, data_loader, utils

LOGGER = logging.getLogger(__name__)


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    """Plot and save the confusion matrix to out_path."""
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 8))
        im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar(im, fraction=0.046, pad=0.04)

        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45, ha="right")
        plt.yticks(tick_marks, labels)

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        LOGGER.info("Saved confusion matrix to %s", out_path)
    except Exception:
        LOGGER.exception("Failed to plot confusion matrix")
        raise


def evaluate_model(model_path: Path, batch_size: int = 64, output_dir: Path | None = None) -> Dict[str, float]:
    """
    Evaluate a saved .keras model on CIFAR-10 test set and produce metrics and plots.

    Args:
        model_path: Path to the saved model (.keras)
        batch_size: Batch size for prediction
        output_dir: Directory to write plots and report. Defaults to images/ in project root.

    Returns:
        A dictionary with keys: accuracy, precision_macro, recall_macro, f1_macro
    """
    try:
        utils.configure_logging()

        if output_dir is None:
            output_dir = config.IMAGES_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Loading model from %s", model_path)
        model = tf.keras.models.load_model(str(model_path))

        LOGGER.info("Loading CIFAR-10 test data")
        _, _, test_images, test_labels = data_loader.load_cifar10()
        y_true = test_labels.reshape(-1)

        LOGGER.info("Predicting on test set (batch_size=%s)", batch_size)
        preds_prob = model.predict(test_images, batch_size=batch_size, verbose=1)
        y_pred = np.argmax(preds_prob, axis=1)

        accuracy = float(accuracy_score(y_true, y_pred))
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES)
        LOGGER.info("Classification report:\n%s", report)

        cm = confusion_matrix(y_true, y_pred)

        # Save classification report
        report_path = output_dir / "classification_report.txt"
        with report_path.open("w", encoding="utf-8") as fh:
            fh.write(f"Accuracy: {accuracy:.6f}\n")
            fh.write(f"Precision (macro): {precision_macro:.6f}\n")
            fh.write(f"Recall (macro): {recall_macro:.6f}\n")
            fh.write(f"F1 (macro): {f1_macro:.6f}\n\n")
            fh.write("Classification Report:\n")
            fh.write(report)
        LOGGER.info("Saved classification report to %s", report_path)

        # Plot confusion matrix
        cm_path = output_dir / "confusion_matrix.png"
        _plot_confusion_matrix(cm, config.CLASS_NAMES, cm_path)

        metrics = {
            "accuracy": accuracy,
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
        }

        LOGGER.info("Evaluation metrics: %s", metrics)
        return metrics

    except Exception:
        LOGGER.exception("Model evaluation failed")
        raise
