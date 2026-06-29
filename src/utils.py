"""Utility functions used across the project.

- configure_logging: standard logging configuration
- ensure_model_path: helper to build default model path
- read_text_file: read small text files robustly
"""
from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Optional

from . import config


def configure_logging(level: int | str = logging.INFO, logger: Optional[Logger] = None) -> None:
    """
    Configure root logger with a simple format. Safe to call multiple times.

    Args:
        level: Logging level or numeric value.
        logger: If provided, configure this specific logger instead of root.
    """
    root = logger if logger is not None else logging.getLogger()
    if root.handlers:
        # Already configured; adjust level only
        root.setLevel(level)
        return

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)


def ensure_model_path(model_dir: Path | None = None) -> Path:
    """
    Return a Path to the default model file within model_dir or config.SAVED_MODEL_DIR.

    Creates the directory if needed.
    """
    if model_dir is None:
        model_dir = config.SAVED_MODEL_DIR
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / config.MODEL_FILENAME


def read_text_file(path: Path) -> str:
    """
    Read a small text file and return its contents as a string.

    Args:
        path: Path to the file.

    Returns:
        Contents of the file as str.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")
