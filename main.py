"""Top-level CLI for the Image Classification project.

Allows training, evaluating, and predicting using the modular src package.
"""
from typing import Optional
import argparse
import logging
import sys

from src.utils import configure_logging
from src.train import train
from src.evaluate import evaluate_model
from src.predict import predict_image

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image Classification CLI")
    sub = parser.add_subparsers(dest="command