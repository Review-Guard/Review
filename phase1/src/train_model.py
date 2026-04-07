"""Compatibility layer for legacy training entrypoint.

Canonical module moved to `phase1/ml/training/train_model.py`.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml.training.train_model import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
