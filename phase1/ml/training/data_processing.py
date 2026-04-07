"""Compatibility layer for legacy imports and commands.

Canonical module moved to `phase1/ml/training/data_processing.py`.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from phase1.ml.training.data_processing import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
