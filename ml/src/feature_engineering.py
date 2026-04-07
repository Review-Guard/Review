"""Compatibility layer for legacy imports.

Canonical module moved to `phase1/ml/training/feature_engineering.py`.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from feature_engineering import *  # noqa: F401,F403
