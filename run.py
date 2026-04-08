"""Convenience launcher for running the Phase 1 backend from project root.

Usage (from phase1/):
    python3 run.py
"""

import os

from app.backend.app import app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
