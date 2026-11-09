"""Pytest configuration and test environment initialization."""

import os

# Prevent app.py imports from triggering slow SHA256 checksums or downloads in tests
os.environ["SKIP_MODEL_BOOTSTRAP"] = "1"
