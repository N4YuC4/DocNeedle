"""Pytest configuration and test environment initialization."""

import os

# Prevent app.py imports from triggering slow SHA256 checksums or downloads in tests
os.environ["SKIP_MODEL_BOOTSTRAP"] = "1"

# Preload CUDA runtime libraries into global symbol space before tests import llama_cpp
try:
    from docneedle.cuda_compat import preload_cuda_runtime_libs

    preload_cuda_runtime_libs()
except Exception:
    pass
