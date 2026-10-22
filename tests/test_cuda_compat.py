"""Unit tests for docneedle.cuda_compat module."""

import os
from unittest.mock import patch

from docneedle.cuda_compat import preload_cuda_runtime_libs


def test_preload_cuda_runtime_libs_runs_safely() -> None:
    """Ensure preload_cuda_runtime_libs runs without crashing in any environment."""
    result = preload_cuda_runtime_libs()
    assert isinstance(result, bool)


def test_preload_cuda_runtime_libs_idempotent() -> None:
    """Ensure subsequent invocations return cached status."""
    first = preload_cuda_runtime_libs()
    second = preload_cuda_runtime_libs()
    assert first == second


def test_preload_cuda_runtime_with_mock_paths() -> None:
    """Ensure library search handles non-existent or empty directories gracefully."""
    with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/tmp/fake_cuda_dir"}):
        result = preload_cuda_runtime_libs()
        assert isinstance(result, bool)
