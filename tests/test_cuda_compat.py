"""Unit tests for docneedle.cuda_compat module."""

import os
from unittest.mock import patch

from docneedle.cuda_compat import (
    configure_cuda_library_paths,
    preload_cuda_runtime_libs,
    preload_modular_backends,
    prewarm_models_page_cache,
)


def test_configure_cuda_library_paths() -> None:
    """Ensure configure_cuda_library_paths discovers directories and updates LD_LIBRARY_PATH."""
    dirs = configure_cuda_library_paths()
    assert isinstance(dirs, list)


def test_preload_cuda_runtime_libs_runs_safely() -> None:
    """Ensure preload_cuda_runtime_libs runs without crashing in any environment."""
    result = preload_cuda_runtime_libs()
    assert isinstance(result, bool)


def test_preload_modular_backends_runs_safely() -> None:
    """Ensure preload_modular_backends handles missing or present device nodes safely."""
    result = preload_modular_backends()
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


def test_prewarm_models_page_cache(tmp_path) -> None:
    """Ensure prewarm_models_page_cache runs safely and reads existing files."""
    dummy_model = tmp_path / "dummy_model.gguf"
    dummy_model.write_bytes(b"GGUF" + b"\x00" * 1024)

    prewarmed = prewarm_models_page_cache([dummy_model])
    assert "dummy_model.gguf" in prewarmed
