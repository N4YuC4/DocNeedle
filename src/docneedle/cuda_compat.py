"""CUDA dynamic library resolution and preloading for llama-cpp-python.

Ensures that NVIDIA CUDA runtime and cuBLAS shared libraries (such as libcudart.so.12,
libcublasLt.so.12, libcublas.so.12) packaged in Python site-packages (e.g. nvidia-*-cu12)
are loaded into the global symbol space with RTLD_GLOBAL before llama_cpp attempts to
load libllama.so. This prevents 'libcudart.so.12: cannot open shared object file' errors
in containerized and serverless environments like Hugging Face Spaces ZeroGPU.
"""

import ctypes
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ZeroGPU decorator compatibility (applied before CUDA imports)
try:
    import spaces  # type: ignore

    gpu_decorator = spaces.GPU
except Exception:

    def gpu_decorator(*args: Any, **kwargs: Any) -> Any:
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


_cuda_preloaded = False


def preload_cuda_runtime_libs() -> bool:
    """Preload NVIDIA CUDA runtime and cuBLAS shared libraries with RTLD_GLOBAL.

    Returns:
        bool: True if at least one CUDA runtime library was successfully preloaded,
              False otherwise.
    """
    global _cuda_preloaded
    if _cuda_preloaded:
        return True

    search_dirs: list[Path] = []

    # 1. Directly check installed nvidia namespace packages if present
    for mod_name in (
        "nvidia.cublas",
        "nvidia.cuda_runtime",
        "nvidia.cudnn",
        "nvidia.curand",
        "nvidia.cufft",
        "nvidia.nvjitlink",
        "nvidia.cu12",
        "nvidia.cu13",
    ):
        try:
            mod = __import__(mod_name, fromlist=["__path__"])
            if hasattr(mod, "__path__") and mod.__path__:
                lib_dir = Path(mod.__path__[0]) / "lib"
                if lib_dir.is_dir():
                    search_dirs.append(lib_dir)
        except Exception:
            pass

    # 2. Probe sys.path for nvidia/*/lib and torch/lib
    for p in sys.path:
        p_path = Path(p)
        if not p_path.is_dir():
            continue
        try:
            for d in p_path.glob("nvidia/**/lib"):
                if d.is_dir():
                    search_dirs.append(d)
        except Exception:
            pass

        torch_lib = p_path / "torch" / "lib"
        if torch_lib.is_dir():
            search_dirs.append(torch_lib)

    # 3. Probe standard system library directories
    for sys_dir in (
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-13/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/nvidia/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
    ):
        p = Path(sys_dir)
        if p.is_dir():
            search_dirs.append(p)

    if "LD_LIBRARY_PATH" in os.environ:
        for lp in os.environ["LD_LIBRARY_PATH"].split(":"):
            lp_clean = lp.strip()
            if lp_clean:
                p = Path(lp_clean)
                if p.is_dir():
                    search_dirs.append(p)

    # De-duplicate directories preserving insertion order
    seen: set[str] = set()
    unique_dirs: list[Path] = []
    for d in search_dirs:
        try:
            resolved = str(d.resolve())
        except Exception:
            resolved = str(d)
        if resolved not in seen:
            seen.add(resolved)
            unique_dirs.append(d)

    # Export discovered directories to LD_LIBRARY_PATH for child processes or dynamic lookups
    if unique_dirs:
        curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_entries = [str(d) for d in unique_dirs if str(d) not in curr_ld.split(":")]
        if new_entries:
            os.environ["LD_LIBRARY_PATH"] = ":".join(new_entries) + (
                f":{curr_ld}" if curr_ld else ""
            )

    # 4. Preload in strict dependency order:
    # libcublas depends on libcublasLt, so load libcublasLt first, then libcublas, then libcudart.
    # Include both CUDA 12 (for llama-cpp-python) and CUDA 13 + cuDNN 9 (for onnxruntime-gpu).
    target_patterns = [
        ("libcublasLt.so.12", "libcublasLt.so.12*"),
        ("libcublas.so.12", "libcublas.so.12*"),
        ("libcudart.so.12", "libcudart.so.12*"),
        ("libcublasLt.so.13", "libcublasLt.so.13*"),
        ("libcublas.so.13", "libcublas.so.13*"),
        ("libnvrtc.so.13", "libnvrtc.so*"),
        ("libcurand.so.10", "libcurand.so*"),
        ("libcufft.so.12", "libcufft.so*"),
        ("libcudart.so.13", "libcudart.so.13*"),
        ("libcudnn.so.9", "libcudnn.so.9*"),
    ]

    loaded_any = False
    for exact_name, glob_pattern in target_patterns:
        loaded = False
        # 4a. Exact name search
        for d in unique_dirs:
            target = d / exact_name
            if target.is_file():
                try:
                    ctypes.CDLL(str(target), mode=ctypes.RTLD_GLOBAL)
                    logger.info("Preloaded CUDA runtime library: %s", target)
                    loaded = True
                    loaded_any = True
                    break
                except Exception as e:
                    logger.debug("Failed loading exact %s: %s", target, e)

        # 4b. Pattern fallback if exact versioned filename isn't directly matched
        if not loaded:
            for d in unique_dirs:
                try:
                    matches = sorted(d.glob(glob_pattern))
                    for match in matches:
                        if match.is_file():
                            try:
                                ctypes.CDLL(str(match), mode=ctypes.RTLD_GLOBAL)
                                logger.info("Preloaded CUDA runtime library via glob: %s", match)
                                loaded = True
                                loaded_any = True
                                break
                            except Exception as e:
                                logger.debug("Failed loading pattern %s: %s", match, e)
                    if loaded:
                        break
                except Exception:
                    pass

    _cuda_preloaded = loaded_any
    return loaded_any
