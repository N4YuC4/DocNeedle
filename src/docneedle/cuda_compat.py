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
_modular_backends_preloaded = False


def configure_cuda_library_paths() -> list[Path]:
    """Discover NVIDIA CUDA library directories and add them to LD_LIBRARY_PATH."""
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

    # 3. Probe standard system CUDA directories
    for sys_dir in (
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-13/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/nvidia/lib64",
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
        custom_dirs = [
            str(d)
            for d in unique_dirs
            if str(d) not in ("/usr/lib/x86_64-linux-gnu", "/usr/local/lib", "/lib", "/usr/lib")
        ]
        new_entries = [d for d in custom_dirs if d not in curr_ld.split(":")]
        if new_entries:
            os.environ["LD_LIBRARY_PATH"] = (f"{curr_ld}:" if curr_ld else "") + ":".join(
                new_entries
            )

    return unique_dirs


def preload_cuda_runtime_libs() -> bool:
    """Preload NVIDIA CUDA runtime and cuBLAS shared libraries with RTLD_GLOBAL.

    Returns:
        bool: True if at least one CUDA runtime library was successfully preloaded,
              False otherwise.
    """
    global _cuda_preloaded
    if _cuda_preloaded:
        preload_modular_backends()
        return True

    unique_dirs = configure_cuda_library_paths()

    # Preload in strict dependency order:
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
    preload_modular_backends()
    return loaded_any


def preload_modular_backends() -> bool:
    """Preload modular CUDA and CPU compute backends for llama.cpp if supported."""
    global _modular_backends_preloaded
    if _modular_backends_preloaded:
        return True

    # Critical ZeroGPU / container guard:
    # If on ZeroGPU, do NOT preload modular CUDA backends at startup.
    # ZeroGPU mounts /dev/nvidia* but blocks access via cgroups until @spaces.GPU leases it.
    # Loading ggml-cuda without an active lease triggers cuInit failure, poisoning memory.
    if bool(os.environ.get("SPACES_ZERO_GPU")) or os.environ.get("SYSTEM") == "spaces":
        logger.debug("ZeroGPU detected. Deferring modular CUDA backend load to GPU lease.")
        return False

    if sys.platform.startswith("linux"):
        has_nvidia_dev = any(
            Path(f"/dev/{dev}").exists() for dev in ("nvidia0", "nvidiactl", "nvhost-ctrl")
        )
        if not has_nvidia_dev:
            logger.debug(
                "No accessible NVIDIA device node (/dev/nvidia*) detected. "
                "Skipping modular CUDA backend preload."
            )
            return False

    try:
        import llama_cpp

        lib_dir = Path(llama_cpp.__file__).resolve().parent / "lib"
        for base_name in ("libggml.so", "libggml-base.so", "libggml_llamacpp.so"):
            lib_path = lib_dir / base_name
            if lib_path.is_file():
                libggml = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                if hasattr(libggml, "ggml_backend_load_all_from_path"):
                    libggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
                    libggml.ggml_backend_load_all_from_path(str(lib_dir).encode("utf-8"))
                    os.environ["GGML_BACKEND_SEARCH_PATH"] = str(lib_dir)
                    logger.info(
                        "--> [CUDA Backends] Preloaded modular llama.cpp backends from %s",
                        lib_dir,
                    )
                    _modular_backends_preloaded = True
                    return True
    except Exception as e:
        logger.debug("Modular backend preloading skipped: %s", e)
    return False


def prewarm_models_page_cache(model_paths: list[str | Path] | None = None) -> list[str]:
    """Pre-warm model files into Linux OS page cache to eliminate disk I/O latency on mmap().

    This function reads model files in 16MB sequential chunks into user memory and hints the kernel
    with POSIX_FADV_WILLNEED. The Linux kernel retains the pages in OS page cache (system RAM).
    When llama.cpp subsequently calls mmap() with use_mmap=True, page faults are serviced
    directly from system RAM without blocking on physical disk I/O.

    This function does NOT call CUDA, touch GPU devices, or initialize libcuda, so it is
    100% safe to run in the parent web process or background threads.

    Returns:
        list[str]: Names of successfully pre-warmed model files.
    """
    import time

    targets: list[Path] = []
    if model_paths:
        for p in model_paths:
            p_obj = Path(p)
            if p_obj.is_file():
                targets.append(p_obj)
    else:
        # Auto-discover from config or standard model storage directories
        try:
            from docneedle.config import settings

            base_dir = Path(settings.model_storage_dir)
        except Exception:
            base_dir = Path("models")

        if base_dir.is_dir():
            for sub in ("llm", "embedding"):
                sub_dir = base_dir / sub
                if sub_dir.is_dir():
                    for f in sorted(sub_dir.glob("*.gguf")):
                        if f.is_file() and not f.name.endswith(".incomplete"):
                            targets.append(f)

    prewarmed: list[str] = []
    if not targets:
        return prewarmed

    t0 = time.perf_counter()
    chunk_size = 16 * 1024 * 1024  # 16 MB buffer
    buf = bytearray(chunk_size)
    total_bytes = 0

    for model_path in targets:
        try:
            file_size = model_path.stat().st_size
            file_t0 = time.perf_counter()
            with open(model_path, "rb") as f:
                fd = f.fileno()
                if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_WILLNEED"):
                    try:
                        os.posix_fadvise(fd, 0, file_size, os.POSIX_FADV_WILLNEED)
                    except Exception:
                        pass
                read_bytes = 0
                while True:
                    n = f.readinto(buf)
                    if not n:
                        break
                    read_bytes += n
            total_bytes += read_bytes
            prewarmed.append(model_path.name)
            file_elapsed = max(time.perf_counter() - file_t0, 0.001)
            speed_mb = (read_bytes / (1024 * 1024)) / file_elapsed
            logger.info(
                "--> [PageCache] Pre-warmed %s (%.1f MB in %.2fs, %.1f MB/s)",
                model_path.name,
                read_bytes / (1024 * 1024),
                file_elapsed,
                speed_mb,
            )
        except Exception as e:
            logger.debug("Failed pre-warming %s: %s", model_path, e)

    total_elapsed = max(time.perf_counter() - t0, 0.001)
    logger.info(
        "--> [PageCache] Pre-warmed %d model(s) (%.1f MB total) into OS RAM in %.2fs (%.1f MB/s).",
        len(prewarmed),
        total_bytes / (1024 * 1024),
        total_elapsed,
        (total_bytes / (1024 * 1024)) / total_elapsed,
    )
    return prewarmed
