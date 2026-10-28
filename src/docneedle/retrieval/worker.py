"""Isolated worker process for EmbeddingGemma-2 multimodal embedder.

Running the C++ EmbeddingGemma-2 GGUF engine in an isolated worker process
guarantees complete symbol isolation from llama-cpp-python, eliminating
cross-version DSO symbol collisions while providing fast in-memory IPC.
"""

import atexit
import logging
import multiprocessing as mp
import threading
from pathlib import Path
from typing import Any

import numpy as np

from docneedle.retrieval.embedder import BaseEmbedder, EmbeddingGemma2Embedder

logger = logging.getLogger(__name__)


def _embedder_worker_loop(
    pipe: Any,
    model_path: str,
    mmproj_path: str,
    dimension: int,
    n_gpu_layers: int | None = None,
) -> None:
    """Worker process loop that hosts EmbeddingGemma2Embedder."""
    try:
        import ctypes
        from pathlib import Path

        try:
            from docneedle.cuda_compat import preload_cuda_runtime_libs

            preload_cuda_runtime_libs()
        except Exception:
            pass

        retrieval_dir = Path(__file__).resolve().parent
        for dep in (
            "libggml-base.so",
            "libggml-cpu.so",
            "libggml-cuda.so",
            "libggml_llamacpp.so",
            "libllama-common.so.0.7.0",
            "libllama-common.so",
            "libllama.so",
            "libllama.so.0",
            "libmtmd.so",
            "libmtmd.so.0",
        ):
            dep_path = retrieval_dir / dep
            if dep_path.exists():
                try:
                    ctypes.CDLL(str(dep_path), mode=ctypes.RTLD_GLOBAL)
                except Exception as lib_err:
                    logger.debug("Optional pre-load for %s: %s", dep, lib_err)

        emb = EmbeddingGemma2Embedder(
            model_path=model_path,
            mmproj_path=mmproj_path,
            dimension=dimension,
            n_gpu_layers=n_gpu_layers or 0,
        )
        pipe.send(("READY", None))
    except Exception as e:
        pipe.send(("ERROR", str(e)))
        return

    while True:
        try:
            if not pipe.poll(timeout=None):
                continue
            cmd, payload = pipe.recv()
            if cmd == "STOP":
                break
            elif cmd == "TEXTS":
                texts, is_query, doc_names = payload
                res = emb.embed_texts(texts, is_query=is_query, doc_names=doc_names)
                pipe.send(("OK", res))
            elif cmd == "IMAGES":
                (image_bytes_list,) = payload
                res = emb.embed_images(image_bytes_list)
                pipe.send(("OK", res))
            else:
                pipe.send(("ERROR", f"Unknown command: {cmd}"))
        except EOFError:
            break
        except Exception as err:
            logger.exception("Worker loop error: %s", err)
            pipe.send(("ERROR", str(err)))


class IsolatedEmbeddingClient(BaseEmbedder):
    """IPC client proxy for the isolated EmbeddingGemma-2 worker process."""

    def __init__(
        self,
        model_path: str | Path,
        mmproj_path: str | Path,
        dimension: int = 768,
        n_gpu_layers: int | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.mmproj_path = str(mmproj_path)
        self.dimension = dimension
        self.n_gpu_layers = n_gpu_layers
        self._lock = threading.Lock()
        self._proc: mp.Process | None = None
        self._pipe: Any = None
        self._start_worker()
        atexit.register(self.close)

    def _start_worker(self) -> None:
        cur_proc = mp.current_process()
        was_daemon = cur_proc.daemon
        try:
            # Allow creating worker child processes when running inside daemonic
            # runner environments such as ZeroGPU task workers.
            cur_proc.daemon = False
            ctx = mp.get_context("spawn")
            parent_pipe, child_pipe = ctx.Pipe(duplex=True)
            self._pipe = parent_pipe
            self._proc = ctx.Process(
                target=_embedder_worker_loop,
                args=(
                    child_pipe,
                    self.model_path,
                    self.mmproj_path,
                    self.dimension,
                    self.n_gpu_layers,
                ),
                daemon=False,
            )
            self._proc.start()
            child_pipe.close()

            status, err = self._recv()
            if status != "READY":
                self.close()
                raise RuntimeError(f"Failed to initialize isolated embedder worker: {err}")
        finally:
            cur_proc.daemon = was_daemon

    def embed_texts(
        self,
        texts: list[str],
        is_query: bool = False,
        doc_names: list[str] | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        with self._lock:
            self._ensure_alive()
            self._pipe.send(("TEXTS", (texts, is_query, doc_names)))
            status, res = self._recv()
            if status != "OK":
                raise RuntimeError(f"Embedder worker error: {res}")
            return res

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query], is_query=True)[0]

    def embed_images(self, image_bytes_list: list[bytes]) -> np.ndarray:
        if not image_bytes_list:
            return np.zeros((0, self.dimension), dtype=np.float32)
        with self._lock:
            self._ensure_alive()
            self._pipe.send(("IMAGES", (image_bytes_list,)))
            status, res = self._recv()
            if status != "OK":
                raise RuntimeError(f"Embedder worker error: {res}")
            return res

    def _recv(self) -> tuple[str, Any]:
        """Receive a worker reply, failing fast if the worker process dies.

        A native crash (e.g. SIGSEGV inside llama.cpp) kills the child without
        sending a reply; a plain blocking recv() would then hang forever.
        """
        while not self._pipe.poll(timeout=0.5):
            if self._proc is None or not self._proc.is_alive():
                exitcode = self._proc.exitcode if self._proc is not None else None
                self._proc = None
                raise RuntimeError(
                    f"Isolated embedder worker died unexpectedly (exit code: {exitcode})."
                )
        return self._pipe.recv()

    def _ensure_alive(self) -> None:
        if self._proc is None or not self._proc.is_alive():
            logger.warning("Isolated embedder worker died, restarting...")
            self._start_worker()

    def close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.send(("STOP", None))
            except Exception:
                pass
            try:
                self._pipe.close()
            except Exception:
                pass
            self._pipe = None
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None

    def __del__(self) -> None:
        self.close()
