"""Script to download quantized models for DocNeedle.

- Embedding: mradermacher/harrier-oss-v1-0.6b-GGUF (harrier-oss-v1-0.6b.Q4_K_M.gguf)
- Reranker: mradermacher/Qwen3-Reranker-0.6B-GGUF (Qwen3-Reranker-0.6B.Q4_K_M.gguf)
- LLM: unsloth/gemma-4-E2B-it-GGUF (gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf)
"""

import argparse
import os
import sys
import urllib.request

MODELS_CONFIG = {
    "embedding": {
        "name": "harrier-oss-v1-0.6b.Q4_K_M",
        "repo_id": "mradermacher/harrier-oss-v1-0.6b-GGUF",
        "filename": "harrier-oss-v1-0.6b.Q4_K_M.gguf",
        "url": (
            "https://huggingface.co/mradermacher/harrier-oss-v1-0.6b-GGUF/resolve/main/"
            "harrier-oss-v1-0.6b.Q4_K_M.gguf"
        ),
        "subpath": os.path.join("embedding", "harrier-oss-v1-0.6b.Q4_K_M.gguf"),
    },
    "reranker": {
        "name": "Qwen3-Reranker-0.6B.Q4_K_M",
        "repo_id": "mradermacher/Qwen3-Reranker-0.6B-GGUF",
        "filename": "Qwen3-Reranker-0.6B.Q4_K_M.gguf",
        "url": (
            "https://huggingface.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/resolve/main/"
            "Qwen3-Reranker-0.6B.Q4_K_M.gguf"
        ),
        "subpath": os.path.join("reranker", "Qwen3-Reranker-0.6B.Q4_K_M.gguf"),
    },
    "llm": {
        "name": "gemma-4-E2B-it-qat-UD-Q4_K_XL",
        "repo_id": "unsloth/gemma-4-E2B-it-qat-GGUF",
        "filename": "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
        "url": (
            "https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF/resolve/main/"
            "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"
        ),
        "subpath": os.path.join("llm", "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"),
    },
}


def download_file(
    url: str,
    dest_path: str,
    repo_id: str | None = None,
    filename: str | None = None,
) -> None:
    """Download a file with progress reporting."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading: {url} -> {dest_path}")
    if repo_id and filename:
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=os.path.dirname(dest_path),
            )
            print("\n  Download complete.")
            return
        except Exception as e:
            print(f"hf_hub_download failed ({e}), falling back to standard download...")

    def report_progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)"
            )
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  Downloaded: {mb_downloaded:.1f}MB")
        sys.stdout.flush()

    headers = {"User-Agent": "DocNeedle-ModelDownloader/1.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        block_size = 1024 * 1024  # 1MB blocks
        downloaded = 0
        block_num = 0
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            block_num += 1
            report_progress(block_num, block_size, total_size)
    print("\n  Download complete.")


def download_onnx_embedding_model(models_dir: str) -> None:
    """Download Harrier-OSS-v1-0.6B ONNX Q4 model and tokenizer from onnx-community."""
    from huggingface_hub import hf_hub_download

    dest_dir = os.path.join(models_dir, "embedding", "harrier-onnx")
    os.makedirs(dest_dir, exist_ok=True)
    repo_id = "onnx-community/harrier-oss-v1-0.6b-ONNX"

    files = [
        "onnx/model_q4.onnx",
        "onnx/model_q4.onnx_data",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    print(f"\nChecking/Downloading Harrier-OSS ONNX model into {dest_dir}...")
    for f in files:
        target = os.path.join(dest_dir, f)
        if not os.path.exists(target):
            print(f"  Downloading {f}...")
            hf_hub_download(repo_id=repo_id, filename=f, local_dir=dest_dir)
        else:
            print(f"  {f} already exists.")


def download_embedding_model(models_dir: str) -> None:
    """Download Harrier-OSS-v1-0.6B Q4_K_M GGUF."""
    cfg = MODELS_CONFIG["embedding"]
    dest_path = os.path.join(models_dir, cfg["subpath"])
    print(f"\n[1/3] Checking/Downloading {cfg['name']} into {dest_path}...")
    download_file(
        cfg["url"],
        dest_path,
        repo_id=cfg.get("repo_id"),
        filename=cfg.get("filename"),
    )


def download_reranker_model(models_dir: str) -> None:
    """Download Qwen3-Reranker-0.6B Q4_K_M GGUF."""
    cfg = MODELS_CONFIG["reranker"]
    dest_path = os.path.join(models_dir, cfg["subpath"])
    print(f"\n[2/3] Checking/Downloading {cfg['name']} into {dest_path}...")
    download_file(
        cfg["url"],
        dest_path,
        repo_id=cfg.get("repo_id"),
        filename=cfg.get("filename"),
    )


def download_llm_model(models_dir: str) -> None:
    """Download Gemma-4-E2B-it-qat-UD-Q4_K_XL GGUF."""
    cfg = MODELS_CONFIG["llm"]
    dest_path = os.path.join(models_dir, cfg["subpath"])
    print(f"\n[3/3] Checking/Downloading {cfg['name']} into {dest_path}...")
    download_file(
        cfg["url"],
        dest_path,
        repo_id=cfg.get("repo_id"),
        filename=cfg.get("filename"),
    )


def download_rapidocr_models() -> None:
    """Pre-download RapidOCR detection and recognition ONNX models without importing onnxruntime."""
    try:
        from pathlib import Path

        import rapidocr
        from rapidocr.inference_engine.base import FileInfo
        from rapidocr.utils.download_models import download_task
        from rapidocr.utils.typings import (
            EngineType,
            LangDet,
            LangRec,
            ModelType,
            OCRVersion,
            TaskType,
        )

        print("\nChecking/Downloading RapidOCR PP-OCRv6 models...")
        ocr_models_dir = Path(rapidocr.__file__).resolve().parent / "models"
        ocr_models_dir.mkdir(parents=True, exist_ok=True)
        for model_tier in (ModelType.SMALL, ModelType.MEDIUM):
            try:
                download_task(
                    ocr_models_dir,
                    FileInfo(
                        EngineType.ONNXRUNTIME,
                        OCRVersion.PPOCRV6,
                        TaskType.DET,
                        LangDet.CH,
                        model_tier,
                    ),
                )
            except ValueError:
                pass
            try:
                download_task(
                    ocr_models_dir,
                    FileInfo(
                        EngineType.ONNXRUNTIME,
                        OCRVersion.PPOCRV6,
                        TaskType.REC,
                        LangRec.CH,
                        model_tier,
                    ),
                )
            except ValueError:
                pass
        print("RapidOCR models are verified and present.")
    except Exception as e:
        print(f"RapidOCR models check bypassed: {e}")


def download_chunker_tokenizer(tokenizer_model: str = "Qwen/Qwen3-Reranker-0.6B") -> None:
    """Pre-download Hugging Face tokenizer used by DoclingChunker so GPU workers do not wait."""
    try:
        from transformers import AutoTokenizer

        print(f"\nChecking/Downloading chunker tokenizer ({tokenizer_model})...")
        AutoTokenizer.from_pretrained(tokenizer_model)
        print("Chunker tokenizer is verified and cached.")
    except Exception as e:
        print(f"Chunker tokenizer check bypassed: {e}")


def ensure_all_models_present(models_dir: str | None = None) -> None:
    """Ensure all required quantized models exist, downloading any missing ones."""
    if models_dir is None:
        models_dir = os.environ.get("MODEL_STORAGE_DIR", "models")

    for _, cfg in MODELS_CONFIG.items():
        dest = os.path.join(models_dir, cfg["subpath"])
        if not os.path.exists(dest):
            print(f"Missing model [{cfg['name']}]. Starting automatic bootstrap download...")
            download_file(
                cfg["url"],
                dest,
                repo_id=cfg.get("repo_id"),
                filename=cfg.get("filename"),
            )
    download_rapidocr_models()
    download_chunker_tokenizer()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DocNeedle quantized models")
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Target directory for downloaded models (default: models)",
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Download only Harrier-OSS-v1-0.6B Q4_K_M GGUF",
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Download only Qwen3-Reranker-0.6B Q4_K_M GGUF",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Download only Gemma-4-E2B-it-qat-UD-Q4_K_XL GGUF",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Download Harrier-OSS-v1-0.6B ONNX Q4 model and tokenizer",
    )

    args = parser.parse_args()
    download_all = not args.embedding and not args.reranker and not args.llm and not args.onnx

    if download_all or args.embedding:
        download_embedding_model(args.models_dir)

    if download_all or args.onnx:
        download_onnx_embedding_model(args.models_dir)

    if download_all or args.reranker:
        download_reranker_model(args.models_dir)

    if download_all or args.llm:
        download_llm_model(args.models_dir)

    print("\nAll requested model downloads are complete.")


if __name__ == "__main__":
    main()
