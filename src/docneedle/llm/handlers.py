"""Multimodal chat handlers and dynamic template introspection for local LLMs."""

import logging
from pathlib import Path
from typing import Any

import llama_cpp
from llama_cpp._utils import suppress_stdout_stderr
from llama_cpp.llama_chat_format import (
    Gemma4ChatHandler,
    Jinja2ChatFormatter,
    MTMDChatHandler,
    Qwen25VLChatHandler,
)

logger = logging.getLogger(__name__)


class DocNeedleMTMDChatHandler(MTMDChatHandler):
    """Custom MTMD chat handler with configurable thinking token support."""

    def __init__(self, clip_model_path: str, verbose: bool = True, use_gpu: bool = True) -> None:
        super().__init__(clip_model_path=clip_model_path, verbose=verbose, use_gpu=use_gpu)
        self.enable_thinking: bool = False

    def _init_mtmd_context(self, llama_model: Any) -> None:
        self.verbose = llama_model.verbose
        if self.mtmd_ctx is not None:
            return

        with suppress_stdout_stderr(disable=self.verbose):
            ctx_params = self._mtmd_cpp.mtmd_context_params_default()
            ctx_params.use_gpu = self.use_gpu
            ctx_params.print_timings = self.verbose
            ctx_params.n_threads = llama_model.n_threads
            ctx_params.warmup = False
            ctx_params.flash_attn_type = (
                llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
                if (
                    llama_model.context_params.flash_attn_type
                    == llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
                )
                else llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED
            )

            self.mtmd_ctx = self._mtmd_cpp.mtmd_init_from_file(
                self.clip_model_path.encode(), llama_model.model, ctx_params
            )

            if self.mtmd_ctx is None:
                raise ValueError(f"Failed to load mtmd context from: {self.clip_model_path}")

            if not self._mtmd_cpp.mtmd_support_vision(self.mtmd_ctx):
                raise ValueError("Vision is not supported by this model")

            def mtmd_free() -> None:
                with suppress_stdout_stderr(disable=self.verbose):
                    if self.mtmd_ctx is not None:
                        self._mtmd_cpp.mtmd_free(self.mtmd_ctx)
                        self.mtmd_ctx = None

            llama_model._stack.callback(mtmd_free)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["enable_thinking"] = getattr(self, "enable_thinking", False)
        return super().__call__(*args, **kwargs)


class DocNeedleQwenChatHandler(Qwen25VLChatHandler):
    """Custom Qwen 2.5-VL chat handler with thinking token toggling."""

    CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if loop.first and message['role'] != 'system' %}"
        "<|im_start|>system\n"
        "{{ self.DEFAULT_SYSTEM_MESSAGE }}<|im_end|>\n"
        "{% endif %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}<|im_end|>\n"
        "{% else %}{% for content in message['content'] %}"
        "{% if content['type'] == 'image_url' %}"
        "{% if content.image_url is string %}{{ content.image_url }}"
        "{% else %}{{ content.image_url.url }}{% endif %}"
        "{% elif content['type'] == 'text' %}{{ content['text'] }}"
        "{% endif %}{% endfor %}<|im_end|>\n{% endif %}"
        "{% endfor %}<|im_start|>assistant\n"
        "{% if enable_thinking %}<think>\n"
        "{% else %}<think>\n\n</think>\n\n{% endif %}"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.enable_thinking: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["enable_thinking"] = getattr(self, "enable_thinking", False)
        return super().__call__(*args, **kwargs)


class DocNeedleGemma4ChatHandler(Gemma4ChatHandler):
    """Custom Gemma-4 chat handler with thinking token toggling."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.enable_thinking: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["enable_thinking"] = getattr(self, "enable_thinking", False)
        return super().__call__(*args, **kwargs)


def create_multimodal_chat_handler(
    model_path: Path,
    mmproj_path: Path,
    is_verbose: bool,
    eff_layers: int,
) -> Any:
    """Instantiate the architecture-appropriate vision chat handler for multimodal inference."""
    mmproj_str = str(mmproj_path).lower()
    model_str = str(model_path).lower()

    if "lfm" in mmproj_str or "lfm" in model_str or "mtmd" in mmproj_str:
        logger.info(
            "Multimodal vision handler: DocNeedleMTMDChatHandler for %s",
            mmproj_path.name,
        )
        return DocNeedleMTMDChatHandler(
            clip_model_path=str(mmproj_path),
            verbose=is_verbose,
            use_gpu=(eff_layers != 0),
        )

    if "qwen" in model_str or "qwen" in mmproj_str:
        logger.info(
            "Multimodal vision handler: DocNeedleQwenChatHandler for %s",
            mmproj_path.name,
        )
        return DocNeedleQwenChatHandler(
            clip_model_path=str(mmproj_path),
            verbose=is_verbose,
        )

    logger.info(
        "Multimodal vision handler: DocNeedleGemma4ChatHandler for %s",
        mmproj_path.name,
    )
    return DocNeedleGemma4ChatHandler(
        clip_model_path=str(mmproj_path),
        verbose=is_verbose,
        use_gpu=(eff_layers != 0),
    )


def _decode_special_token(llm: Any, token_id: int) -> str:
    """Decode a special token ID into its string representation if valid."""
    if token_id < 0:
        return ""
    return llm.detokenize([token_id], special=True).decode("utf-8", errors="ignore")


def _probe_template_delimiters(
    template_str: str,
    bos_token: str,
    eos_token: str,
) -> tuple[Any, str, str]:
    """Build chat formatter from GGUF template and dynamically inspect reasoning delimiters."""
    gen_formatter = Jinja2ChatFormatter(
        template=template_str,
        eos_token=eos_token,
        bos_token="",
        add_generation_prompt=True,
    )
    nogen_formatter = Jinja2ChatFormatter(
        template=template_str,
        eos_token=eos_token,
        bos_token="",
        add_generation_prompt=False,
    )

    think_open = "<think>" if "<think>" in template_str else "<think>\n"
    think_close = "</think>" if "</think>" in template_str else "</think>"
    try:
        t_sentinel = "\x00__DOCNEEDLE_THINK__\x00"
        a_sentinel = "\x00__DOCNEEDLE_ANSWER__\x00"
        base_msgs = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ]
        prefix = gen_formatter(messages=base_msgs, enable_thinking=True).prompt
        full = nogen_formatter(
            messages=[
                *base_msgs,
                {
                    "role": "assistant",
                    "content": a_sentinel,
                    "reasoning_content": t_sentinel,
                },
            ],
            enable_thinking=True,
        ).prompt
        if t_sentinel in full and a_sentinel in full:
            t_idx = full.index(t_sentinel)
            a_idx = full.index(a_sentinel)
            if full.startswith(prefix) and t_idx >= len(prefix):
                derived_open = full[len(prefix) : t_idx]
                if derived_open.strip():
                    think_open = derived_open
            derived_close = full[t_idx + len(t_sentinel) : a_idx]
            if derived_close.strip():
                think_close = derived_close
    except Exception as err:
        logger.debug("Using default reasoning tags during template inspection: %s", err)

    return gen_formatter, think_open, think_close


def _extract_llm_chat_template(llm: Any) -> tuple[Any, str, str]:
    """Extract embedded chat template formatter and reasoning tags from loaded Llama instance."""
    default_open = "<think>\n"
    default_close = "</think>"
    metadata = getattr(llm, "metadata", None) or {}
    template_str = metadata.get("tokenizer.chat_template")
    if not template_str:
        return None, default_open, default_close

    if "<think>" in template_str:
        default_open = "<think>"
        default_close = "</think>"

    try:
        bos_str = _decode_special_token(llm, llm.token_bos())
        eos_str = _decode_special_token(llm, llm.token_eos())
        return _probe_template_delimiters(template_str, bos_str, eos_str)
    except Exception as tmpl_err:
        logger.warning("Could not initialize GGUF Jinja2ChatFormatter: %s", tmpl_err)
        return None, default_open, default_close
