"""Local in-process GGUF LLM engine for information extraction."""

import base64
import io
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from docneedle.config import settings
from docneedle.llm.handlers import (
    DocNeedleGemma4ChatHandler,
    DocNeedleMTMDChatHandler,
    DocNeedleQwenChatHandler,
    _decode_special_token,
    _extract_llm_chat_template,
    _probe_template_delimiters,
    create_multimodal_chat_handler,
)
from docneedle.llm.parsing import (
    LLMExtractionResult,
    _clean_json_placeholders,
    _extract_prefixed_answer,
    _is_copied_from_prompt_example,
    _is_label_hallucination,
    _is_not_found_value,
    _normalize_typed_value,
    _parse_model_response,
    _process_extracted_answer,
    _process_schema_extraction_output,
    _rescue_schema_fields_from_text,
    _sanitize_extracted_value,
    _strip_field_prefix,
    _strip_thought_open_tag,
    _try_parse_json_value,
    _try_parse_schema_json,
)
from docneedle.llm.prompts import (
    BASE_RULES,
    CONSTRAINTS,
    CORE_RULES,
    EXTRACTION_RULES,
    NON_THINKING_DIRECTIVE,
    NON_THINKING_RULE,
    NOT_FOUND_RULE,
    SYSTEM_IDENTITY,
    SYSTEM_PROMPT,
    TASK_DEFINITION,
    THINKING_DIRECTIVE,
    THINKING_RULE,
    TYPE_GUIDELINES,
    _build_extraction_prompt,
    _build_schema_extraction_prompt,
    _build_type_instruction,
    _get_base_rules,
    _render_template_messages,
)
from docneedle.llm.speculative import _create_speculative_draft_model

logger = logging.getLogger(__name__)

__all__ = [
    "BASE_RULES",
    "CONSTRAINTS",
    "CORE_RULES",
    "DocNeedleGemma4ChatHandler",
    "DocNeedleMTMDChatHandler",
    "DocNeedleQwenChatHandler",
    "EXTRACTION_RULES",
    "LLMExtractionResult",
    "LocalGemmaEngine",
    "LocalLLMEngine",
    "NON_THINKING_DIRECTIVE",
    "NON_THINKING_RULE",
    "NOT_FOUND_RULE",
    "SYSTEM_IDENTITY",
    "SYSTEM_PROMPT",
    "TASK_DEFINITION",
    "THINKING_DIRECTIVE",
    "THINKING_RULE",
    "TYPE_GUIDELINES",
    "_build_extraction_prompt",
    "_build_type_instruction",
    "_clean_json_placeholders",
    "_create_speculative_draft_model",
    "_decode_special_token",
    "_extract_llm_chat_template",
    "_extract_prefixed_answer",
    "_get_base_rules",
    "_is_copied_from_prompt_example",
    "_is_label_hallucination",
    "_is_not_found_value",
    "_log_extraction_trace",
    "_normalize_typed_value",
    "_parse_model_response",
    "_probe_template_delimiters",
    "_process_extracted_answer",
    "_render_template_messages",
    "_rescue_schema_fields_from_text",
    "_resolve_default_model_path",
    "_resolve_llm_mmproj_path",
    "_sanitize_extracted_value",
    "_strip_field_prefix",
    "_strip_thought_open_tag",
    "_try_parse_json_value",
    "_try_parse_schema_json",
    "create_multimodal_chat_handler",
    "get_local_engine",
]


def _log_extraction_trace(
    query_key: str,
    thinking: str | None,
    value: Any,
    found: bool = True,
) -> None:
    """Log model extraction trace without leaking sensitive PII unless explicitly enabled."""
    if not settings.llm_log_extraction_payloads:
        logger.debug(
            "LLM extraction finished for '%s' (found=%s, has_thinking=%s)",
            query_key,
            found,
            bool(thinking),
        )
        return

    sep = "=" * 25
    val_str = str(value).strip() if value is not None else "[None]"
    if thinking:
        logger.debug(
            "\n%s [MODEL THINKING (%s)] %s\n%s\n%s [MODEL ANSWER (%s)] %s\n%s\n%s",
            sep,
            query_key,
            sep,
            thinking.strip(),
            sep,
            query_key,
            sep,
            val_str,
            "=" * 75,
        )
    else:
        logger.debug(
            "\n%s [MODEL OUTPUT (%s)] %s\n%s\n%s",
            sep,
            query_key,
            sep,
            val_str,
            "=" * 75,
        )


def _resolve_default_model_path() -> Path:
    """Resolve the configured GGUF model file strictly without silent fallbacks."""
    base = Path(settings.model_storage_dir) / "llm"
    model_name = settings.llm_model_name
    filename = model_name if model_name.endswith(".gguf") else f"{model_name}.gguf"
    candidate = base / filename
    if not candidate.exists():
        raise FileNotFoundError(f"Configured GGUF model '{model_name}' not found at: {candidate}")
    return candidate


def _resolve_llm_mmproj_path() -> Path | None:
    """Resolve multimodal projector GGUF for the LLM if present."""
    base = Path(settings.model_storage_dir) / "llm"
    mmproj_name = settings.llm_mmproj_name
    filename = mmproj_name if mmproj_name.endswith(".gguf") else f"{mmproj_name}.gguf"
    candidate = base / filename
    if candidate.exists():
        return candidate
    alt_candidate = base / "mmproj-F16.gguf"
    if alt_candidate.exists():
        return alt_candidate
    return None


class LocalLLMEngine:
    """Manages model-agnostic in-process GGUF LLM inference with dynamic template inspection."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        from docneedle.cuda_compat import preload_cuda_runtime_libs

        t_init_start = time.perf_counter()
        preload_cuda_runtime_libs()
        import llama_cpp
        from llama_cpp import Llama

        if model_path is None:
            self.model_path = _resolve_default_model_path()
        else:
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"GGUF model file not found at: {self.model_path}")

        supports_gpu = getattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)()
        eff_layers = settings.effective_gpu_layers
        self._log_hardware_diagnostics(eff_layers, supports_gpu)

        self.mmproj_path = _resolve_llm_mmproj_path()
        chat_handler = None
        is_verbose = settings.gpu_enabled or logger.isEnabledFor(logging.DEBUG)

        if self.mmproj_path is not None:
            try:
                chat_handler = create_multimodal_chat_handler(
                    model_path=self.model_path,
                    mmproj_path=self.mmproj_path,
                    is_verbose=is_verbose,
                    eff_layers=eff_layers,
                )
                logger.info(
                    "Multimodal vision support enabled with %s (use_gpu=%s)",
                    self.mmproj_path,
                    eff_layers != 0,
                )
            except Exception as e:
                logger.error("Could not initialize multimodal chat handler: %s", e)
                raise

        draft_model = _create_speculative_draft_model(self.model_path, Llama)
        logits_all = draft_model is not None
        self.n_ctx = min(settings.llm_n_ctx, 4096) if logits_all else settings.llm_n_ctx

        t_llama_start = time.perf_counter()
        self.llm = Llama(
            model_path=str(self.model_path),
            chat_handler=chat_handler,
            draft_model=draft_model,
            logits_all=logits_all,
            n_ctx=self.n_ctx,
            n_gpu_layers=eff_layers,
            n_threads=settings.cpu_threads,
            n_threads_batch=settings.cpu_threads_batch,
            n_batch=settings.model_n_batch,
            n_ubatch=settings.model_n_ubatch,
            type_k=settings.model_kv_cache_type_k,
            type_v=settings.model_kv_cache_type_v,
            flash_attn=settings.model_flash_attn,
            use_mmap=True,
            use_mlock=False,
            verbose=is_verbose,
        )
        t_llama_elapsed = time.perf_counter() - t_llama_start

        t_mtmd_elapsed = 0.0
        if chat_handler is not None:
            try:
                t_mtmd_start = time.perf_counter()
                chat_handler._init_mtmd_context(self.llm)
                t_mtmd_elapsed = time.perf_counter() - t_mtmd_start
                logger.info("Multimodal vision context initialized successfully on GPU.")
            except Exception as e:
                logger.error("Failed to initialize multimodal vision context: %s", e)
                raise

        (
            self._chat_formatter,
            self._thought_open_tag,
            self._thought_close_tag,
        ) = _extract_llm_chat_template(self.llm)

        self._lock = threading.Lock()
        t_total_init = time.perf_counter() - t_init_start
        logger.info(
            "--> [LLM LOAD TIMING] Llama Model: %.2fs | MTMD Context: %.2fs | Total Init: %.2fs "
            "(model=%s, n_ctx=%d, gpu_layers=%d).",
            t_llama_elapsed,
            t_mtmd_elapsed,
            t_total_init,
            self.model_path.name,
            self.n_ctx,
            eff_layers,
        )

    @staticmethod
    def _log_hardware_diagnostics(eff_layers: int, supports_gpu: bool) -> None:
        """Inspect and log host hardware, CUDA drivers, and thread runtime configurations."""
        cuda_device_name = "N/A"
        cuda_available = False
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                cuda_device_name = (
                    f"{torch.cuda.get_device_name(0)} (devices={torch.cuda.device_count()})"
                )
        except Exception:
            pass

        logger.info(
            "--> [HARDWARE DIAGNOSTIC] GPU Enabled: %s | eff_layers: %d | llama_supports_gpu: %s "
            "| PyTorch CUDA: %s (%s) | cuda_env: %s | threads: %d (batch=%d)",
            settings.gpu_enabled,
            eff_layers,
            supports_gpu,
            cuda_available,
            cuda_device_name,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            settings.cpu_threads,
            settings.cpu_threads_batch,
        )

        if settings.gpu_enabled and not supports_gpu:
            logger.warning(
                "--> [CUDA WARNING] GPU offloading is requested (eff_layers=%d), but llama_cpp "
                "is compiled without GPU offload support!",
                eff_layers,
            )

    def _attempt_continuation(self, prompt: str, thinking: str, query_key: str) -> str | None:
        """Execute a fast continuation turn when reasoning terminated without closing thought."""
        logger.info(
            "Model reasoning did not output closed answer for '%s'. "
            "Running fast continuation to extract final answer...",
            query_key,
        )
        close_sep = self._thought_close_tag
        if not close_sep.endswith("\n"):
            close_sep = f"{close_sep}\n"
        continuation_prompt = f"{prompt}{thinking.strip()}{close_sep}"

        with self._lock:
            cont_tokens = self.llm.tokenize(continuation_prompt.encode("utf-8"))
            remaining_ctx = self.n_ctx - len(cont_tokens) - 8
            if remaining_ctx < 16:
                logger.warning(
                    "Skipping continuation for '%s': insufficient remaining context (%d tokens).",
                    query_key,
                    remaining_ctx,
                )
                return None
            max_cont_tokens = min(128, remaining_ctx)
            cont_res = self.llm(
                prompt=continuation_prompt,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                max_tokens=max_cont_tokens,
                stop=settings.llm_stop_tokens,
            )
        cont_text = cont_res["choices"][0]["text"] or ""
        _, cont_answer = _parse_model_response(
            cont_text,
            started_in_thought=False,
            thought_open_tag=self._thought_open_tag,
            thought_close_tag=self._thought_close_tag,
        )
        return cont_answer or None

    def _extract_item_multimodal(
        self,
        query_key: str,
        target_query: str,
        context: str,
        valid_images: list[bytes],
        expected_type: str | None,
        items_schema: dict[str, Any] | None,
        use_thinking: bool,
        is_table: bool,
    ) -> LLMExtractionResult:
        """Execute multimodal item extraction combining document text and high-res page images."""
        logger.info(
            "--> [LLM Multimodal] Analyzing %d image chunk(s) for field '%s'...",
            len(valid_images),
            query_key,
        )
        type_prompt = _build_type_instruction(expected_type, items_schema)
        active_sys_prompt = _get_base_rules() + (
            THINKING_RULE if use_thinking else NON_THINKING_RULE
        )

        acro_section = ""
        if "### Dijital Form Alanları (AcroForm)" in context:
            parts = context.split("### Dijital Form Alanları (AcroForm)")
            if len(parts) > 1:
                raw_acro = parts[1].split("\n\n---\n\n")[0].split("## Sayfa")[0].strip()
                if raw_acro:
                    acro_section = f"Digital Form Fields (AcroForm):\n{raw_acro}\n\n"

        target_gen_limit = (
            min(settings.llm_max_tokens, 4096)
            if is_table
            else (
                settings.llm_thinking_max_tokens
                if use_thinking
                else settings.llm_single_field_max_tokens
            )
        )
        reserved_gen_room = min(target_gen_limit, 512)
        safety_margin = max(512, settings.model_n_ubatch * 2)

        total_image_tokens = 0
        if valid_images:
            for img_b in valid_images:
                try:
                    with Image.open(io.BytesIO(img_b)) as im:
                        w, h = im.size
                        toks = max(64, math.ceil(w / 32) * math.ceil(h / 32))
                        total_image_tokens += toks
                except Exception:
                    total_image_tokens += 1800

        max_prompt_ceiling = min(self.n_ctx - reserved_gen_room - safety_margin, 24000)

        clean_context = context.strip()
        has_real_text = any(
            line.strip() and not line.strip().startswith("[Görsel")
            for line in clean_context.splitlines()
        )

        if has_real_text and clean_context:
            wrapper_no_ctx = (
                f"{active_sys_prompt}\n\n"
                f'Document Context:\n"""\n\n"""\n\n'
                f"Examine both the document text excerpts and the attached page image(s):\n"
                f"- Extract character sequences, codes, names, dates, amounts verbatim.\n"
                f"- Inspect page images directly for checkboxes [X] and visual form layout.\n\n"
                f"{acro_section}"
                f"Question / Target Field: {target_query}{type_prompt}"
            )
            wrapper_tokens = len(self.llm.tokenize(wrapper_no_ctx.encode("utf-8")))
            max_ctx_budget = max(
                256,
                max_prompt_ceiling - wrapper_tokens - total_image_tokens,
            )

            ctx_tokens = self.llm.tokenize(clean_context.encode("utf-8"))
            if len(ctx_tokens) > max_ctx_budget:
                logger.warning(
                    "--> [LLM Multimodal] Field '%s' context tokens (%d) exceed budget (%d). "
                    "Truncating context.",
                    query_key,
                    len(ctx_tokens),
                    max_ctx_budget,
                )
                safe_ctx_tokens = ctx_tokens[:max_ctx_budget]
                clean_context = self.llm.detokenize(safe_ctx_tokens).decode(
                    "utf-8", errors="ignore"
                )

            instruction_text = (
                f"{active_sys_prompt}\n\n"
                f'Document Context:\n"""\n{clean_context}\n"""\n\n'
                f"Examine both the document text excerpts and the attached page image(s):\n"
                f"- Extract character sequences, codes, names, dates, amounts verbatim.\n"
                f"- Inspect page images directly for checkboxes [X] and visual form layout.\n\n"
                f"{acro_section}"
                f"Question / Target Field: {target_query}{type_prompt}"
            )
        else:
            instruction_text = (
                f"{active_sys_prompt}\n\n"
                f"Examine the attached high-resolution document page image(s) directly "
                f"to locate and extract the requested field value verbatim.\n\n"
                f"{acro_section}"
                f"Question / Target Field: {target_query}{type_prompt}"
            )

        user_content: list[dict[str, Any]] = [{"type": "text", "text": instruction_text}]
        for img_b in valid_images:
            b64_str = base64.b64encode(img_b).decode("utf-8")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"},
                }
            )

        messages = [{"role": "user", "content": user_content}]
        with self._lock:
            self.llm.reset()
            if hasattr(self.llm.chat_handler, "enable_thinking"):
                self.llm.chat_handler.enable_thinking = use_thinking
            actual_text_tokens = len(self.llm.tokenize(instruction_text.encode("utf-8")))
            total_prompt_tokens = total_image_tokens + actual_text_tokens + 16
            avail_tokens = max(
                16, min(target_gen_limit, self.n_ctx - total_prompt_tokens - safety_margin)
            )
            res = self.llm.create_chat_completion(
                messages=messages,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                max_tokens=avail_tokens,
                stop=settings.llm_stop_tokens,
            )

        msg_dict = res["choices"][0]["message"]
        raw_content = msg_dict.get("content") or ""
        reasoning_content = msg_dict.get("reasoning_content") or ""
        usage = res.get("usage", {})
        finish_reason = res["choices"][0].get("finish_reason")
        logger.info(
            "--> [LLM Multimodal Complete] field '%s' | finish=%s | tokens=%s | raw='%s'",
            query_key,
            finish_reason,
            usage,
            raw_content.replace("\n", "\\n")[:120],
        )
        if reasoning_content and not raw_content:
            thinking, raw_answer = reasoning_content, raw_content
        else:
            thinking, raw_answer = _parse_model_response(
                raw_content,
                started_in_thought=use_thinking,
                thought_open_tag=self._thought_open_tag,
                thought_close_tag=self._thought_close_tag,
            )

        is_found, normalized_val = _process_extracted_answer(
            raw_answer=raw_answer,
            expected_type=expected_type,
            items_schema=items_schema,
            query_key=query_key,
            query_desc=target_query,
            context=context,
        )

        _log_extraction_trace(
            query_key=query_key,
            thinking=thinking,
            value=normalized_val if is_found else "[Not found]",
            found=is_found,
        )

        return {
            "found": is_found,
            "value": normalized_val,
            "thinking": thinking,
        }

    def _extract_item_text(
        self,
        query_key: str,
        target_query: str,
        context: str,
        expected_type: str | None,
        items_schema: dict[str, Any] | None,
        use_thinking: bool,
        is_table: bool,
    ) -> LLMExtractionResult:
        """Execute text-based item extraction using dynamically synthesized prompt template."""
        if is_table:
            target_gen_limit = min(settings.llm_max_tokens, 4096)
            continuation_reserve = 0
        elif use_thinking:
            target_gen_limit = settings.llm_thinking_max_tokens
            continuation_reserve = 144
        else:
            target_gen_limit = settings.llm_single_field_max_tokens
            continuation_reserve = 0
        reserved_gen_room = min(target_gen_limit, 512) + continuation_reserve
        max_prompt_budget = max(256, self.n_ctx - reserved_gen_room - 64)

        logger.info("--> [LLM] Analyzing context for field '%s'...", query_key)
        with self._lock:
            prompt = _build_extraction_prompt(
                context=context,
                query=target_query,
                expected_type=expected_type,
                items_schema=items_schema,
                max_tokens=max_prompt_budget,
                llm=self.llm,
                enable_thinking=use_thinking,
                chat_formatter=self._chat_formatter,
                thought_open_tag=self._thought_open_tag,
            )
            tokens = self.llm.tokenize(prompt.encode("utf-8"))
            hard_limit = self.n_ctx - 32
            if len(tokens) >= hard_limit:
                raise ValueError(
                    f"Prompt tokens ({len(tokens)}) exceed context window limit ({hard_limit}) "
                    f"for field '{query_key}'."
                )

            avail_tokens = max(
                16,
                min(target_gen_limit, self.n_ctx - len(tokens) - continuation_reserve - 8),
            )

            res = self.llm(
                prompt=prompt,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                max_tokens=avail_tokens,
                stop=settings.llm_stop_tokens,
            )

        raw_content = res["choices"][0]["text"] or ""
        thinking, raw_answer = _parse_model_response(
            raw_content,
            started_in_thought=use_thinking,
            thought_open_tag=self._thought_open_tag,
            thought_close_tag=self._thought_close_tag,
        )

        if not raw_answer and thinking and use_thinking:
            cont_answer = self._attempt_continuation(prompt, thinking, query_key)
            if cont_answer:
                raw_answer = cont_answer

        is_found, normalized_val = _process_extracted_answer(
            raw_answer=raw_answer,
            expected_type=expected_type,
            items_schema=items_schema,
            query_key=query_key,
            query_desc=target_query,
            context=context,
        )

        _log_extraction_trace(
            query_key=query_key,
            thinking=thinking,
            value=normalized_val if is_found else "[Not found]",
            found=is_found,
        )

        return {
            "found": is_found,
            "value": normalized_val,
            "thinking": thinking,
        }

    def extract_item(
        self,
        query_key: str,
        query_description: str,
        context: str,
        image_bytes_list: list[bytes] | None = None,
        expected_type: str | None = None,
        items_schema: dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
    ) -> LLMExtractionResult:
        """Extract requested item from document context chunks using local neural model."""
        target_query = (query_description or query_key).strip()
        use_thinking = settings.llm_enable_thinking if enable_thinking is None else enable_thinking
        is_table = (expected_type or "").lower() in ("list", "array") or (items_schema is not None)

        valid_images = [b for b in (image_bytes_list or []) if b]
        if valid_images:
            return self._extract_item_multimodal(
                query_key=query_key,
                target_query=target_query,
                context=context,
                valid_images=valid_images,
                expected_type=expected_type,
                items_schema=items_schema,
                use_thinking=use_thinking,
                is_table=is_table,
            )

        return self._extract_item_text(
            query_key=query_key,
            target_query=target_query,
            context=context,
            expected_type=expected_type,
            items_schema=items_schema,
            use_thinking=use_thinking,
            is_table=is_table,
        )

    def extract_schema(
        self,
        fields: list[dict[str, Any]],
        context: str,
        image_bytes_list: list[bytes] | None = None,
        enable_thinking: bool | None = None,
    ) -> dict[str, LLMExtractionResult]:
        """Extract multiple schema items in a single unified LLM pass over consolidated context."""
        t_start = time.perf_counter()
        use_thinking = settings.llm_enable_thinking if enable_thinking is None else enable_thinking

        valid_images: list[bytes] = []
        if self.mmproj_path is not None and image_bytes_list:
            for img_bytes in image_bytes_list:
                if img_bytes and len(img_bytes) > 0:
                    valid_images.append(img_bytes)

        target_gen_limit = min(settings.llm_max_tokens, 4096)
        reserved_gen_room = min(target_gen_limit, 2048)
        safety_margin = max(512, settings.model_n_ubatch * 2)

        total_image_tokens = 0
        if valid_images:
            for img_b in valid_images:
                try:
                    with Image.open(io.BytesIO(img_b)) as im:
                        w, h = im.size
                        toks = max(64, math.ceil(w / 32) * math.ceil(h / 32))
                        total_image_tokens += toks
                except Exception:
                    total_image_tokens += 1800

        max_prompt_ceiling = min(self.n_ctx - reserved_gen_room - safety_margin, 24000)

        # Measure prompt wrapper without context
        wrapper_prompt = _build_schema_extraction_prompt(fields, "")
        wrapper_tokens = len(self.llm.tokenize(wrapper_prompt.encode("utf-8")))

        max_ctx_budget = max(
            256,
            max_prompt_ceiling - wrapper_tokens - total_image_tokens,
        )

        ctx_tokens = self.llm.tokenize(context.encode("utf-8"))
        if len(ctx_tokens) > max_ctx_budget:
            logger.warning(
                "--> [LLM Unified Schema] Context tokens (%d) exceed budget (%d). "
                "Truncating context chunks to fit context window.",
                len(ctx_tokens),
                max_ctx_budget,
            )
            delimiter = settings.extraction_context_delimiter
            raw_chunks = context.split(delimiter)
            budgeted_chunks: list[str] = []
            accum_tokens = 0
            for ch in raw_chunks:
                ch_tok_count = len(self.llm.tokenize(ch.encode("utf-8")))
                if accum_tokens + ch_tok_count <= max_ctx_budget:
                    budgeted_chunks.append(ch)
                    accum_tokens += ch_tok_count
                else:
                    rem_tokens = max_ctx_budget - accum_tokens
                    if rem_tokens >= 128 and not budgeted_chunks:
                        partial_toks = self.llm.tokenize(ch.encode("utf-8"))[:rem_tokens]
                        budgeted_chunks.append(
                            self.llm.detokenize(partial_toks).decode("utf-8", errors="ignore")
                        )
                    break
            context = delimiter.join(budgeted_chunks)

        full_user_text = _build_schema_extraction_prompt(fields, context)

        if valid_images:
            user_content: list[dict[str, Any]] = [{"type": "text", "text": full_user_text}]
            for img_b in valid_images:
                b64_str = base64.b64encode(img_b).decode("utf-8")
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"},
                    }
                )
            messages = [{"role": "user", "content": user_content}]
        else:
            messages = [{"role": "user", "content": full_user_text}]

        t_prep_elapsed = time.perf_counter() - t_start
        logger.info(
            "--> [LLM Unified Schema] Executing pass for %d fields (images=%d, prep=%.2fs)...",
            len(fields),
            len(valid_images),
            t_prep_elapsed,
        )

        t_inf_start = time.perf_counter()
        with self._lock:
            self.llm.reset()
            if hasattr(self.llm.chat_handler, "enable_thinking"):
                self.llm.chat_handler.enable_thinking = use_thinking
            actual_text_tokens = len(self.llm.tokenize(full_user_text.encode("utf-8")))
            total_prompt_tokens = total_image_tokens + actual_text_tokens + 16
            avail_tokens = max(
                64, min(target_gen_limit, self.n_ctx - total_prompt_tokens - safety_margin)
            )
            res = self.llm.create_chat_completion(
                messages=messages,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                max_tokens=avail_tokens,
                stop=settings.llm_stop_tokens,
            )
        t_inf_elapsed = max(time.perf_counter() - t_inf_start, 0.001)

        msg_obj = res["choices"][0]["message"]
        reasoning_content = msg_obj.get("reasoning_content")
        raw_content = msg_obj.get("content") or ""

        usage = res.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        tok_speed = completion_tokens / t_inf_elapsed

        t_post_start = time.perf_counter()
        out = _process_schema_extraction_output(
            raw_content=raw_content,
            fields=fields,
            context=context,
            reasoning_content=reasoning_content,
        )
        t_post_elapsed = time.perf_counter() - t_post_start
        t_total = time.perf_counter() - t_start

        logger.info(
            "--> [LLM SCHEMA TIMING] Input Prep: %.2fs | "
            "Chat Completion: %.2fs (%d prompt tok, %d gen tok @ %.1f tok/s) | "
            "Post-process: %.2fs | Total Pass: %.2fs",
            t_prep_elapsed,
            t_inf_elapsed,
            prompt_tokens,
            completion_tokens,
            tok_speed,
            t_post_elapsed,
            t_total,
        )
        return out


# Backward-compatible alias for external imports
LocalGemmaEngine = LocalLLMEngine

_local_engine_instance: LocalLLMEngine | None = None
_local_engine_lock = threading.Lock()


def get_local_engine() -> LocalLLMEngine:
    """Return the process-wide cached LocalLLMEngine singleton."""
    global _local_engine_instance
    if _local_engine_instance is None:
        with _local_engine_lock:
            if _local_engine_instance is None:
                model_path = _resolve_default_model_path()
                _local_engine_instance = LocalLLMEngine(model_path=model_path)
    return _local_engine_instance
