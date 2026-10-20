"""Local in-process Gemma 4 E2B QAT engine for thinking-enabled information extraction."""

import logging
import re
import threading
from pathlib import Path
from typing import Any, TypedDict

from docneedle.config import settings

logger = logging.getLogger(__name__)

TYPE_GUIDELINES: dict[str, dict[str, str]] = {
    "string": {
        "description": "Plain text or expression",
        "example": "John Doe",
    },
    "number": {
        "description": "Numeric value",
        "example": "42 or 1500",
    },
    "money": {
        "description": "Monetary amount and currency",
        "example": "$25,000 or 1,500 USD or 25.000 TL",
    },
    "date": {
        "description": "Date information (day, month, year)",
        "example": "June 15, 2026 or 15.05.1992",
    },
    "time": {
        "description": "Time of day",
        "example": "14:30 or 09:00 AM",
    },
    "email": {
        "description": "Email address",
        "example": "example@company.com",
    },
    "phone": {
        "description": "Phone number",
        "example": "+1 555 123 4567 or 0532 123 45 67",
    },
    "id": {
        "description": "ID number, tax ID, registration or passport number",
        "example": "22334455667",
    },
    "iban": {
        "description": "Bank account / IBAN number",
        "example": "TR12 0006 1005 1234 5678 9012 34",
    },
    "percentage": {
        "description": "Percentage rate",
        "example": "18% or 20.5%",
    },
    "person": {
        "description": "Full name of a person",
        "example": "John Smith",
    },
    "organization": {
        "description": "Company, institution or legal entity title",
        "example": "Acme Technology Inc.",
    },
    "address": {
        "description": "Physical address or location",
        "example": "123 Main Street, Suite 400, New York, NY",
    },
    "boolean": {
        "description": "Yes/No or True/False confirmation status",
        "example": "Yes or No",
    },
    "list": {
        "description": "Comma-separated or bulleted list",
        "example": "Item 1, Item 2, Item 3",
    },
}

BASE_RULES = (
    "You are a rigorous official document analysis assistant answering questions "
    "based strictly on the provided document excerpts.\n"
    "Rules:\n"
    "1. Only use information explicitly stated in the provided document text. "
    "NEVER calculate, divide, or extrapolate mathematical numbers. "
    "Extract the exact stated amount and currency verbatim as written in the text.\n"
    "2. If the requested information specifically targets a distinct role or third party "
    "(e.g., 'sender', 'recipient', 'guarantor', 'mother's name', 'proxy', 'seller'), "
    "extract the exact value corresponding to that specific role.\n"
    "3. When a general or plain field is requested (e.g., simply 'name', 'surname', 'location', "
    "'address', 'ID number'), extract the primary, definitive value belonging to the subject "
    "of the document, rather than incidental third parties in tables or bank/corporate "
    "headquarters addresses.\n"
    "4. Domain specifics:\n"
    "   - In rental contracts, 'Kira Bedeli' represents the agreed rent / monthly rent "
    "('aylık kira bedeli'). Extract this stated amount directly (e.g., '25.000 TL').\n"
    "   - For first name ('ad' / 'isim'), extract ONLY the given first name "
    "(e.g., in 'AD, SOYAD : TOLGA, ARSLAN' or 'TOLGA ARSLAN', first name is 'TOLGA').\n"
    "   - For surname ('soyad' / 'soyisim'), extract ONLY the family surname "
    "(e.g., in 'AD, SOYAD : TOLGA, ARSLAN' or 'TOLGA ARSLAN', surname is 'ARSLAN').\n"
    "   - NEVER concatenate adjacent column values, parent names ('baba adı', 'ana adı'), "
    "gender, dates, or locations into the person's name.\n"
    "5. If the requested information is not present in the document at all, "
    'reply strictly with "Not found".\n'
)

THINKING_RULE = (
    "6. Keep your reasoning inside <think>...</think> concise (1-3 sentences). "
    "Then immediately close with </think> and output ONLY the final extracted value "
    "(or 'Not found') directly without markdown headers, prefixes (such as '**Final Answer:**', "
    "'**Output:**'), or filler.\n"
)

NON_THINKING_RULE = (
    "6. Do NOT output any thinking, reasoning, explanations, markdown headers, prefixes, or "
    "filler. Output strictly and ONLY the final extracted value (or 'Not found') directly.\n"
)

SYSTEM_PROMPT = BASE_RULES + THINKING_RULE

_NOT_FOUND_TERMS = {
    "not found",
    "none",
    "n/a",
    "null",
    "false",
    "bulunamadı",
    "bulunamadi",
    "yok",
    "yoktur",
}

_THOUGHT_CLOSE_PATTERN = re.compile(
    r"(?:<channel\|>|<\|channel\|>|<\|channel>text|</channel>|"
    r"</thought>|</think>|<channel>|\[END OF THOUGHT\]|\[/THINK\])",
    re.IGNORECASE,
)
_CONTROL_TOKENS_PATTERN = re.compile(
    r"(?:<\|turn>|<turn\|>|<\|channel>(?:text|thought)?|<channel\|>|<\|channel\|>|</channel>|"
    r"</thought>|</think>|<end_of_turn>|\[SYSTEM_PROMPT\]|\[/SYSTEM_PROMPT\]|\[INST\]|\[/INST\]|"
    r"\[THINK\]|\[/THINK\]|<s>|</s>)",
    re.IGNORECASE,
)
_ANSWER_PREFIX_PATTERN = re.compile(
    r"^(?:<\|channel>(?:text)?\s*)?(?:Answer|Cevap|Result|Sonuç(?:\s+olarak)?|Value|Değer|Final\s+Answer|Nihai\s+Cevap):\s*",
    re.IGNORECASE,
)
_CODE_BLOCK_PATTERN = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)


class LLMExtractionResult(TypedDict):
    """Normalized output contract of LLM item extraction."""

    found: bool
    value: str | None
    thinking: str | None


def _is_mistral_model() -> bool:
    name = getattr(settings, "llm_model_name", "").lower()
    return "ministral" in name or "mistral" in name


def _build_extraction_prompt(
    context: str,
    query: str,
    expected_type: str | None = None,
    max_tokens: int | None = None,
    llm: Any = None,
    enable_thinking: bool | None = None,
) -> str:
    """Construct prompt adhering to Ministral/Mistral or Gemma turn templates."""
    type_prompt = ""
    if expected_type and expected_type.lower() in TYPE_GUIDELINES:
        guide = TYPE_GUIDELINES[expected_type.lower()]
        type_prompt = (
            f"\nExpected Data Type: {expected_type} ({guide['description']})\n"
            f"Example Output Format: {guide['example']}"
        )

    use_thinking = settings.llm_enable_thinking if enable_thinking is None else enable_thinking
    active_sys_prompt = BASE_RULES + (THINKING_RULE if use_thinking else NON_THINKING_RULE)

    if _is_mistral_model():
        header = f'[SYSTEM_PROMPT]{active_sys_prompt}[/SYSTEM_PROMPT][INST]Document Context:\n"""\n'
        footer = f'\n"""\n\nQuestion / Target Field: {query}{type_prompt}\n[/INST]'
        turn_close = "<think>\n" if use_thinking else ""
    else:
        header = f'<|turn>system\n{active_sys_prompt}<turn|>\n<|turn>user\nDocument Context:\n"""\n'
        footer = f'\n"""\n\nQuestion / Target Field: {query}{type_prompt}'
        turn_close = (
            "<turn|>\n<|turn>model\n<|channel>thought\n"
            if use_thinking
            else "<turn|>\n<|turn>model\n"
        )

    if llm is not None and max_tokens is not None:
        try:
            head_tokens = llm.tokenize(header.encode("utf-8"))
            foot_tokens = llm.tokenize((footer + turn_close).encode("utf-8"))
            avail_ctx_budget = max_tokens - len(head_tokens) - len(foot_tokens)
            if avail_ctx_budget > 64:
                ctx_tokens = llm.tokenize(context.encode("utf-8"))
                if len(ctx_tokens) > avail_ctx_budget:
                    logger.warning(
                        f"Document context tokens ({len(ctx_tokens)}) exceed budget "
                        f"({avail_ctx_budget}). Truncating lowest-priority context chunks."
                    )
                    safe_ctx_tokens = ctx_tokens[:avail_ctx_budget]
                    context = llm.detokenize(safe_ctx_tokens).decode("utf-8", errors="ignore")
        except Exception as e:
            logger.debug(f"Prompt context token budget calculation bypassed: {e}")

    return f"{header}{context}{footer}{turn_close}"


def _extract_prefixed_answer(lines: list[str]) -> tuple[str | None, list[str]]:
    """Scan lines in reverse for answer prefixes (e.g., 'Cevap: ...', 'Answer: ...').

    Returns a tuple of (extracted_answer, prior_reasoning_lines).
    """
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        match = _ANSWER_PREFIX_PATTERN.match(line)
        if match:
            answer = line[match.end() :].strip()
            prior_lines = lines[:idx]
            return answer, prior_lines
    return None, lines


def _parse_model_response(
    raw_text: str,
    started_in_thought: bool = False,
) -> tuple[str | None, str]:
    """Separate model thinking reasoning from final answer text."""
    text = raw_text.strip()
    thinking: str | None = None
    answer: str = text

    match = _THOUGHT_CLOSE_PATTERN.search(text)
    if match:
        thinking_part = (
            text[: match.start()].replace("<|channel>thought", "").replace("<think>", "").strip()
        )
        answer_part = text[match.end() :].strip()
        thinking = thinking_part if thinking_part else None
        answer = answer_part
    elif started_in_thought or text.startswith("<|channel>thought") or text.startswith("<think>"):
        # Model generated inside thought channel but never reached or emitted a closing tag
        cleaned_text = text.replace("<|channel>thought", "").replace("<think>", "").strip()
        lines = [line.strip() for line in cleaned_text.split("\n") if line.strip()]
        extracted_answer, extracted_thinking_lines = _extract_prefixed_answer(lines)

        if extracted_answer is not None:
            thinking = "\n".join(extracted_thinking_lines).strip() or None
            answer = extracted_answer
        else:
            # Entire text is unclosed reasoning trace; do not leak as answer
            thinking = cleaned_text or None
            answer = ""
    else:
        # Direct answer without thinking
        thinking = None
        answer = text

    # Salvage answer if answer portion is empty but thinking ended with an answer line
    if not answer and thinking:
        lines = [line.strip() for line in thinking.split("\n") if line.strip()]
        extracted_answer, extracted_thinking_lines = _extract_prefixed_answer(lines)
        if extracted_answer is not None:
            answer = extracted_answer
            thinking = "\n".join(extracted_thinking_lines).strip() or None

    return thinking, answer


def _sanitize_extracted_value(text: str) -> str:
    """Clean markdown code blocks, control tokens, prefixes, and formatting."""
    cleaned = text.strip()
    cleaned = _CONTROL_TOKENS_PATTERN.sub("", cleaned).strip()

    # Strip markdown horizontal rules at start/end
    cleaned = re.sub(r"^(?:---|\*\*\*|___)\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s*(?:---|\*\*\*|___)$", "", cleaned).strip()

    # Strip markdown bold/italic wrapper on answer prefixes (e.g. **Answer:**, **Final Answer:**)
    prefix_re = re.compile(
        r"^(?:[-*]\s*)?(?:\*{1,2})?(?:Answer|Cevap|Result|Sonuç(?:\s+olarak)?|Value|Değer|Output|Çıktı|Final\s+Answer|Nihai\s+Cevap):(?:\*{1,2})?\s*",
        re.IGNORECASE,
    )
    cleaned = prefix_re.sub("", cleaned).strip()

    if cleaned.startswith("```"):
        cleaned = _CODE_BLOCK_PATTERN.sub("", cleaned).strip()

    cleaned = prefix_re.sub("", cleaned).strip()
    cleaned = _CONTROL_TOKENS_PATTERN.sub("", cleaned).strip()

    # If multi-line, check if there's a final answer line
    lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
    if len(lines) > 1:
        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx]
            match = prefix_re.match(line)
            if match:
                cleaned = line[match.end() :].strip()
                break
        else:
            if lines[0].startswith(("---", "#", "**Final")):
                cleaned = lines[-1].strip()

    # Strip surrounding markdown bold / italic formatting if applied to whole answer
    if (cleaned.startswith("**") and cleaned.endswith("**")) or (
        cleaned.startswith("*") and cleaned.endswith("*")
    ):
        cleaned = cleaned.strip("*").strip()
    return cleaned


def _is_not_found_value(text: str, expected_type: str | None = None) -> bool:
    """Determine whether cleaned text indicates that the queried entity was not found."""
    norm = text.lower().strip("\"'., ")
    if not norm:
        return True
    if expected_type and expected_type.lower() == "boolean" and norm == "false":
        return False
    if norm in _NOT_FOUND_TERMS:
        return True
    for term in _NOT_FOUND_TERMS:
        if norm.endswith(term):
            return True
    return False


def _log_extraction_trace(query_key: str, thinking: str | None, value: str) -> None:
    """Log model extraction trace for debugging and audit."""
    sep = "=" * 25
    if thinking:
        logger.info(
            f"\n{sep} [MODEL THINKING ({query_key})] {sep}\n{thinking.strip()}\n"
            f"{sep} [MODEL ANSWER ({query_key})] {sep}\n{value.strip()}\n{'=' * 75}"
        )
    else:
        logger.info(f"\n{sep} [MODEL OUTPUT ({query_key})] {sep}\n{value.strip()}\n{'=' * 75}")


def _resolve_default_model_path() -> Path:
    """Find the configured GGUF model file in the model storage directory."""
    base = Path(settings.model_storage_dir) / "llm"
    model_name = getattr(settings, "llm_model_name", "gemma-4-E2B-it-qat-UD-Q4_K_XL")
    candidate = base / f"{model_name}.gguf"
    if candidate.exists():
        return candidate
    if base.exists():
        # First look for model file containing model_name substring
        for f in base.glob("*.gguf"):
            if not f.name.startswith("mtp") and model_name.lower() in f.name.lower():
                return f
        # Otherwise fallback to any valid GGUF
        ggufs = [f for f in base.glob("*.gguf") if not f.name.startswith("mtp")]
        if ggufs:
            return ggufs[0]
    return candidate


class LocalGemmaEngine:
    """Manages in-process LLM (Ministral / Gemma) GGUF inference with thinking support."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        from llama_cpp import Llama

        if model_path is None:
            model_path = _resolve_default_model_path()

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"GGUF model file not found at: {self.model_path}")

        logger.info(f"Loading local LLM ({self.model_path.name}) from {self.model_path}...")
        logits_all = False
        draft_model = None
        use_speculative = settings.llm_speculative_enabled and (
            settings.gpu_enabled or getattr(settings, "llm_speculative_cpu_enabled", False)
        )
        if use_speculative:
            from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

            num_pred = (
                settings.llm_speculative_num_pred_tokens
                if not settings.gpu_enabled
                else max(10, settings.llm_speculative_num_pred_tokens)
            )
            draft_model = LlamaPromptLookupDecoding(
                max_ngram_size=settings.llm_speculative_max_ngram_size,
                num_pred_tokens=num_pred,
            )
            logger.info(
                "Prompt Lookup Decoding enabled: max_ngram_size=%d, num_pred_tokens=%d",
                settings.llm_speculative_max_ngram_size,
                num_pred,
            )

        llm_ctx = settings.llm_n_ctx
        if draft_model is not None and llm_ctx > 4096:
            llm_ctx = 4096

        self.n_ctx = llm_ctx

        self.llm = Llama(
            model_path=str(self.model_path),
            draft_model=draft_model,
            logits_all=logits_all,
            n_ctx=llm_ctx,
            n_gpu_layers=settings.effective_gpu_layers,
            n_threads=settings.cpu_threads,
            n_threads_batch=settings.cpu_threads_batch,
            n_batch=settings.model_n_batch,
            type_k=settings.model_kv_cache_type_k,
            type_v=settings.model_kv_cache_type_v,
            flash_attn=settings.model_flash_attn,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        self._lock = threading.Lock()
        logger.info(f"Local LLM ({self.model_path.name}) loaded successfully (n_ctx={self.n_ctx}).")

    def _attempt_continuation(self, prompt: str, thinking: str, query_key: str) -> str | None:
        """Execute a fast continuation turn when reasoning terminated without closing thought."""
        logger.info(
            f"Model reasoning did not output closed answer for '{query_key}'. "
            "Running fast continuation to extract final answer..."
        )
        if _is_mistral_model():
            continuation_prompt = f"{prompt}{thinking.strip()}\n</think>\n"
        else:
            continuation_prompt = f"{prompt}{thinking.strip()}\n<channel|>\n"
        try:
            with self._lock:
                cont_res = self.llm(
                    prompt=continuation_prompt,
                    temperature=settings.llm_temperature,
                    top_p=settings.llm_top_p,
                    max_tokens=128,
                    stop=settings.llm_stop_tokens,
                )
            cont_text = cont_res["choices"][0]["text"] or ""
            _, cont_answer = _parse_model_response(cont_text, started_in_thought=False)
            return cont_answer or None
        except Exception as cont_err:
            logger.warning(f"Continuation extraction failed for '{query_key}': {cont_err}")
            return None

    def extract_item(
        self,
        query_key: str,
        query_description: str,
        context: str,
        expected_type: str | None = None,
        enable_thinking: bool | None = None,
    ) -> LLMExtractionResult:
        """Extract requested item from document context chunks using local neural model."""
        target_query = (query_description or query_key).strip()
        use_thinking = settings.llm_enable_thinking if enable_thinking is None else enable_thinking

        # Compute safe prompt budget respecting self.n_ctx while reserving space for generation
        if use_thinking:
            target_gen_limit = settings.llm_thinking_max_tokens
        else:
            target_gen_limit = settings.llm_single_field_max_tokens
        reserved_gen_room = min(target_gen_limit, 512)
        max_prompt_budget = max(256, self.n_ctx - reserved_gen_room - 64)

        prompt = _build_extraction_prompt(
            context=context,
            query=target_query,
            expected_type=expected_type,
            max_tokens=max_prompt_budget,
            llm=self.llm,
            enable_thinking=use_thinking,
        )

        logger.info(f"--> [LLM] Gemma 4 analyzing context for field '{query_key}'...")
        try:
            with self._lock:
                tokens = self.llm.tokenize(prompt.encode("utf-8"))
                hard_limit = self.n_ctx - 16
                if len(tokens) >= hard_limit:
                    logger.warning(
                        f"Prompt tokens ({len(tokens)}) reach hard limit ({hard_limit}). "
                        "Applying emergency truncation to prevent context window overflow."
                    )
                    tokens = tokens[: hard_limit - 1]
                    prompt = self.llm.detokenize(tokens).decode("utf-8", errors="ignore")

                # Dynamically size max_tokens to stay within context bounds and target field limit
                avail_tokens = max(32, min(target_gen_limit, self.n_ctx - len(tokens) - 8))

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
            )

            # If the model produced thinking but did not reach the answer, attempt continuation
            if not raw_answer and thinking and use_thinking:
                cont_answer = self._attempt_continuation(prompt, thinking, query_key)
                if cont_answer:
                    raw_answer = cont_answer

            cleaned_val = _sanitize_extracted_value(raw_answer)

            _log_extraction_trace(query_key=query_key, thinking=thinking, value=cleaned_val)

            if _is_not_found_value(cleaned_val, expected_type=expected_type):
                return {"found": False, "value": None, "thinking": thinking}

            return {
                "found": True,
                "value": cleaned_val,
                "thinking": thinking,
            }

        except Exception as e:
            logger.error("Error during LLM extraction: %s", e, exc_info=True)
            return {"found": False, "value": None, "thinking": None}


_local_engine_instance: LocalGemmaEngine | None = None
_local_engine_lock = threading.Lock()


def get_local_engine() -> LocalGemmaEngine | None:
    """Thread-safe singleton getter for the in-process local engine."""
    global _local_engine_instance
    if _local_engine_instance is None:
        with _local_engine_lock:
            if _local_engine_instance is None:
                model_path = _resolve_default_model_path()
                if model_path.exists():
                    try:
                        _local_engine_instance = LocalGemmaEngine(model_path=model_path)
                    except Exception as e:
                        logger.warning(f"Could not initialize LocalGemmaEngine: {e}")
    return _local_engine_instance
