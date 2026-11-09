"""Response parsing, cleanup, normalization, and hallucination detection for LLM output."""

import json
import logging
import re
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

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
_BOOLEAN_VALID_NEGATIVES = {"false", "no", "yok", "yoktur", "none", "0", "hayır", "hayir"}
_NOT_FOUND_SUFFIX_PATTERN = re.compile(
    r"(?:^|\b)(?:"
    + "|".join(re.escape(term) for term in sorted(_NOT_FOUND_TERMS, key=len, reverse=True))
    + r")$",
    re.IGNORECASE,
)

_THOUGHT_OPEN_PATTERN = re.compile(
    r"^(?:<\|channel>thought|<think>|<thought>|\[THINK\])\s*",
    re.IGNORECASE,
)
_THOUGHT_CLOSE_PATTERN = re.compile(
    r"(?:<channel\|>|<\|channel\|>|<\|channel>(?:text|call)|</channel>|"
    r"</thought>|</think>|<channel>|\[END OF THOUGHT\]|\[/THINK\])",
    re.IGNORECASE,
)
_CONTROL_TOKENS_PATTERN = re.compile(
    r"(?:<\|turn>|<turn\|>|<\|channel>(?:text|thought|call)?|<channel\|>|<\|channel\|>|</channel>|"
    r"<think>|</think>|<thought>|</thought>|<end_of_turn>|<\|im_start\|>|<\|im_end\|>|<\|eot_id\|>|"
    r"\[SYSTEM_PROMPT\]|\[/SYSTEM_PROMPT\]|\[INST\]|\[/INST\]|"
    r"\[THINK\]|\[/THINK\]|<s>|</s>|<bos>|<eos>)",
    re.IGNORECASE,
)
_ANSWER_PREFIX_PATTERN = re.compile(
    r"^(?:<\|channel>(?:text)?\s*)?(?:[-*]\s*)?(?:\*{1,2})?"
    r"(?:Answer|Cevap|Result|Sonuç(?:\s+olarak)?|Value|Değer|Output|Çıktı|"
    r"Final\s+Answer|Nihai\s+Cevap):(?:\*{1,2})?\s*",
    re.IGNORECASE,
)
_CODE_BLOCK_PATTERN = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)
_HORIZONTAL_RULE_START_PATTERN = re.compile(r"^(?:---|\*\*\*|___)\s*")
_HORIZONTAL_RULE_END_PATTERN = re.compile(r"\s*(?:---|\*\*\*|___)$")


class LLMExtractionResult(TypedDict):
    """Normalized output contract of LLM item extraction."""

    found: bool
    value: Any | None
    thinking: str | None


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


def _strip_thought_open_tag(text: str, custom_open_tag: str | None = None) -> str:
    cleaned = text.strip()
    if custom_open_tag and custom_open_tag.strip():
        cleaned = cleaned.replace(custom_open_tag.strip(), "")
    cleaned = _THOUGHT_OPEN_PATTERN.sub("", cleaned.strip())
    return cleaned.strip()


def _parse_model_response(
    raw_text: str,
    started_in_thought: bool = False,
    thought_open_tag: str | None = None,
    thought_close_tag: str | None = None,
) -> tuple[str | None, str]:
    """Separate model thinking reasoning from final answer text."""
    text = raw_text.strip()
    thinking: str | None = None
    answer: str = text

    close_match = None
    if thought_close_tag and thought_close_tag.strip() and thought_close_tag.strip() in text:
        tag = thought_close_tag.strip()
        pos = text.find(tag)
        close_span = (pos, pos + len(tag))
    else:
        close_match = _THOUGHT_CLOSE_PATTERN.search(text)
        close_span = (close_match.start(), close_match.end()) if close_match else None

    has_open_tag = bool(_THOUGHT_OPEN_PATTERN.match(text)) or (
        bool(thought_open_tag and thought_open_tag.strip())
        and text.startswith(thought_open_tag.strip())
    )

    if close_span is not None:
        thinking_part = _strip_thought_open_tag(text[: close_span[0]], thought_open_tag)
        answer_part = text[close_span[1] :].strip()
        thinking = thinking_part if thinking_part else None
        answer = answer_part
    elif started_in_thought or has_open_tag:
        # Model generated inside thought channel but never reached or emitted a closing tag
        cleaned_text = _strip_thought_open_tag(text, thought_open_tag)
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
    cleaned = _CONTROL_TOKENS_PATTERN.sub("", text.strip()).strip()
    cleaned = _HORIZONTAL_RULE_START_PATTERN.sub("", cleaned).strip()
    cleaned = _HORIZONTAL_RULE_END_PATTERN.sub("", cleaned).strip()
    cleaned = _ANSWER_PREFIX_PATTERN.sub("", cleaned).strip()

    if cleaned.startswith("```"):
        cleaned = _CODE_BLOCK_PATTERN.sub("", cleaned).strip()
        cleaned = _ANSWER_PREFIX_PATTERN.sub("", cleaned).strip()
        cleaned = _CONTROL_TOKENS_PATTERN.sub("", cleaned).strip()

    # If multi-line, check if there's a final answer line
    lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
    if len(lines) > 1:
        extracted, _ = _extract_prefixed_answer(lines)
        if extracted is not None:
            cleaned = extracted
        elif lines[0].startswith(("---", "#", "**Final")):
            cleaned = lines[-1]

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
    if expected_type and expected_type.lower() == "boolean" and norm in _BOOLEAN_VALID_NEGATIVES:
        return False
    if norm in _NOT_FOUND_TERMS:
        return True
    return bool(_NOT_FOUND_SUFFIX_PATTERN.search(norm))


def _try_parse_json_value(raw_text: str) -> tuple[bool, Any]:
    """Attempt to parse a structured JSON array or object from model response."""
    text = raw_text.strip()
    if not text:
        return False, None

    # Strip code block wrappers if present
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()

    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        json_candidate = text[first_bracket : last_bracket + 1]
        try:
            parsed = json.loads(json_candidate)
            if isinstance(parsed, list):
                return True, parsed
        except Exception:
            pass

    # Salvage truncated JSON arrays if token limit cut off before closing ']'
    if first_bracket != -1:
        last_brace = text.rfind("}")
        if last_brace != -1 and last_brace > first_bracket:
            candidate = text[first_bracket : last_brace + 1] + "\n]"
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return True, parsed
            except Exception:
                pass

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = text[first_brace : last_brace + 1]
        try:
            parsed = json.loads(json_candidate)
            if isinstance(parsed, dict):
                return True, parsed
        except Exception:
            pass

    return False, None


def _clean_json_placeholders(obj: Any) -> Any:
    """Recursively convert null-equivalent string placeholders to Python None."""
    if isinstance(obj, dict):
        return {k: _clean_json_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json_placeholders(item) for item in obj]
    if isinstance(obj, str):
        s = obj.strip().lower()
        if s in ("null", "none", "n/a", "na", "-", "--", "nil"):
            return None
        return obj
    return obj


def _try_parse_schema_json(raw_text: str) -> tuple[bool, Any]:
    """Parse a dictionary or list mapping from model output with cleaning and repairs."""
    text = raw_text.strip()
    if not text:
        return False, {}

    # Strip code block wrappers if present
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")

    # Check whether an outer array [...] encloses the braces or exists alone
    is_outer_array = (
        first_bracket != -1
        and last_bracket != -1
        and last_bracket > first_bracket
        and (first_brace == -1 or first_bracket < first_brace)
        and (last_brace == -1 or last_bracket > last_brace)
    )

    if is_outer_array:
        json_candidate = text[first_bracket : last_bracket + 1]
        cleaned_candidate = re.sub(r",\s*([\}\]])", r"\1", json_candidate)
        try:
            parsed = json.loads(cleaned_candidate, strict=False)
            if isinstance(parsed, (dict, list)):
                return True, parsed
        except Exception:
            try:
                parsed = json.loads(json_candidate, strict=False)
                if isinstance(parsed, (dict, list)):
                    return True, parsed
            except Exception:
                pass

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = text[first_brace : last_brace + 1]
        cleaned_candidate = re.sub(r",\s*([\}\]])", r"\1", json_candidate)
        try:
            parsed = json.loads(cleaned_candidate, strict=False)
            if isinstance(parsed, (dict, list)):
                return True, parsed
        except Exception:
            try:
                parsed = json.loads(json_candidate, strict=False)
                if isinstance(parsed, (dict, list)):
                    return True, parsed
            except Exception:
                pass

    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        json_candidate = text[first_bracket : last_bracket + 1]
        cleaned_candidate = re.sub(r",\s*([\}\]])", r"\1", json_candidate)
        try:
            parsed = json.loads(cleaned_candidate, strict=False)
            if isinstance(parsed, (dict, list)):
                return True, parsed
        except Exception:
            try:
                parsed = json.loads(json_candidate, strict=False)
                if isinstance(parsed, (dict, list)):
                    return True, parsed
            except Exception:
                pass

    return False, {}


def _rescue_schema_fields_from_text(raw_text: str, fields: list[dict[str, Any]]) -> dict[str, Any]:
    """Fallback regex parser to extract key-value pairs when strict JSON decoding fails."""
    rescued: dict[str, Any] = {}
    for f in fields:
        k = f["key"]
        pattern = re.compile(
            rf'"{re.escape(k)}"\s*:\s*(?P<val>".*?(?<!\\)"|-?\d+(?:\.\d+)?|true|false|null|\[.*?\]|\{{.*?\}})',
            re.DOTALL | re.IGNORECASE,
        )
        m = pattern.search(raw_text)
        if m:
            val_str = m.group("val").strip()
            if val_str.startswith(("[", "{")):
                val_str = re.sub(r",\s*([\}\]])", r"\1", val_str)
            try:
                rescued[k] = json.loads(val_str, strict=False)
            except Exception:
                lower = val_str.lower()
                if lower == "true":
                    rescued[k] = True
                elif lower == "false":
                    rescued[k] = False
                elif lower == "null":
                    rescued[k] = None
                else:
                    rescued[k] = val_str.strip('"')
    return rescued


def _is_label_hallucination(
    val_str: str,
    query_key: str | None,
    query_desc: str | None = None,
) -> bool:
    """Check if extracted value merely repeats the form label / field key instead of content."""
    if not val_str:
        return False
    clean_val = re.sub(r"^[\d\.\s\-_#]+", "", val_str).strip().lower()
    clean_val = re.sub(r"[\s\-_]+", " ", clean_val)
    if not clean_val:
        return False

    if query_key:
        clean_key = re.sub(r"^[\d\.\s\-_#]+", "", query_key.lower()).strip()
        clean_key = re.sub(r"[\s\-_]+", " ", clean_key)
        if clean_val == clean_key:
            return True
        key_words = set(re.findall(r"\w{3,}", clean_key))
        val_words = set(re.findall(r"\w{3,}", clean_val))
        if val_words and val_words.issubset(key_words):
            return True

    if query_desc:
        clean_desc = re.sub(r"^[\d\.\s\-_#]+", "", query_desc.lower()).strip()
        clean_desc = re.sub(r"[\s\-_]+", " ", clean_desc)
        if clean_val == clean_desc:
            return True
        desc_words = set(re.findall(r"\w{3,}", clean_desc))
        val_words = set(re.findall(r"\w{3,}", clean_val))
        if val_words and val_words.issubset(desc_words) and len(val_words) >= 2:
            if clean_val.endswith(
                ("method", "specify", "type", "description", "details", "explanation", "name")
            ):
                return True

    return False


def _is_copied_from_prompt_example(
    val_str: str,
    query_desc: str | None,
    context: str | None,
) -> bool:
    """Detect if the LLM copied an example from prompt query that is absent in context."""
    if not query_desc or not context or not val_str or len(val_str.strip()) < 3:
        return False
    # For visual/scanned pages, text is inside images, not in context string
    if "[Görsel Belge" in context or "[Görsel Doküman" in context:
        return False
    norm_val = val_str.strip().casefold()
    norm_ctx = context.casefold()
    if norm_val in norm_ctx:
        return False
    norm_desc = query_desc.casefold()
    return norm_val in norm_desc


def _strip_field_prefix(val: str, query_key: str | None, query_desc: str | None) -> str:
    """Strip redundant field name or description prefixes from extracted scalar values."""
    if not val or not isinstance(val, str):
        return val

    prefixes_to_try: list[str] = []
    if query_desc:
        prefixes_to_try.append(query_desc.strip())
    if query_key:
        prefixes_to_try.append(query_key.replace("_", " ").strip())
        prefixes_to_try.append(query_key.strip())

    val_stripped = val.strip()
    for p in prefixes_to_try:
        norm_p = p.lower()
        if not norm_p or len(norm_p) < 3:
            continue
        pattern = re.compile(rf"^{re.escape(norm_p)}\s*[:\-–—]\s*", re.IGNORECASE)
        m = pattern.match(val_stripped)
        if m:
            candidate = val_stripped[m.end() :].strip()
            if candidate:
                val_stripped = candidate
                break
    return val_stripped


def _normalize_typed_value(
    val: Any,
    expected_type: str | None,
    items_schema: dict[str, Any] | None = None,
    query_key: str | None = None,
    query_desc: str | None = None,
    context: str | None = None,
) -> tuple[bool, Any]:
    """Normalize extracted value against expected data type contract."""
    if isinstance(val, (list, dict)):
        if len(val) == 0:
            return False, None
        if isinstance(val, list):
            has_substance = False
            for r in val:
                if isinstance(r, dict):
                    if any(
                        v is not None and str(v).strip() not in ("", "null", "none")
                        for v in r.values()
                    ):
                        has_substance = True
                        break
                elif r is not None and str(r).strip() not in ("", "null", "none"):
                    has_substance = True
                    break
            if not has_substance:
                return False, None

            if items_schema and isinstance(items_schema, dict):
                expected_keys = items_schema.get("properties", {}).keys()
                if expected_keys:
                    norm_rows = []
                    for row in val:
                        if isinstance(row, dict):
                            # Heal flattened value_1, value_2 keys if values array is missing
                            if "values" in expected_keys and "values" not in row:
                                val_keys = [
                                    k
                                    for k in row.keys()
                                    if k.startswith("value_") or k.startswith("val_")
                                ]
                                if val_keys:
                                    val_keys.sort(
                                        key=lambda x: (
                                            int(re.search(r"\d+", x).group())
                                            if re.search(r"\d+", x)
                                            else 0
                                        )
                                    )
                                    row["values"] = [row[vk] for vk in val_keys]
                            # Also heal nested rows in table objects
                            if "rows" in row and isinstance(row["rows"], list):
                                for r in row["rows"]:
                                    if isinstance(r, dict) and "values" not in r:
                                        r_val_keys = [
                                            k
                                            for k in r.keys()
                                            if k.startswith("value_") or k.startswith("val_")
                                        ]
                                        if r_val_keys:
                                            r_val_keys.sort(
                                                key=lambda x: (
                                                    int(re.search(r"\d+", x).group())
                                                    if re.search(r"\d+", x)
                                                    else 0
                                                )
                                            )
                                            r["values"] = [r[rvk] for rvk in r_val_keys]
                            norm_rows.append({k: row.get(k) for k in expected_keys})
                        else:
                            norm_rows.append(row)
                    return True, norm_rows
        return True, val

    if not isinstance(val, str):
        return True, val

    # If expected type is list/array or items_schema is provided, try parsing JSON from string
    is_table = (expected_type or "").lower() in ("list", "array") or (items_schema is not None)
    if is_table:
        parsed_ok, parsed_json = _try_parse_json_value(val)
        if parsed_ok:
            if isinstance(parsed_json, dict) and "value" in parsed_json:
                parsed_json = parsed_json["value"]
            if isinstance(parsed_json, list):
                if len(parsed_json) == 0:
                    return False, None
                if items_schema and isinstance(items_schema, dict):
                    expected_keys = items_schema.get("properties", {}).keys()
                    if expected_keys:
                        return True, [
                            {k: row.get(k) for k in expected_keys} if isinstance(row, dict) else row
                            for row in parsed_json
                        ]
                return True, parsed_json
            return True, parsed_json

    if _is_not_found_value(val, expected_type=expected_type):
        return False, None

    if query_key and _is_label_hallucination(val, query_key, query_desc):
        return False, None

    if _is_copied_from_prompt_example(val, query_desc, context):
        return False, None

    val = _strip_field_prefix(val, query_key, query_desc)

    if expected_type and expected_type.lower() == "boolean":
        lower = val.strip().lower().strip("\"'., ")
        if lower in {"yes", "evet", "true", "var", "mevcut", "1"}:
            return True, True
        if lower in _BOOLEAN_VALID_NEGATIVES or lower in {"no", "hayır", "hayir", "0"}:
            return True, False
        # If the model extracted an entity/ID (e.g. passport number 'U30154391')
        # for a boolean query, it confirms presence of the requested entity
        return True, True

    return True, val


def _process_extracted_answer(
    raw_answer: str,
    expected_type: str | None = None,
    items_schema: dict[str, Any] | None = None,
    query_key: str | None = None,
    query_desc: str | None = None,
    context: str | None = None,
) -> tuple[bool, Any]:
    """Parse raw LLM response (JSON object/array or scalar) and normalize typed value."""
    is_table = (expected_type or "").lower() in ("list", "array") or (items_schema is not None)
    if is_table:
        parsed_ok, parsed_json = _try_parse_json_value(raw_answer)
        if parsed_ok:
            parsed_json = _clean_json_placeholders(parsed_json)
            if isinstance(parsed_json, dict) and "value" in parsed_json:
                parsed_json = parsed_json["value"]
            return _normalize_typed_value(
                parsed_json,
                expected_type,
                items_schema,
                query_key=query_key,
                query_desc=query_desc,
                context=context,
            )
        cleaned_val = _sanitize_extracted_value(raw_answer)
        return _normalize_typed_value(
            cleaned_val,
            expected_type,
            items_schema,
            query_key=query_key,
            query_desc=query_desc,
            context=context,
        )

    parsed_ok, parsed_json = _try_parse_json_value(raw_answer)
    if parsed_ok and isinstance(parsed_json, dict) and "value" in parsed_json:
        parsed_json = _clean_json_placeholders(parsed_json)
        is_found = bool(parsed_json.get("found", True))
        raw_extracted = parsed_json.get("value")
        if raw_extracted is None or not is_found:
            return False, None
        cleaned_val = _sanitize_extracted_value(str(raw_extracted))
        return _normalize_typed_value(
            cleaned_val,
            expected_type,
            items_schema,
            query_key=query_key,
            query_desc=query_desc,
            context=context,
        )

    cleaned_val = _sanitize_extracted_value(raw_answer)
    return _normalize_typed_value(
        cleaned_val,
        expected_type,
        items_schema,
        query_key=query_key,
        query_desc=query_desc,
        context=context,
    )


def _process_schema_extraction_output(
    raw_content: str,
    fields: list[dict[str, Any]],
    context: str,
    reasoning_content: str | None = None,
) -> dict[str, LLMExtractionResult]:
    """Parse, salvage, and normalize raw LLM output into field extraction results."""

    if reasoning_content:
        raw_answer = raw_content
    else:
        _, raw_answer = _parse_model_response(
            raw_content,
            started_in_thought=False,
        )

    target_text = raw_answer.strip() if raw_answer.strip() else raw_content.strip()

    parsed_ok, parsed_json = _try_parse_schema_json(target_text)

    if not parsed_ok or not isinstance(parsed_json, (dict, list)) or not parsed_json:
        rescued_fields = _rescue_schema_fields_from_text(target_text, fields)
        if not rescued_fields and raw_content != target_text:
            rescued_fields = _rescue_schema_fields_from_text(raw_content, fields)
        if rescued_fields:
            logger.info(
                "--> [LLM Unified Schema] Salvaged %d/%d fields via regex rescue fallback",
                len(rescued_fields),
                len(fields),
            )
            parsed_json = rescued_fields
        else:
            logger.warning(
                "Could not parse or salvage valid JSON schema from response (len=%d): %s",
                len(target_text),
                target_text[:200],
            )
            parsed_json = {}

    if isinstance(parsed_json, list) and len(fields) == 1:
        parsed_json = {fields[0]["key"]: parsed_json}

    if isinstance(parsed_json, dict) and len(parsed_json) == 1:
        first_val = next(iter(parsed_json.values()))
        if isinstance(first_val, dict) and any(f["key"] in first_val for f in fields):
            parsed_json = first_val

    parsed_json = _clean_json_placeholders(parsed_json)
    results: dict[str, LLMExtractionResult] = {}

    for f in fields:
        key = f["key"]
        expected_type = f.get("type")
        items_schema = f.get("items_schema")
        query_desc = f.get("description") or key

        raw_val = parsed_json.get(key) if isinstance(parsed_json, dict) else None
        if (
            raw_val is None
            and len(fields) == 1
            and isinstance(parsed_json, dict)
            and len(parsed_json) == 1
        ):
            raw_val = next(iter(parsed_json.values()))

        if isinstance(raw_val, dict) and (expected_type or "").lower() not in (
            "object",
            "dict",
        ):
            if "value" in raw_val:
                raw_val = raw_val["value"]
            elif "description" in raw_val:
                raw_val = raw_val["description"]

        if raw_val is None:
            results[key] = {"found": False, "value": None, "thinking": None}
            continue

        is_found, normalized_val = _normalize_typed_value(
            raw_val,
            expected_type=expected_type,
            items_schema=items_schema,
            query_key=key,
            query_desc=query_desc,
            context=context,
        )
        results[key] = {
            "found": is_found and normalized_val is not None,
            "value": normalized_val if is_found else None,
            "thinking": None,
        }

    return results
