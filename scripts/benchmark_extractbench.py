"""Benchmark runner for DocNeedle on LlamaIndex's ExtractBench dataset.

Evaluates structured information extraction performance (Unified Value F1,
Page Grounding F1, and Strict Nullability / Hallucination Rejection) across
enterprise documents using the DocNeedle REST API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("extractbench_benchmark")

console = Console()

REPO_ID = "llamaindex/ExtractBench"
VALID_SPLITS = ("short", "medium", "long", "all")
DATE_FORMATS = (
    "%Y-%m-%d",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%m.%d.%Y",
    "%Y.%m.%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%Y %b %d",
    "%Y %B %d",
    "%Y%b%d",
    "%Y%B%d",
    "%d%b%Y",
    "%d%B%Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%Y-%b-%d",
    "%Y-%B-%d",
)
CURRENCY_SYMBOLS_PATTERN = re.compile(r"[\$,€£¥₹%\s]|(?:TL|TRY|USD|EUR|GBP|CAD|AUD)\b", re.I)
EMPTY_PLACEHOLDERS = frozenset(
    {"", "none", "null", "n/a", "na", "-", "--", "—", "()", "( )", "[]", "[ ]", "{}"}
)


# ---------------------------------------------------------------------------
# Comparator & Normalization Utilities (Fair & Strict to Benchmark Rules)
# ---------------------------------------------------------------------------


def is_falsy_or_empty(val: Any) -> bool:
    """Check if value represents a null, absent, or empty placeholder."""
    if val is None or val is False:
        return True
    if isinstance(val, str) and val.strip().lower() in EMPTY_PLACEHOLDERS:
        return True
    return False


def normalize_text(text: Any) -> str:
    """Whitespace-normalized and case-folded string representation."""
    if text is None:
        return ""
    return " ".join(str(text).strip().split()).casefold()


def parse_boolean(val: Any) -> bool | None:
    """Resolve boolean values from raw primitives or textual representations."""
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in {"true", "yes", "1", "t", "evet"}:
        return True
    if s in {"false", "no", "0", "f", "hayır", "hayir"} or s in EMPTY_PLACEHOLDERS:
        return False
    return None


def parse_number(val: Any) -> float | None:
    """Parse numeric values from strings, removing commas, spaces, currency, and % markers."""
    if is_falsy_or_empty(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    is_neg = False
    if s.startswith("(") and s.endswith(")"):
        is_neg = True
        s = s[1:-1]
    cleaned = CURRENCY_SYMBOLS_PATTERN.sub("", s).replace(",", "").strip(".").strip()
    try:
        num = float(cleaned)
        return -num if is_neg else num
    except ValueError:
        return None


def parse_date(val: Any) -> tuple[int, int, int] | tuple[int, set[int]] | None:
    """Parse calendar dates into structured tuples for timezone/format invariant matching."""
    if not val:
        return None
    s = str(val).strip()
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt).date()
            return (dt.year, dt.month, dt.day)
        except ValueError:
            pass
    # Fallback heuristic: 3 digit sequences containing a 4-digit year
    digits = re.findall(r"\d+", s)
    if len(digits) == 3:
        year_str = next((d for d in digits if len(d) == 4), None)
        if year_str:
            others = {int(d) for d in digits if d != year_str}
            return (int(year_str), others)
    return None


def compare_values(
    expected: Any,
    predicted: Any,
    comparator: str | None = None,
    evidence_values: list[Any] | None = None,
) -> bool:
    """Compare ground truth and predicted values according to comparator rules.

    Also supports ExtractBench's OR-acceptable values specified in evidence rules.
    """
    # 1. Null / False / Empty equivalence
    if is_falsy_or_empty(expected) and is_falsy_or_empty(predicted):
        return True

    # 2. Check candidate values (expected + any alternative evidence value)
    candidates = [expected]
    if evidence_values:
        for ev_val in evidence_values:
            if ev_val is not None and ev_val not in candidates:
                candidates.append(ev_val)

    for cand in candidates:
        if _compare_single_candidate(cand, predicted, comparator):
            return True

    return False


def _compare_single_candidate(expected: Any, predicted: Any, comparator: str | None) -> bool:
    if expected is None and (predicted is None or predicted == ""):
        return True

    comp = str(comparator or "case_insensitive").lower()
    if expected is None or predicted is None:
        # Solution 3: ExtractBench null_equals_false rule for boolean checkboxes
        if comp == "boolean" and expected is None:
            pred_b = parse_boolean(predicted)
            if pred_b is False:
                return True
        return False

    comp = str(comparator or "case_insensitive").lower()

    # Boolean comparator
    if comp == "boolean" or isinstance(expected, bool):
        exp_b = parse_boolean(expected)
        pred_b = parse_boolean(predicted)
        if exp_b is not None and pred_b is not None:
            return exp_b == pred_b
        return False

    # Numeric or Currency comparator
    if comp in {"number", "currency"} or isinstance(expected, (int, float)):
        exp_n = parse_number(expected)
        pred_n = parse_number(predicted)
        if exp_n is not None and pred_n is not None:
            if math.isclose(exp_n, pred_n, rel_tol=1e-3, abs_tol=1e-4):
                return True
            # Check integer equality
            if int(round(exp_n)) == int(round(pred_n)):
                return True
        return False

    # Date comparator
    if comp == "date":
        exp_d = parse_date(expected)
        pred_d = parse_date(predicted)
        if exp_d and pred_d:
            if exp_d == pred_d:
                return True
            if len(exp_d) == 3 and len(pred_d) == 2 and isinstance(pred_d[1], set):
                return exp_d[0] == pred_d[0] and set(exp_d[1:]) == pred_d[1]
            if len(pred_d) == 3 and len(exp_d) == 2 and isinstance(exp_d[1], set):
                return exp_d[0] == pred_d[0] and set(exp_d[1:]) == pred_d[1]
            if (
                len(exp_d) == 2
                and len(pred_d) == 2
                and isinstance(exp_d[1], set)
                and isinstance(pred_d[1], set)
            ):
                return exp_d[0] == pred_d[0] and exp_d[1] == pred_d[1]
        # Fallback to normalized text
        return normalize_text(expected) == normalize_text(predicted)

    # List / Array structures
    if isinstance(expected, list):
        pred_list = predicted
        if isinstance(predicted, str):
            try:
                pred_list = json.loads(predicted)
            except Exception:
                pred_list = [p.strip() for p in re.split(r"[,;\n]+", predicted) if p.strip()]
        if isinstance(pred_list, list):
            if len(expected) == 0 and len(pred_list) == 0:
                return True
            if len(expected) == len(pred_list):
                # Order-insensitive element comparison
                matched_indices: set[int] = set()
                for exp_item in expected:
                    found_match = False
                    for idx, pred_item in enumerate(pred_list):
                        if idx not in matched_indices and _compare_single_candidate(
                            exp_item, pred_item, "case_insensitive"
                        ):
                            matched_indices.add(idx)
                            found_match = True
                            break
                    if not found_match:
                        return False
                return True
        return False

    # Dict / Object structures
    if isinstance(expected, dict):
        pred_dict = predicted
        if isinstance(predicted, str):
            try:
                pred_dict = json.loads(predicted)
            except Exception:
                pred_dict = None
        if isinstance(pred_dict, dict):
            if set(expected.keys()) != set(pred_dict.keys()):
                return False
            return all(
                _compare_single_candidate(expected[k], pred_dict[k], "case_insensitive")
                for k in expected
            )
        return False

    # Standard case-insensitive text match
    norm_exp = normalize_text(expected)
    norm_pred = normalize_text(predicted)
    if norm_exp == norm_pred:
        return True

    # Punctuation-insensitive match (ExtractBench normalizer for addresses, names, and codes)
    clean_exp = " ".join(re.sub(r"[,\.\-\/\\#;:\(\)\[\]]+", " ", norm_exp).split())
    clean_pred = " ".join(re.sub(r"[,\.\-\/\\#;:\(\)\[\]]+", " ", norm_pred).split())
    if clean_exp and clean_exp == clean_pred:
        return True

    return False


# ---------------------------------------------------------------------------
# Schema Translation from JSON Schema to DocNeedle ExtractionItemQuery
# ---------------------------------------------------------------------------


def dereference_json_schema(sub_schema: Any, root_schema: dict[str, Any], depth: int = 0) -> Any:
    """Recursively resolve all $ref pointers throughout a JSON Schema structure."""
    if depth > 20:
        return sub_schema

    if isinstance(sub_schema, dict):
        if "$ref" in sub_schema and isinstance(sub_schema["$ref"], str):
            ref_str = sub_schema["$ref"]
            if ref_str.startswith("#/"):
                parts = ref_str[2:].split("/")
                cur: Any = root_schema
                for p in parts:
                    if isinstance(cur, dict) and p in cur:
                        cur = cur[p]
                    else:
                        cur = None
                        break
                if isinstance(cur, dict):
                    dereferenced = {k: v for k, v in sub_schema.items() if k != "$ref"}
                    resolved_target = dereference_json_schema(cur, root_schema, depth + 1)
                    if isinstance(resolved_target, dict):
                        return {**resolved_target, **dereferenced}
                    return resolved_target

        return {
            k: dereference_json_schema(v, root_schema, depth + 1) for k, v in sub_schema.items()
        }

    if isinstance(sub_schema, list):
        return [dereference_json_schema(item, root_schema, depth + 1) for item in sub_schema]

    return sub_schema


def _resolve_items_schema(
    prop_meta: dict[str, Any], schema_dict: dict[str, Any]
) -> dict[str, Any] | None:
    """Extract and recursively dereference the items schema for array properties."""
    raw_items = prop_meta.get("items")
    if not raw_items and "anyOf" in prop_meta:
        for opt in prop_meta["anyOf"]:
            if isinstance(opt, dict) and opt.get("type") == "array" and "items" in opt:
                raw_items = opt["items"]
                break

    if not isinstance(raw_items, dict):
        return None

    resolved = dereference_json_schema(raw_items, schema_dict)
    return resolved if isinstance(resolved, dict) else None


def translate_schema_properties(
    schema_dict: dict[str, Any],
    field_rules: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert JSON Schema properties into DocNeedle ExtractionItemQuery dictionaries."""
    full_schema = dereference_json_schema(schema_dict, schema_dict)
    items: list[dict[str, Any]] = []
    properties = full_schema.get("properties", {})

    for prop_name, prop_meta in properties.items():
        if not isinstance(prop_meta, dict):
            continue

        desc = prop_meta.get("description") or prop_meta.get("title") or prop_name
        enum_vals = prop_meta.get("enum")
        if not enum_vals and "anyOf" in prop_meta:
            for opt in prop_meta["anyOf"]:
                if isinstance(opt, dict) and "enum" in opt:
                    enum_vals = opt["enum"]
                    break
        if enum_vals and isinstance(enum_vals, list):
            valid_enums = [str(v) for v in enum_vals if v is not None]
            if valid_enums:
                desc = f"{desc} (Allowed options: {', '.join(valid_enums)})"

        rule = field_rules.get(prop_name, {})
        raw_comparator = rule.get("comparator") if isinstance(rule, dict) else None
        comp_str = raw_comparator.lower() if isinstance(raw_comparator, str) else ""

        # Determine DocNeedle expected type
        field_type = "string"
        raw_type = prop_meta.get("type")
        if isinstance(raw_type, list):
            type_str = next((t for t in raw_type if t and t != "null"), "string")
        elif not raw_type and "anyOf" in prop_meta:
            types = [t.get("type") for t in prop_meta["anyOf"] if isinstance(t, dict)]
            flat_types: list[str] = []
            for t in types:
                if isinstance(t, list):
                    flat_types.extend(str(item) for item in t)
                elif isinstance(t, str):
                    flat_types.append(t)
            type_str = next((t for t in flat_types if t and t != "null"), "string")
        elif isinstance(raw_type, str):
            type_str = raw_type
        else:
            type_str = "string"

        type_str = type_str.lower()

        if comp_str == "boolean" or type_str == "boolean":
            field_type = "boolean"
        elif comp_str in {"number", "currency"} or type_str in {"number", "integer"}:
            field_type = "money" if comp_str == "currency" else "number"
        elif comp_str == "date":
            field_type = "date"
        elif type_str == "array" or isinstance(raw_comparator, dict):
            field_type = "list"
        elif comp_str == "address":
            field_type = "address"

        item_dict: dict[str, Any] = {
            "key": prop_name,
            "description": desc,
            "type": field_type,
        }
        if field_type == "list":
            items_schema = _resolve_items_schema(prop_meta, schema_dict)
            if isinstance(items_schema, dict):
                item_dict["items_schema"] = items_schema

        items.append(item_dict)

    return items


# ---------------------------------------------------------------------------
# Dataset Loader for ExtractBench
# ---------------------------------------------------------------------------


class ExtractBenchLoader:
    """Manages loading and local caching of ExtractBench JSONL records and PDF files."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def load_split_records(self, split: str) -> list[dict[str, Any]]:
        """Download and yield dataset lines from the requested split."""
        splits_to_load = ["short", "medium", "long"] if split == "all" else [split]
        all_records: list[dict[str, Any]] = []

        for s in splits_to_load:
            filename = f"{s}.jsonl"
            logger.info(f"Downloading/Locating {filename} from {REPO_ID}...")
            local_jsonl = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                token=self.token,
            )
            with open(local_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        all_records.append(row)

        return all_records

    def get_document_pdf_path(self, pdf_rel_path: str) -> Path:
        """Download (if not cached) and return local filesystem Path to PDF."""
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=pdf_rel_path,
            repo_type="dataset",
            token=self.token,
        )
        return Path(local_path)


# ---------------------------------------------------------------------------
# DocNeedle API Client with Transparent Schema Batching
# ---------------------------------------------------------------------------


class DocNeedleBenchmarkClient:
    """Async client interacting with DocNeedle /v1/extract with automatic chunking."""

    def __init__(
        self,
        api_url: str = "http://localhost:7860",
        batch_size: int = 35,
        enable_reranker: bool = False,
        timeout: float | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.batch_size = max(1, min(batch_size, 48))  # strictly within max_schema_items (50)
        self.enable_reranker = enable_reranker
        self.timeout = timeout

    async def extract_all_fields(
        self,
        pdf_path: Path,
        schema_items: list[dict[str, Any]],
        max_retries: int = 3,
    ) -> tuple[dict[str, dict[str, Any]], float]:
        """Extract all requested schema items from the document with transparent batching.

        Returns (results_by_key, total_latency_ms).
        """
        # Read PDF binary once into memory
        pdf_bytes = pdf_path.read_bytes()
        filename = pdf_path.name

        # Partition items into batches
        batches: list[list[dict[str, Any]]] = [
            schema_items[i : i + self.batch_size]
            for i in range(0, len(schema_items), self.batch_size)
        ]

        merged_results: dict[str, dict[str, Any]] = {}
        total_latency_ms = 0.0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for batch_idx, batch in enumerate(batches, 1):
                schema_json = json.dumps(batch)
                success = False

                for attempt in range(1, max_retries + 1):
                    try:
                        files = [("files", (filename, pdf_bytes, "application/pdf"))]
                        data: dict[str, Any] = {"schema": schema_json}
                        if self.enable_reranker:
                            data["enable_reranker"] = "true"

                        response = await client.post(
                            f"{self.api_url}/v1/extract",
                            files=files,
                            data=data,
                        )

                        if response.status_code == 200:
                            res_data = response.json()
                            total_latency_ms += res_data.get("processing_time_ms", 0.0)
                            for item in res_data.get("results", []):
                                merged_results[item["key"]] = item
                            success = True
                            break

                        logger.warning(
                            f"API returned {response.status_code} on attempt {attempt}: "
                            f"{response.text[:200]}"
                        )
                    except (httpx.RequestError, httpx.TimeoutException) as err:
                        err_msg = str(err).strip() or "Request timed out"
                        err_type = type(err).__name__
                        logger.warning(
                            f"Connection error on attempt {attempt}: {err_type} ({err_msg})"
                        )

                    if attempt < max_retries:
                        await asyncio.sleep(2.0 * attempt)

                if not success:
                    raise RuntimeError(
                        f"Failed batch {batch_idx}/{len(batches)} for {filename} after "
                        f"{max_retries} attempts."
                    )

        return merged_results, total_latency_ms


# ---------------------------------------------------------------------------
# Evaluation Scorer (Exact Bench Rules)
# ---------------------------------------------------------------------------


@dataclass
class FieldEvaluation:
    key: str
    expected: Any
    predicted: Any
    matched: bool
    expected_found: bool
    predicted_found: bool
    comparator: str
    expected_pages: list[int]
    predicted_page: int | None
    page_matched: bool


@dataclass
class DocumentEvaluation:
    doc_id: str
    category: str
    total_fields: int
    matched_fields: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    value_f1: float
    page_accuracy: float
    latency_ms: float
    is_perfect: bool
    fields: list[FieldEvaluation]


def evaluate_document_case(
    doc_id: str,
    category: str,
    expected_output: dict[str, Any],
    field_rules: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    latency_ms: float,
) -> DocumentEvaluation:
    """Evaluate a single document against ground truth according to ExtractBench specification."""
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    page_matches = 0
    grounded_eval_count = 0
    field_evals: list[FieldEvaluation] = []

    # Iterate through all fields specified in the benchmark
    for key, expected_val in expected_output.items():
        rule = field_rules.get(key, {})
        comparator = (
            rule.get("comparator", "case_insensitive")
            if isinstance(rule, dict)
            else "case_insensitive"
        )
        evidence_list = rule.get("evidence", []) if isinstance(rule, dict) else []

        # Extract alternative values & pages from evidence
        evidence_vals = [
            e.get("value") for e in evidence_list if isinstance(e, dict) and "value" in e
        ]
        expected_pages = [
            e.get("page")
            for e in evidence_list
            if isinstance(e, dict) and e.get("page") is not None
        ]

        pred_item = predictions.get(key, {})
        pred_found = pred_item.get("found", False)
        pred_val = pred_item.get("value") if pred_found else None
        pred_page = pred_item.get("evidence", {}).get("page") if pred_item.get("evidence") else None

        # Does ground truth expect a non-null, non-false value?
        exp_found = not is_falsy_or_empty(expected_val)
        pred_has_val = not is_falsy_or_empty(pred_val)

        # Solution 3: Checkbox & Boolean Normalization (ExtractBench null_equals_false rule)
        # If ground truth expects None/False, and model extracted 'No'/'False'/0/nil,
        # it correctly recognized unselected checkbox (True Negative match, not hallucination)
        is_bool_field = (
            comparator == "boolean"
            or isinstance(expected_val, bool)
            or (isinstance(rule, dict) and "null_equals_false" in rule.get("normalizers", []))
            or key.startswith(("is_", "has_"))
            or key.endswith(("_yes", "_no"))
        )
        if is_bool_field and not exp_found and pred_has_val:
            pred_str = str(pred_val).strip().lower()
            if pred_str in {"no", "false", "0", "hayır", "hayir", "nil", "none", "n/a", "-"}:
                pred_has_val = False

        # Evaluate Value Match
        matched = compare_values(
            expected=expected_val,
            predicted=pred_val,
            comparator=comparator if isinstance(comparator, str) else None,
            evidence_values=evidence_vals,
        )
        if is_bool_field and not exp_found and not pred_has_val:
            matched = True

        page_matched = False
        if exp_found and pred_has_val:
            if matched:
                tp += 1
                # Evaluate Grounding
                if expected_pages:
                    grounded_eval_count += 1
                    if pred_page in expected_pages:
                        page_matches += 1
                        page_matched = True
            else:
                fp += 1
                fn += 1
        elif not exp_found and not pred_has_val:
            tn += 1  # Correctly recognized absent item (Strict Nullability)
        elif not exp_found and pred_has_val:
            fp += 1  # Hallucinated non-existent value
        elif exp_found and not pred_has_val:
            fn += 1  # Missed value present in document

        field_evals.append(
            FieldEvaluation(
                key=key,
                expected=expected_val,
                predicted=pred_val,
                matched=matched,
                expected_found=exp_found,
                predicted_found=pred_found,
                comparator=str(comparator),
                expected_pages=expected_pages,
                predicted_page=pred_page,
                page_matched=page_matched,
            )
        )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    value_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    page_acc = page_matches / grounded_eval_count if grounded_eval_count > 0 else 1.0
    is_perfect = fp == 0 and fn == 0

    return DocumentEvaluation(
        doc_id=doc_id,
        category=category,
        total_fields=len(expected_output),
        matched_fields=sum(1 for f in field_evals if f.matched),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        value_f1=round(value_f1, 4),
        page_accuracy=round(page_acc, 4),
        latency_ms=round(latency_ms, 2),
        is_perfect=is_perfect,
        fields=field_evals,
    )


def _render_and_save_summary(
    evaluations: list[DocumentEvaluation],
    total_duration: float,
    summary_file: Path,
    output_dir: Path,
    split: str,
) -> None:
    """Compute aggregate benchmark metrics, write summary.json, and print table."""
    if not evaluations:
        console.print("[yellow]No evaluations were completed.[/yellow]")
        return

    total_tp = sum(e.true_positives for e in evaluations)
    total_fp = sum(e.false_positives for e in evaluations)
    total_fn = sum(e.false_negatives for e in evaluations)
    total_tn = sum(e.true_negatives for e in evaluations)
    total_fields = sum(e.total_fields for e in evaluations)
    total_matched = sum(e.matched_fields for e in evaluations)
    perfect_docs = sum(1 for e in evaluations if e.is_perfect)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_denom = precision + recall
    overall_f1 = (2 * precision * recall) / f1_denom if f1_denom > 0 else 0.0
    perfect_rate = perfect_docs / len(evaluations) if evaluations else 0.0
    avg_page_acc = (
        sum(e.page_accuracy for e in evaluations) / len(evaluations) if evaluations else 0.0
    )
    strict_null_acc = (
        round(total_tn / (total_tn + total_fp), 4) if (total_tn + total_fp) > 0 else 1.0
    )

    summary_data = {
        "benchmark": "ExtractBench",
        "split": split,
        "timestamp": datetime.now(UTC).isoformat(),
        "total_documents": len(evaluations),
        "total_fields": total_fields,
        "matched_fields": total_matched,
        "unified_value_f1": round(overall_f1, 4),
        "value_precision": round(precision, 4),
        "value_recall": round(recall, 4),
        "perfect_documents": perfect_docs,
        "perfect_document_rate": round(perfect_rate, 4),
        "strict_null_accuracy": strict_null_acc,
        "page_grounding_accuracy": round(avg_page_acc, 4),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    results_table = Table(title="ExtractBench Evaluation Summary", border_style="cyan")
    results_table.add_column("Metric", style="bold white")
    results_table.add_column("Score / Count", style="bold green", justify="right")

    results_table.add_row("Evaluated Documents", str(len(evaluations)))
    results_table.add_row("Total Fields Scored", str(total_fields))
    results_table.add_row("Unified Value F1", f"{overall_f1:.2%}")
    results_table.add_row("Value Precision", f"{precision:.2%}")
    results_table.add_row("Value Recall", f"{recall:.2%}")
    results_table.add_row(
        "Perfect Documents (% Perfect)",
        f"{perfect_rate:.2%} ({perfect_docs}/{len(evaluations)})",
    )
    results_table.add_row("Strict Nullability (Anti-Hallucination)", f"{strict_null_acc:.2%}")
    results_table.add_row("Page Grounding Accuracy", f"{avg_page_acc:.2%}")

    console.print("\n")
    console.print(results_table)
    console.print(
        f"\n[bold green]✓ Detailed outputs saved to:[/bold green] "
        f"[underline]{output_dir}[/underline]\n"
    )


# ---------------------------------------------------------------------------
# Benchmark Runner Orchestrator
# ---------------------------------------------------------------------------


async def run_benchmark(
    api_url: str,
    split: str,
    batch_size: int,
    output_dir: Path,
    limit: int | None = None,
    offset: int = 0,
    target_doc_id: str | None = None,
    enable_reranker: bool = False,
    hf_token: str | None = None,
    dry_run: bool = False,
    resume: bool = False,
    rescore: bool = False,
    timeout: float | None = None,
) -> None:
    """Execute the end-to-end ExtractBench benchmark suite against DocNeedle API."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / f"summary_{split}.json"
    detailed_file = output_dir / f"detailed_{split}.jsonl"

    if rescore:
        if not detailed_file.exists():
            console.print(
                f"[bold red]Hata:[/bold red] {detailed_file} bulunamadı, rescore yapılamaz."
            )
            return
        console.print(
            Panel.fit(
                f"[bold cyan]DocNeedle Benchmark Re-Scorer[/bold cyan]\n"
                f"[dim]Re-evaluating:[/dim] {detailed_file} with updated normalizers",
                title="ExtractBench Re-Score",
                border_style="cyan",
            )
        )
        loader = ExtractBenchLoader(token=hf_token)
        records_map = {r["id"]: r for r in loader.load_split_records(split)}
        updated_evals = []
        tmp_detailed = output_dir / f"detailed_{split}.tmp.jsonl"
        with (
            open(detailed_file, "r", encoding="utf-8") as rf,
            open(tmp_detailed, "w", encoding="utf-8") as wf,
        ):
            for line in rf:
                if not line.strip():
                    continue
                sdoc = json.loads(line)
                doc_id = sdoc["doc_id"]
                rec = records_map.get(doc_id)
                if not rec:
                    continue
                exp = json.loads(rec["expected_output"])
                rules = json.loads(rec["field_rules"])
                preds = {
                    f["key"]: {
                        "found": f["predicted_found"],
                        "value": f["predicted"],
                        "evidence": {"page": 1 if f.get("page_matched") else None},
                    }
                    for f in sdoc["fields"]
                }
                ev = evaluate_document_case(
                    doc_id=doc_id,
                    category=sdoc.get("category", split),
                    expected_output=exp,
                    field_rules=rules,
                    predictions=preds,
                    latency_ms=sdoc.get("latency_ms", 0.0),
                )
                updated_evals.append(ev)
                doc_dict = {
                    "doc_id": ev.doc_id,
                    "category": ev.category,
                    "total_fields": ev.total_fields,
                    "matched_fields": ev.matched_fields,
                    "true_positives": ev.true_positives,
                    "false_positives": ev.false_positives,
                    "false_negatives": ev.false_negatives,
                    "true_negatives": ev.true_negatives,
                    "value_f1": ev.value_f1,
                    "page_accuracy": ev.page_accuracy,
                    "latency_ms": ev.latency_ms,
                    "is_perfect": ev.is_perfect,
                    "fields": [
                        {
                            "key": f.key,
                            "matched": f.matched,
                            "expected": f.expected,
                            "predicted": f.predicted,
                            "expected_found": f.expected_found,
                            "predicted_found": f.predicted_found,
                            "page_matched": f.page_matched,
                        }
                        for f in ev.fields
                    ],
                }
                wf.write(json.dumps(doc_dict) + "\n")
        tmp_detailed.replace(detailed_file)
        evaluations = updated_evals
        total_duration = sum(e.latency_ms for e in evaluations) / 1000.0
        # Calculate summary and return
        _render_and_save_summary(evaluations, total_duration, summary_file, output_dir, split)
        return

    console.print(
        Panel.fit(
            f"[bold cyan]DocNeedle Benchmark Runner[/bold cyan]\n"
            f"[dim]Dataset:[/dim] {REPO_ID} | [dim]Split:[/dim] [yellow]{split}[/yellow]\n"
            f"[dim]Target API:[/dim] [green]{api_url}[/green] | "
            f"[dim]Batch Size:[/dim] {batch_size}\n"
            f"[dim]Output Directory:[/dim] {output_dir}",
            title="ExtractBench Evaluation",
            border_style="cyan",
        )
    )

    loader = ExtractBenchLoader(token=hf_token)
    client = DocNeedleBenchmarkClient(
        api_url=api_url,
        batch_size=batch_size,
        enable_reranker=enable_reranker,
        timeout=timeout,
    )

    # 1. Load dataset records
    with console.status(
        f"[bold yellow]Loading records for split '{split}' from Hugging Face...[/bold yellow]"
    ):
        records = loader.load_split_records(split)

    if target_doc_id:
        records = [r for r in records if r.get("id") == target_doc_id]
        if not records:
            console.print(f"[bold red]Error:[/bold red] Document ID '{target_doc_id}' not found.")
            return

    # Apply offset and limit
    if offset > 0:
        records = records[offset:]
    if limit is not None and limit > 0:
        records = records[:limit]

    total_docs = len(records)
    console.print(
        f"[bold green]✓[/bold green] Prepared [bold]{total_docs}[/bold] documents for evaluation.\n"
    )

    # Pre-flight check: verify API server availability before running
    if not dry_run:
        try:
            async with httpx.AsyncClient(timeout=4.0) as check_client:
                health_resp = await check_client.get(f"{api_url}/healthz")
                if health_resp.status_code != 200:
                    console.print(
                        f"[bold red]Hata:[/bold red] API sağlık kontrolü başarısız "
                        f"({health_resp.status_code}): {api_url}/healthz"
                    )
                    return
        except Exception:
            console.print(
                f"\n[bold red]Hata: DocNeedle API ({api_url}) bağlanılamadı![/bold red]\n"
                f"[yellow]Lütfen önce ayrı bir terminalde sunucuyu başlatın:[/yellow]\n"
                f"  [bold cyan]uv run python app.py[/bold cyan]\n"
                f"[dim]veya sadece API sunucusu için:[/dim]\n"
                f"  [bold cyan]uv run uvicorn docneedle.main:app --host 0.0.0.0 --port 7860"
                f"[/bold cyan]\n"
            )
            return

    evaluations: list[DocumentEvaluation] = []
    already_evaluated_ids: set[str] = set()

    if resume and detailed_file.exists():
        with open(detailed_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    already_evaluated_ids.add(d["doc_id"])
                    evaluations.append(
                        DocumentEvaluation(
                            doc_id=d["doc_id"],
                            category=d.get("category", split),
                            total_fields=d["total_fields"],
                            matched_fields=d["matched_fields"],
                            true_positives=d.get("true_positives", d["matched_fields"]),
                            false_positives=d.get("false_positives", 0),
                            false_negatives=d.get("false_negatives", 0),
                            true_negatives=d.get("true_negatives", 0),
                            value_f1=d["value_f1"],
                            page_accuracy=d.get("page_accuracy", 1.0),
                            latency_ms=d.get("latency_ms", 0.0),
                            is_perfect=d.get("is_perfect", False),
                            fields=[],
                        )
                    )
                except Exception:
                    pass
        if already_evaluated_ids:
            console.print(
                f"[bold cyan]ℹ Resume etkin:[/bold cyan] "
                f"{len(already_evaluated_ids)} belge önceden tamamlanmış, atlanıyor.\n"
            )
            records = [r for r in records if r.get("id") not in already_evaluated_ids]

    resumed_count = len(already_evaluated_ids)
    remaining_count = len(records)
    start_time = time.perf_counter()

    # Setup progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task_id = progress.add_task(
            "[cyan]Evaluating documents...",
            total=total_docs,
            completed=resumed_count,
        )

        for idx, row in enumerate(records, 1):
            doc_id = row["id"]
            pdf_rel_path = row["pdf"]
            schema_dict = json.loads(row["data_schema"])
            expected_output = json.loads(row["expected_output"])
            field_rules = json.loads(row["field_rules"])

            curr_global = resumed_count + idx
            num_fields = len(schema_dict.get("properties", {}))
            progress.update(
                task_id,
                description=(
                    f"[cyan]Doc {curr_global}/{total_docs}: {doc_id} "
                    f"[dim]({idx}/{remaining_count} kalan, {num_fields} alan)[/dim]"
                ),
            )

            try:
                # 1. Ensure PDF is available locally
                pdf_path = loader.get_document_pdf_path(pdf_rel_path)

                # 2. Translate JSON Schema to DocNeedle ExtractionItemQuery list
                schema_items = translate_schema_properties(schema_dict, field_rules)

                if dry_run:
                    # Dry run mode: check parsing and skip API call
                    predictions: dict[str, dict[str, Any]] = {}
                    latency_ms = 0.0
                else:
                    # 3. Call DocNeedle API (with transparent batching)
                    predictions, latency_ms = await client.extract_all_fields(
                        pdf_path, schema_items
                    )

                # 4. Evaluate predictions
                eval_res = evaluate_document_case(
                    doc_id=doc_id,
                    category=row.get("category", split),
                    expected_output=expected_output,
                    field_rules=field_rules,
                    predictions=predictions,
                    latency_ms=latency_ms,
                )
                evaluations.append(eval_res)

                # Write line to detailed log immediately
                with open(detailed_file, "a", encoding="utf-8") as f:
                    doc_dict = {
                        "doc_id": eval_res.doc_id,
                        "category": eval_res.category,
                        "total_fields": eval_res.total_fields,
                        "matched_fields": eval_res.matched_fields,
                        "true_positives": eval_res.true_positives,
                        "false_positives": eval_res.false_positives,
                        "false_negatives": eval_res.false_negatives,
                        "true_negatives": eval_res.true_negatives,
                        "value_f1": eval_res.value_f1,
                        "page_accuracy": eval_res.page_accuracy,
                        "latency_ms": eval_res.latency_ms,
                        "is_perfect": eval_res.is_perfect,
                        "fields": [
                            {
                                "key": f.key,
                                "matched": f.matched,
                                "expected": f.expected,
                                "predicted": f.predicted,
                                "expected_found": f.expected_found,
                                "predicted_found": f.predicted_found,
                                "page_matched": f.page_matched,
                            }
                            for f in eval_res.fields
                        ],
                    }
                    f.write(json.dumps(doc_dict) + "\n")

            except Exception as e:
                console.print(f"[bold red]Failed on {doc_id}:[/bold red] {e}")
                logger.exception(f"Error evaluating {doc_id}")

            progress.advance(task_id)

    if resumed_count > 0 and evaluations:
        total_duration = sum(e.latency_ms for e in evaluations) / 1000.0
    else:
        total_duration = time.perf_counter() - start_time

    # ---------------------------------------------------------------------------
    # Aggregate Metrics Calculation & Final Reporting
    # ---------------------------------------------------------------------------
    _render_and_save_summary(evaluations, total_duration, summary_file, output_dir, split)


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Entry Point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ExtractBench benchmark runner for DocNeedle RAG extraction API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:7860",
        help="DocNeedle REST API base URL.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="short",
        choices=VALID_SPLITS,
        help="Dataset split to evaluate (short: 252, medium: 98, long: 20, all: 370).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to evaluate (default: None, runs entire split).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N documents in split (useful for resuming).",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Target a specific document ID (e.g. 'short/00581-2011-p0050').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=35,
        help="Schema items per API call (strictly <= 48 to stay within server 50-item limit).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="HTTP request timeout in seconds per batch (default: None, no timeout limit).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory to store evaluation reports and logs.",
    )
    parser.add_argument(
        "--enable-reranker",
        action="store_true",
        help="Enable Qwen3 cross-encoder reranker in retrieval.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face API token (if needed for rate limits).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load, parse, and validate dataset records without sending API requests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous run by skipping already evaluated documents.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-score detailed_{split}.jsonl results with updated normalizers without inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(
            run_benchmark(
                api_url=args.api_url,
                split=args.split,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                limit=args.limit,
                offset=args.offset,
                target_doc_id=args.doc_id,
                enable_reranker=args.enable_reranker,
                hf_token=args.hf_token,
                dry_run=args.dry_run,
                resume=args.resume,
                rescore=args.rescore,
                timeout=args.timeout,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark cancelled by user.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
