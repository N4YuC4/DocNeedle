"""Hugging Face Spaces and Local Entry Point for DocNeedle.

Provides a standalone, frontend-identical Gradio UI dashboard alongside the complete FastAPI REST API.
Features dynamic field-by-field schema creation matching frontend/src/App.tsx, dual formatted and raw JSON output tabs,
ephemeral OCR deallocation, and model bootstrapping.
Zero dependency on external frontend frameworks.
"""

import html
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Add src and root to sys.path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"

if src_dir.exists():
    sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(root_dir))

import gradio as gr  # noqa: E402

from docneedle.config import settings  # noqa: E402
from docneedle.main import app as fastapi_app  # noqa: E402

logger = logging.getLogger("DocNeedleSpace")
logging.basicConfig(level=logging.INFO)

# 1. Automatic model bootstrapping for Hugging Face Spaces and local startup
try:
    from scripts.download_models import ensure_all_models_present

    logger.info("Checking and bootstrapping quantized models...")
    ensure_all_models_present(settings.model_storage_dir)
except Exception as e:
    logger.warning(f"Automatic model download check failed or bypassed: {e}")


CUSTOM_CSS = """
/* Dark mode base styles */
:root, body, .gradio-container {
    max-width: 1360px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    background-color: #09090b !important;
    color: #f4f4f5 !important;
}

/* Headings and labels */
.gradio-container .prose *,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container p {
    color: #f4f4f5 !important;
}

/* Container blocks */
.gradio-container .block,
.gradio-container .form {
    background-color: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 8px !important;
}

/* Field Card matching frontend */
.dn-field-card {
    background-color: #141416 !important;
    border: 1px solid #27272a !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
    margin-bottom: 10px !important;
    transition: border-color 0.15s ease-in-out !important;
}
.dn-field-card:hover {
    border-color: #3f3f46 !important;
}

/* Add Field dashed button matching frontend */
.dn-add-field-btn {
    border: 1px dashed #3f3f46 !important;
    background-color: transparent !important;
    color: #a1a1aa !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    width: 100% !important;
    padding: 10px 14px !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
}
.dn-add-field-btn:hover {
    border-color: #71717a !important;
    color: #f4f4f5 !important;
    background-color: #27272a !important;
}

/* Delete button */
.dn-del-btn {
    background: transparent !important;
    border: none !important;
    color: #71717a !important;
    font-size: 14px !important;
    cursor: pointer !important;
    padding: 2px 6px !important;
    transition: color 0.15s ease-in-out !important;
}
.dn-del-btn:hover {
    color: #f43f5e !important;
}
 
/* Header action container */
.dn-header-actions {
    display: flex !important;
    gap: 6px !important;
    align-items: center !important;
    justify-content: flex-end !important;
}

/* Action button for import, export, clear */
.dn-action-btn {
    background-color: #27272a !important;
    border: 1px solid #3f3f46 !important;
    color: #e4e4e7 !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 4px 10px !important;
    min-width: 0 !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
}
.dn-action-btn:hover {
    background-color: #3f3f46 !important;
    color: #ffffff !important;
    border-color: #52525b !important;
}

/* Clear all button */
.dn-clear-btn {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: #71717a !important;
    font-size: 11px !important;
    cursor: pointer !important;
}
.dn-clear-btn:hover {
    color: #f43f5e !important;
    background-color: rgba(244, 63, 94, 0.1) !important;
    border-color: rgba(244, 63, 94, 0.2) !important;
}

/* Secondary toolbar button */
.dn-secondary-btn {
    background-color: #18181b !important;
    border: 1px solid #3f3f46 !important;
    color: #cbd5e1 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 6px 12px !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
}
.dn-secondary-btn:hover {
    background-color: #27272a !important;
    border-color: #71717a !important;
    color: #f4f4f5 !important;
}

.dn-raw-hint {
    font-size: 11px !important;
    color: #a1a1aa !important;
    margin-bottom: 6px !important;
}

.dn-input-tabs {
    margin-top: 4px !important;
}

/* Primary Action Button */
.dn-primary-btn {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 12px 18px !important;
    border: none !important;
    font-size: 14px !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
}
.dn-primary-btn:hover {
    background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%) !important;
    box-shadow: 0 4px 14px rgba(234, 88, 12, 0.35) !important;
}

/* File upload container & preview */
.gradio-container .file-preview-holder,
.gradio-container .upload-container,
.gradio-container [data-testid="dropzone"] {
    background-color: #141416 !important;
    border: 1px dashed #3f3f46 !important;
    color: #f4f4f5 !important;
}

/* Output Tabs */
.gradio-container .tab-nav {
    border-bottom: 1px solid #27272a !important;
}
.gradio-container .tab-nav button {
    color: #a1a1aa !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-bottom: 2px solid transparent !important;
}
.gradio-container .tab-nav button.selected {
    color: #f4f4f5 !important;
    border-bottom-color: #f97316 !important;
    font-weight: 600 !important;
}

/* JSON viewer */
.gradio-container .json-holder {
    background-color: #09090b !important;
    border: 1px solid #27272a !important;
    border-radius: 6px !important;
}

/* Hide Gradio footer completely */
footer, .gradio-container footer, .gradio-container .footer {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}
"""

HEADER_HTML = """
<div style="display: flex; align-items: center; justify-content: space-between; padding: 14px 20px; background: #18181b; border: 1px solid #27272a; margin: -20px -20px 20px -20px; border-radius: 8px;">
    <div style="display: flex; align-items: center; gap: 14px;">
        <svg viewBox="170 170 684 684" style="width: 38px; height: 38px; flex-shrink: 0;" fill="none" xmlns="http://www.w3.org/2000/svg">
            <g>
                <path d="M 727 271 L 418 270 L 294 400 L 297 708 L 320 704 L 328 704 L 304 723 L 294 729 L 294 763 L 297 762 L 343 717 L 399 657 L 469 576 L 639 371 Z" fill="#2E46AA" />
                <path d="M 418 270 L 294 400 L 418 400 Z" fill="#256CD2" />
                <path d="M 728 444 L 671 501 L 552 611 L 464 695 L 385 775 L 330 834 L 317 850 L 341 849 L 349 838 L 367 818 L 368 822 L 363 850 L 729 849 L 729 506 L 723 506 L 693 513 L 688 512 L 728 480 Z" fill="#09BDCE" />
                <path d="M 741 338 L 733 335 L 723 335 L 704 343 L 679 363 L 658 385 L 563 502 L 296 819 L 372 748 L 601 528 L 688 448 L 721 415 L 743 386 L 752 364 L 751 350 L 749 346 Z M 726 367 L 725 378 L 719 389 L 709 402 L 685 425 L 677 430 L 671 432 L 665 432 L 659 425 L 660 416 L 665 406 L 678 389 L 693 374 L 705 365 L 715 361 L 719 361 Z" fill="#FFFFFF" stroke="#0F172A" stroke-width="8" stroke-linejoin="round" />
            </g>
        </svg>
        <div>
            <div style="font-weight: 700; font-size: 17px; color: #f4f4f5; letter-spacing: -0.02em;">DocNeedle</div>
            <div style="font-size: 12px; color: #a1a1aa; font-weight: 500;">Document Extraction Microservice</div>
        </div>
    </div>
</div>
"""

FIELD_TYPE_KEYS = [
    "string",
    "number",
    "money",
    "date",
    "time",
    "percentage",
    "person",
    "organization",
    "id",
    "iban",
    "email",
    "phone",
    "address",
    "boolean",
    "list",
]

TYPE_MAP = {
    "string": "string",
    "text": "string",
    "metin": "string",
    "number": "number",
    "integer": "number",
    "float": "number",
    "sayı": "number",
    "money": "money",
    "currency": "money",
    "amount": "money",
    "para": "money",
    "date": "date",
    "tarih": "date",
    "time": "time",
    "saat": "time",
    "email": "email",
    "phone": "phone",
    "id": "id",
    "iban": "iban",
    "percentage": "percentage",
    "percent": "percentage",
    "oran": "percentage",
    "person": "person",
    "name": "person",
    "organization": "organization",
    "company": "organization",
    "address": "address",
    "boolean": "boolean",
    "list": "list",
}

MAX_FIELDS = 10

INITIAL_STARTER_FIELDS = [
    ("", "", "string"),
]


def parse_raw_json_schema(raw_json: str) -> list[dict[str, str]]:
    """Parse raw JSON string into a validated list of extraction item dicts."""
    if not raw_json or not raw_json.strip():
        return []
    try:
        data = json.loads(raw_json)
    except Exception as e:
        raise ValueError(f"Invalid JSON syntax: {e}") from e

    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError("Schema must be a JSON array of objects or a single JSON object.")

    parsed_items: list[dict[str, str]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        key = str(
            item.get("key") or item.get("name") or item.get("id") or item.get("field") or ""
        ).strip()
        desc = str(
            item.get("description")
            or item.get("question")
            or item.get("prompt")
            or item.get("desc")
            or ""
        ).strip()
        t = str(item.get("type", "string")).strip().lower()
        resolved_t = TYPE_MAP.get(t, t) if t not in ("none", "n/a", "") else "string"
        if resolved_t not in FIELD_TYPE_KEYS:
            resolved_t = "string"

        if key or desc:
            final_key = key or desc or f"item_{idx + 1}"
            final_desc = desc or key
            parsed_items.append(
                {
                    "key": final_key,
                    "description": final_desc,
                    "type": resolved_t,
                }
            )
    return parsed_items


def serialize_form_to_json(
    q_vals: list[Any],
    k_vals: list[Any],
    t_vals: list[Any],
    vis_vals: list[bool] | None = None,
) -> str:
    """Convert form fields to formatted JSON string."""
    items: list[dict[str, str]] = []
    num_fields = min(len(q_vals), len(k_vals), len(t_vals))
    for i in range(num_fields):
        is_vis = vis_vals[i] if vis_vals is not None and i < len(vis_vals) else True
        if not is_vis:
            continue
        q = str(q_vals[i]).strip() if q_vals[i] else ""
        k = str(k_vals[i]).strip() if k_vals[i] else ""
        t = str(t_vals[i]).strip().lower() if t_vals[i] else "string"
        resolved_t = TYPE_MAP.get(t, t) if t not in ("none", "n/a", "") else "string"
        if resolved_t not in FIELD_TYPE_KEYS:
            resolved_t = "string"

        if q or k:
            items.append(
                {
                    "key": k or q or f"item_{len(items) + 1}",
                    "description": q or k,
                    "type": resolved_t,
                }
            )

    if not items:
        return json.dumps(
            [
                {
                    "key": "example_field",
                    "description": "Describe what you want to extract from the document",
                    "type": "string",
                }
            ],
            indent=2,
            ensure_ascii=False,
        )

    return json.dumps(items, indent=2, ensure_ascii=False)


def apply_items_to_form_updates(
    items: list[dict[str, str]],
) -> tuple[list[bool], list[Any], list[Any], list[Any], list[Any], str]:
    """Given parsed items, generate updates for vis_state, field_groups, q_inputs, k_inputs, t_inputs, and raw_json."""
    count = min(len(items), MAX_FIELDS)
    new_vis = [(i < count) for i in range(MAX_FIELDS)]
    if count == 0:
        new_vis[0] = True

    group_updates = [gr.update(visible=new_vis[i]) for i in range(MAX_FIELDS)]
    q_updates: list[Any] = []
    k_updates: list[Any] = []
    t_updates: list[Any] = []

    for i in range(MAX_FIELDS):
        if i < count:
            it = items[i]
            q_updates.append(gr.update(value=it.get("description", "")))
            k_updates.append(gr.update(value=it.get("key", "")))
            t_updates.append(gr.update(value=it.get("type", "string")))
        else:
            q_updates.append(gr.update(value=""))
            k_updates.append(gr.update(value=""))
            t_updates.append(gr.update(value="string"))

    json_str = json.dumps(
        items
        if items
        else [
            {
                "key": "example_field",
                "description": "Describe what you want to extract from the document",
                "type": "string",
            }
        ],
        indent=2,
        ensure_ascii=False,
    )

    return new_vis, group_updates, q_updates, k_updates, t_updates, json_str


def render_results_html(response_dict: dict[str, Any] | None) -> str:
    """Render extracted results matching frontend/src/App.tsx card design."""
    if not response_dict:
        return """
        <div style="border: 2px dashed #27272a; border-radius: 8px; padding: 56px 24px; text-align: center; color: #a1a1aa; background: #141416;">
            <svg style="width: 40px; height: 40px; margin: 0 auto 12px; color: #71717a;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
            </svg>
            <div style="font-weight: 600; font-size: 15px; color: #f4f4f5;">No analysis performed yet</div>
            <div style="font-size: 13px; margin-top: 6px; color: #a1a1aa;">Upload documents on the left and define target fields to start extraction.</div>
        </div>
        """

    if "error" in response_dict:
        err_msg = html.escape(str(response_dict["error"]))
        return f"""
        <div style="border: 1px solid #881337; background: #4c0519; color: #fecdd3; border-radius: 8px; padding: 16px; font-size: 13px;">
            <strong>Extraction Error:</strong> {err_msg}
        </div>
        """

    total = response_dict.get("total_items", 0)
    found = response_dict.get("found_items", 0)
    time_ms = response_dict.get("processing_time_ms", 0)
    time_s = f"{time_ms / 1000:.2f} s"
    results = response_dict.get("results", [])

    cards_html = ""
    for r in results:
        is_found = r.get("found", False)
        key = html.escape(str(r.get("key", "")))
        raw_val = r.get("value")
        val = (
            html.escape(str(raw_val))
            if raw_val is not None and str(raw_val) != ""
            else "Not found in document"
        )
        conf = int(r.get("confidence", 0) * 100)
        evidence = r.get("evidence")
        thinking = r.get("thinking")

        badge_html = (
            '<span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">Found</span>'
            if is_found
            else '<span style="background: #27272a; color: #a1a1aa; border: 1px solid #3f3f46; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">Not Found</span>'
        )
        conf_html = (
            f'<span style="font-family: monospace; font-size: 11px; color: #a1a1aa;">%{conf} Confidence</span>'
            if is_found
            else ""
        )

        evidence_html = ""
        if evidence:
            doc_name = html.escape(str(evidence.get("document", "")))
            page_num = evidence.get("page")
            raw_snippet = evidence.get("snippet", "")
            snippet = html.escape(str(raw_snippet)) if raw_snippet else ""
            page_str = f" • Page {int(page_num)}" if isinstance(page_num, int) else ""

            snippet_box = (
                f'<div style="margin-top: 6px; padding: 8px 10px; background: #09090b; border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 11px; color: #cbd5e1; font-style: italic;">"{snippet}"</div>'
                if snippet
                else ""
            )

            evidence_html = f"""
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #27272a; font-size: 11px; color: #a1a1aa;">
                <span style="font-weight: 600; color: #f4f4f5;">Source Document:</span> {doc_name}{page_str}
                {snippet_box}
            </div>
            """

        thinking_html = ""
        if thinking:
            safe_thinking = html.escape(str(thinking))
            thinking_html = f"""
            <details style="margin-top: 8px; font-size: 11px; color: #a1a1aa;">
                <summary style="cursor: pointer; font-weight: 500; color: #94a3b8;">🧠 Model Reasoning (Thinking Process)</summary>
                <div style="margin-top: 4px; padding: 8px; background: #09090b; border: 1px solid #27272a; border-radius: 4px; font-family: monospace; white-space: pre-wrap; color: #e2e8f0; font-size: 11px;">{safe_thinking}</div>
            </details>
            """

        cards_html += f"""
        <div style="background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 14px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-family: monospace; font-size: 13px; font-weight: 700; color: #f4f4f5;">{key}</span>
                    {badge_html}
                </div>
                {conf_html}
            </div>
            <div style="background: #09090b; border: 1px solid #27272a; border-radius: 6px; padding: 10px 12px; font-size: 13px; font-weight: 600; color: #f4f4f5; word-break: break-word;">
                {val}
            </div>
            {evidence_html}
            {thinking_html}
        </div>
        """

    summary_html = f"""
    <div style="background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.3);">
        <div style="display: flex; gap: 24px;">
            <div>
                <span style="font-size: 11px; color: #a1a1aa; display: block; font-weight: 500;">Extracted Fields</span>
                <span style="font-family: monospace; font-size: 16px; font-weight: 700; color: #f4f4f5;">{found} / {total}</span>
            </div>
            <div style="border-left: 1px solid #27272a; padding-left: 24px;">
                <span style="font-size: 11px; color: #a1a1aa; display: block; font-weight: 500;">Processing Time</span>
                <span style="font-family: monospace; font-size: 16px; font-weight: 700; color: #f4f4f5;">{time_s}</span>
            </div>
        </div>
        <div>
            <span style="background: #27272a; color: #e4e4e7; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; border: 1px solid #3f3f46;">Extraction Completed</span>
        </div>
    </div>
    """

    return summary_html + cards_html


def make_status_html(message: str, is_success: bool = False, is_loading: bool = False) -> str:
    border_color = "#10b981" if is_success else ("#f97316" if is_loading else "#27272a")
    bg_color = (
        "rgba(16, 185, 129, 0.08)"
        if is_success
        else ("rgba(249, 115, 22, 0.08)" if is_loading else "#141416")
    )
    text_color = "#34d399" if is_success else ("#fb923c" if is_loading else "#a1a1aa")
    icon = "✅" if is_success else ("⏳" if is_loading else "📄")
    return f"""
    <div style="padding: 10px 14px; background: {bg_color}; border: 1px solid {border_color}; border-radius: 6px; font-size: 12px; color: {text_color}; margin-top: 10px; display: flex; align-items: center; gap: 8px;">
        <span>{icon}</span>
        <span>{message}</span>
    </div>
    """


async def perform_extraction_ui(
    files: Any,
    session_chunks: list[Any] | None,
    active_tab: str,
    raw_json: str,
    vis: list[bool],
    *field_args: Any,
) -> tuple[str, dict[str, Any], list[Any], str]:
    """Execute complete document extraction pipeline from dynamic Gradio UI fields or raw JSON schema.

    Uses initial OCR/ingested chunks from session state to avoid redundant OCR runs on queries.
    """
    if not files and not session_chunks:
        err = {"error": "Please upload at least one document (PDF, DOCX, XLSX, or Image)."}
        return (
            render_results_html(err),
            err,
            [],
            make_status_html("Please upload at least one document to start.", is_success=False),
        )

    file_list = files if isinstance(files, list) else ([files] if files else [])
    if len(file_list) == 0 and not session_chunks:
        err = {"error": "Please upload at least one document."}
        return (
            render_results_html(err),
            err,
            [],
            make_status_html("Please upload at least one document.", is_success=False),
        )

    from docneedle.extraction.schemas import ExtractionItemQuery
    from docneedle.extraction.service import ExtractionService
    from docneedle.ingest.service import IngestionService
    from docneedle.retrieval.store import SessionVectorStore

    queries: list[ExtractionItemQuery] = []
    use_raw = active_tab == "raw"

    # If raw mode is active, parse queries from raw_json
    if use_raw and raw_json and raw_json.strip():
        try:
            parsed_items = parse_raw_json_schema(raw_json)
            for it in parsed_items:
                queries.append(
                    ExtractionItemQuery(
                        key=it["key"],
                        description=it["description"],
                        type=it["type"],
                    )
                )
        except Exception as e:
            err = {"error": f"Invalid JSON schema: {e}"}
            return (
                render_results_html(err),
                err,
                list(session_chunks) if session_chunks else [],
                make_status_html(f"Schema Error: {e}", is_success=False),
            )

    # Otherwise (or if raw mode was empty), collect from form builder fields
    if not queries:
        num_fields = MAX_FIELDS
        q_vals = field_args[0:num_fields]
        k_vals = field_args[num_fields : 2 * num_fields]
        t_vals = field_args[2 * num_fields : 3 * num_fields]

        for i in range(num_fields):
            if i < len(vis) and not vis[i]:
                continue
            q = str(q_vals[i]).strip() if i < len(q_vals) and q_vals[i] else ""
            k = str(k_vals[i]).strip() if i < len(k_vals) and k_vals[i] else ""
            t = str(t_vals[i]).strip().lower() if i < len(t_vals) and t_vals[i] else "string"
            if q or k:
                key_name = k or q or f"item_{len(queries) + 1}"
                desc = q or k
                resolved_type = TYPE_MAP.get(t, t) if t not in ("none", "n/a", "") else "string"
                queries.append(
                    ExtractionItemQuery(
                        key=key_name,
                        description=desc,
                        type=resolved_type,
                    )
                )

    if not queries:
        err = {"error": "Please define at least one field to extract via Form Builder or Raw JSON."}
        return (
            render_results_html(err),
            err,
            list(session_chunks) if session_chunks else [],
            make_status_html("Please define at least one field to extract.", is_success=False),
        )

    ingest_svc = IngestionService()
    extraction_svc = ExtractionService()
    store = SessionVectorStore()

    try:
        chunks_to_use = list(session_chunks) if session_chunks else []
        if not chunks_to_use:
            with ingest_svc.parser:
                for f in file_list:
                    file_path = Path(f.name if hasattr(f, "name") else str(f))
                    content = file_path.read_bytes()
                    filename = file_path.name
                    ingest_res = ingest_svc.ingest_bytes(content=content, filename=filename)
                    chunks_to_use.extend(ingest_res.chunks)

        if not chunks_to_use:
            err = {"error": "Could not extract readable text from the uploaded documents."}
            return (
                render_results_html(err),
                err,
                [],
                make_status_html(
                    "Could not extract readable text from documents.", is_success=False
                ),
            )

        # Reclaim RAM between ingestion and retrieval
        import gc

        gc.collect()

        logger.info(
            f"--> [PIPELINE] Indexing {len(chunks_to_use)} text chunks across {len(file_list)} files into in-memory vector store..."
        )
        store.add_chunks(chunks_to_use)
        logger.info(
            f"--> [PIPELINE] Starting extraction pipeline for {len(queries)} target field query(s)..."
        )
        response = await extraction_svc.extract_items(queries, store)
        resp_dict = response.model_dump()
        status_msg = (
            f"Extraction complete: {resp_dict.get('found_items', 0)} of "
            f"{resp_dict.get('total_items', 0)} items found in "
            f"{resp_dict.get('processing_time_ms', 0):.0f} ms."
        )

        return (
            render_results_html(resp_dict),
            resp_dict,
            chunks_to_use,
            make_status_html(status_msg, is_success=True),
        )
    except Exception as e:
        logger.error(f"UI extraction error: {e}", exc_info=True)
        err = {"error": str(e)}
        return (
            render_results_html(err),
            err,
            list(session_chunks) if session_chunks else [],
            make_status_html(f"Error: {e}", is_success=False),
        )
    finally:
        store.clear()
        import gc

        gc.collect()


def check_system_health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "ocr_model": settings.ocr_model_name,
        "llm_model": settings.llm_model_name,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "gpu_enabled": settings.gpu_enabled,
    }


# 2. Build interactive Gradio Dashboard (dark mode & English)
with gr.Blocks(title="DocNeedle - Document Extraction") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        # Left Column: Documents & Dynamic Fields (5 Columns)
        with gr.Column(scale=5):
            with gr.Group():
                gr.Markdown(
                    "### 📁 DOCUMENTS\nUpload PDFs, scanned files, spreadsheets, or text files"
                )
                file_input = gr.File(
                    label="Upload Documents (Drag & Drop)",
                    file_count="multiple",
                    file_types=[
                        ".pdf",
                        ".docx",
                        ".doc",
                        ".xlsx",
                        ".xls",
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                        ".txt",
                    ],
                )

                session_chunks_state = gr.State([])
                doc_status_view = gr.HTML(
                    value=make_status_html(
                        "Upload documents to begin. Processing will run when you click Extract Data."
                    )
                )

            with gr.Group():
                with gr.Row():
                    gr.Markdown(
                        "### 🎯 FIELDS TO EXTRACT\nDefine target fields via form or raw JSON"
                    )
                    with gr.Row(elem_classes=["dn-header-actions"]):
                        import_schema_btn = gr.UploadButton(
                            "📥 Import",
                            file_types=[".json"],
                            elem_classes=["dn-action-btn"],
                            size="sm",
                        )
                        export_schema_btn = gr.DownloadButton(
                            "📤 Export",
                            elem_classes=["dn-action-btn"],
                            size="sm",
                        )
                        clear_all_btn = gr.Button(
                            "🗑️ Clear",
                            elem_classes=["dn-action-btn", "dn-clear-btn"],
                            size="sm",
                        )

                active_tab_state = gr.State("form")

                with gr.Tabs(elem_classes=["dn-input-tabs"]) as input_tabs:
                    with gr.Tab("📝 Form Builder", id="tab_form_input") as tab_form_input:
                        vis_state = gr.State(
                            [True] * len(INITIAL_STARTER_FIELDS)
                            + [False] * (MAX_FIELDS - len(INITIAL_STARTER_FIELDS))
                        )
                        field_groups: list[Any] = []
                        q_inputs: list[Any] = []
                        k_inputs: list[Any] = []
                        t_inputs: list[Any] = []
                        del_btns: list[Any] = []

                        for i in range(MAX_FIELDS):
                            is_init_visible = i < len(INITIAL_STARTER_FIELDS)
                            init_q = INITIAL_STARTER_FIELDS[i][0] if is_init_visible else ""
                            init_k = INITIAL_STARTER_FIELDS[i][1] if is_init_visible else ""
                            init_t = INITIAL_STARTER_FIELDS[i][2] if is_init_visible else "string"

                            with gr.Group(
                                visible=is_init_visible, elem_classes=["dn-field-card"]
                            ) as f_grp:
                                with gr.Row():
                                    q_box = gr.Textbox(
                                        value=init_q,
                                        label="Field Question / Prompt",
                                        placeholder="e.g. What is the total invoice amount including VAT?",
                                        scale=9,
                                    )
                                    d_btn = gr.Button(
                                        "🗑️",
                                        scale=1,
                                        min_width=36,
                                        elem_classes=["dn-del-btn"],
                                    )
                                with gr.Row():
                                    k_box = gr.Textbox(
                                        value=init_k,
                                        label="Target Key (Optional)",
                                        placeholder="e.g. total_amount",
                                        scale=1,
                                    )
                                    t_dropdown = gr.Dropdown(
                                        choices=FIELD_TYPE_KEYS,
                                        value=init_t,
                                        label="Expected Type",
                                        scale=1,
                                    )

                            field_groups.append(f_grp)
                            q_inputs.append(q_box)
                            k_inputs.append(k_box)
                            t_inputs.append(t_dropdown)
                            del_btns.append(d_btn)

                        add_field_btn = gr.Button("＋ Add Field", elem_classes=["dn-add-field-btn"])

                    with gr.Tab("📊 Raw JSON View", id="tab_raw_input") as tab_raw_input:
                        gr.Markdown(
                            "*Enter or edit schema directly as a JSON list of objects (`key`, `description`, `type`).*",
                            elem_classes=["dn-raw-hint"],
                        )
                        raw_json_input = gr.Code(
                            value=serialize_form_to_json(
                                [f[0] for f in INITIAL_STARTER_FIELDS],
                                [f[1] for f in INITIAL_STARTER_FIELDS],
                                [f[2] for f in INITIAL_STARTER_FIELDS],
                            ),
                            language="json",
                            label="JSON Schema Specification",
                            lines=12,
                        )
                        with gr.Row():
                            apply_raw_to_form_btn = gr.Button(
                                "⬇️ Apply to Form",
                                elem_classes=["dn-secondary-btn"],
                                size="sm",
                            )
                            sync_form_to_raw_btn = gr.Button(
                                "⬆️ Sync from Form",
                                elem_classes=["dn-secondary-btn"],
                                size="sm",
                            )

            extract_btn = gr.Button(
                "Extract Data",
                variant="primary",
                elem_classes=["dn-primary-btn"],
            )

        # Right Column: Result Cards & Provenance (7 Columns) with Dual Views
        with gr.Column(scale=7):
            with gr.Tabs(elem_classes=["dn-output-tabs"]):
                with gr.Tab("📋 Formatted View", id="tab_formatted"):
                    results_html_view = gr.HTML(value=render_results_html(None))

                with gr.Tab("📊 Raw JSON View", id="tab_raw"):
                    raw_json_output = gr.JSON(label="Complete JSON Extraction Response")

            with gr.Accordion("⚙️ System Health & API Info", open=False):
                health_btn = gr.Button("Check System Health", size="sm")
                health_json = gr.JSON(label="System Details")
                health_btn.click(check_system_health, outputs=health_json, api_name=False)
                gr.Markdown(
                    """
                    * **Swagger API Documentation:** [/docs](/docs)
                    * **Extraction Endpoint:** `POST /v1/extract`
                    * **Health Endpoint:** `GET /healthz`
                    """
                )

    # Event handlers for dynamic schema builder (all internal UI events with api_name=False)
    def on_add_field(vis: list[bool]) -> list[Any]:
        new_vis = list(vis)
        for idx in range(len(new_vis)):
            if not new_vis[idx]:
                new_vis[idx] = True
                break
        return [new_vis] + [gr.update(visible=v) for v in new_vis]

    add_field_btn.click(
        on_add_field,
        inputs=[vis_state],
        outputs=[vis_state] + field_groups,
        api_name=False,
    )

    def make_delete_handler(target_idx: int):
        def on_delete(vis: list[bool]) -> tuple[Any, ...]:
            new_vis = list(vis)
            new_vis[target_idx] = False
            return (
                new_vis,
                gr.update(visible=False),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value="string"),
            )

        return on_delete

    for idx, d_btn in enumerate(del_btns):
        d_btn.click(
            make_delete_handler(idx),
            inputs=[vis_state],
            outputs=[vis_state, field_groups[idx], q_inputs[idx], k_inputs[idx], t_inputs[idx]],
            api_name=False,
        )

    def on_clear_all() -> list[Any]:
        new_vis = [True] + [False] * (MAX_FIELDS - 1)
        updates: list[Any] = [new_vis]
        for i in range(MAX_FIELDS):
            updates.append(gr.update(visible=(i == 0)))
        for i in range(MAX_FIELDS):
            updates.append(gr.update(value=""))
        for i in range(MAX_FIELDS):
            updates.append(gr.update(value=""))
        for i in range(MAX_FIELDS):
            updates.append(gr.update(value="string"))
        empty_json = json.dumps(
            [
                {
                    "key": "example_field",
                    "description": "Describe what you want to extract from the document",
                    "type": "string",
                }
            ],
            indent=2,
        )
        updates.append(empty_json)
        return updates

    clear_all_btn.click(
        on_clear_all,
        outputs=[vis_state] + field_groups + q_inputs + k_inputs + t_inputs + [raw_json_input],
        api_name=False,
    )

    def on_sync_form_to_raw(vis: list[bool], *field_args: Any) -> str:
        num_fields = MAX_FIELDS
        q_vals = field_args[0:num_fields]
        k_vals = field_args[num_fields : 2 * num_fields]
        t_vals = field_args[2 * num_fields : 3 * num_fields]
        return serialize_form_to_json(q_vals, k_vals, t_vals, vis)

    sync_form_to_raw_btn.click(
        on_sync_form_to_raw,
        inputs=[vis_state] + q_inputs + k_inputs + t_inputs,
        outputs=[raw_json_input],
        api_name=False,
    )

    def on_apply_raw_to_form(raw_json: str) -> tuple[Any, ...]:
        try:
            items = parse_raw_json_schema(raw_json)
            if not items:
                gr.Warning("No valid fields found in raw JSON schema.")
                return (gr.skip(),) * (1 + MAX_FIELDS * 4)
            new_vis, grp_ups, q_ups, k_ups, t_ups, _ = apply_items_to_form_updates(items)
            gr.Info(f"Loaded {len(items)} fields into Form Builder.")
            return tuple([new_vis] + grp_ups + q_ups + k_ups + t_ups)
        except Exception as e:
            gr.Warning(f"Invalid JSON: {e}")
            return (gr.skip(),) * (1 + MAX_FIELDS * 4)

    apply_raw_to_form_btn.click(
        on_apply_raw_to_form,
        inputs=[raw_json_input],
        outputs=[vis_state] + field_groups + q_inputs + k_inputs + t_inputs,
        api_name=False,
    )

    def on_import_schema(file: Any) -> tuple[Any, ...]:
        if not file:
            return (gr.skip(),) * (1 + MAX_FIELDS * 4 + 1)
        try:
            file_path = Path(file.name if hasattr(file, "name") else str(file))
            content = file_path.read_text(encoding="utf-8")
            items = parse_raw_json_schema(content)
            if not items:
                gr.Warning("No valid extraction fields found in imported JSON file.")
                return (gr.skip(),) * (1 + MAX_FIELDS * 4 + 1)
            new_vis, grp_ups, q_ups, k_ups, t_ups, json_str = apply_items_to_form_updates(items)
            gr.Info(f"Successfully imported {len(items)} fields from file.")
            return tuple([new_vis] + grp_ups + q_ups + k_ups + t_ups + [json_str])
        except Exception as e:
            gr.Warning(f"Import failed: {e}")
            return (gr.skip(),) * (1 + MAX_FIELDS * 4 + 1)

    import_schema_btn.upload(
        on_import_schema,
        inputs=[import_schema_btn],
        outputs=[vis_state] + field_groups + q_inputs + k_inputs + t_inputs + [raw_json_input],
        api_name=False,
    )

    def on_export_schema(
        active_tab: str,
        raw_json: str,
        vis: list[bool],
        *field_args: Any,
    ) -> str:
        items: list[dict[str, str]] = []
        if active_tab == "raw" and raw_json and raw_json.strip():
            try:
                items = parse_raw_json_schema(raw_json)
            except Exception:
                items = []

        if not items:
            num_fields = MAX_FIELDS
            q_vals = field_args[0:num_fields]
            k_vals = field_args[num_fields : 2 * num_fields]
            t_vals = field_args[2 * num_fields : 3 * num_fields]
            for i in range(num_fields):
                if i < len(vis) and not vis[i]:
                    continue
                q = str(q_vals[i]).strip() if i < len(q_vals) and q_vals[i] else ""
                k = str(k_vals[i]).strip() if i < len(k_vals) and k_vals[i] else ""
                t = str(t_vals[i]).strip().lower() if i < len(t_vals) and t_vals[i] else "string"
                if q or k:
                    items.append(
                        {
                            "key": k or q or f"item_{len(items) + 1}",
                            "description": q or k,
                            "type": TYPE_MAP.get(t, t)
                            if t not in ("none", "n/a", "")
                            else "string",
                        }
                    )

        if not items:
            items = [
                {
                    "key": "example_field",
                    "description": "Describe what you want to extract from the document",
                    "type": "string",
                }
            ]

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="docneedle_schema_",
            delete=False,
            encoding="utf-8",
        )
        json.dump(items, tmp, indent=2, ensure_ascii=False)
        tmp.close()
        return tmp.name

    export_schema_btn.click(
        on_export_schema,
        inputs=[active_tab_state, raw_json_input, vis_state] + q_inputs + k_inputs + t_inputs,
        outputs=[export_schema_btn],
        api_name=False,
    )

    tab_form_input.select(lambda: "form", outputs=[active_tab_state], api_name=False)

    def on_select_raw_tab(vis: list[bool], *field_args: Any) -> tuple[str, str]:
        num_fields = MAX_FIELDS
        q_vals = field_args[0:num_fields]
        k_vals = field_args[num_fields : 2 * num_fields]
        t_vals = field_args[2 * num_fields : 3 * num_fields]
        json_str = serialize_form_to_json(q_vals, k_vals, t_vals, vis)
        return "raw", json_str

    tab_raw_input.select(
        on_select_raw_tab,
        inputs=[vis_state] + q_inputs + k_inputs + t_inputs,
        outputs=[active_tab_state, raw_json_input],
        api_name=False,
    )

    def on_documents_changed(files: Any) -> tuple[str, list[Any]]:
        if not files:
            return (
                make_status_html(
                    "Upload documents to begin. Processing will run when you click Extract Data."
                ),
                [],
            )
        file_list = files if isinstance(files, list) else [files]
        if not file_list:
            return (
                make_status_html(
                    "Upload documents to begin. Processing will run when you click Extract Data."
                ),
                [],
            )

        names = [Path(f.name if hasattr(f, "name") else str(f)).name for f in file_list]
        total_kb = (
            sum(
                Path(f.name if hasattr(f, "name") else str(f)).stat().st_size
                for f in file_list
                if Path(f.name if hasattr(f, "name") else str(f)).exists()
            )
            / 1024
        )
        names_str = ", ".join(names[:3]) + (f" (+{len(names) - 3} more)" if len(names) > 3 else "")
        status_text = (
            f"📁 <strong>{len(file_list)} document(s) selected</strong> ({names_str} · {total_kb:.1f} KB). "
            "Click <strong>Extract Data</strong> to process."
        )
        return make_status_html(status_text, is_success=False), []

    file_input.change(
        on_documents_changed,
        inputs=[file_input],
        outputs=[doc_status_view, session_chunks_state],
        api_name=False,
    )

    extract_btn.click(
        perform_extraction_ui,
        inputs=[
            file_input,
            session_chunks_state,
            active_tab_state,
            raw_json_input,
            vis_state,
        ]
        + q_inputs
        + k_inputs
        + t_inputs,
        outputs=[results_html_view, raw_json_output, session_chunks_state, doc_status_view],
        api_name=False,
    )

# 3. Mount Gradio to root path with footer_links=[] to completely remove auto-generated API footer
if "pytest" not in sys.modules:
    fastapi_app.router.routes = [
        r for r in fastapi_app.router.routes if getattr(r, "path", None) != "/"
    ]
    app = gr.mount_gradio_app(fastapi_app, demo, path="/", css=CUSTOM_CSS, footer_links=[])
else:
    app = fastapi_app

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
