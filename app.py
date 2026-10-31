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

# Suppress benign asyncio event loop destructor log noise on Python 3.12+
try:
    import asyncio.base_events as _base_events

    _orig_loop_del = _base_events.BaseEventLoop.__del__

    def _silent_loop_del(self: Any) -> None:
        try:
            _orig_loop_del(self)
        except Exception as _e:
            if "Invalid file descriptor" not in str(_e):
                raise

    _base_events.BaseEventLoop.__del__ = _silent_loop_del
except Exception:
    pass

# ZeroGPU decorator compatibility (applied before CUDA imports)
try:
    import spaces

    gpu_decorator = spaces.GPU
except Exception:

    def gpu_decorator(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


# Add src and root to sys.path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"

if src_dir.exists():
    sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(root_dir))

# Preload NVIDIA CUDA runtime and cuBLAS shared libraries into global symbol space
# (libcudart.so.12, libcublas.so.12) to ensure llama-cpp-python can link on ZeroGPU
try:
    from docneedle.cuda_compat import preload_cuda_runtime_libs

    preload_cuda_runtime_libs()
except Exception as e:
    logging.getLogger("DocNeedleSpace").debug("CUDA runtime preload info: %s", e)

import gradio as gr  # noqa: E402

from docneedle.config import settings  # noqa: E402

logger = logging.getLogger("DocNeedleSpace")
logging.basicConfig(level=logging.INFO)

# 1. Automatic model bootstrapping for Hugging Face Spaces and local startup
if os.environ.get("SKIP_MODEL_BOOTSTRAP", "").lower() not in ("1", "true"):
    try:
        from scripts.download_models import ensure_all_models_present

        logger.info("Checking and bootstrapping quantized models...")
        ensure_all_models_present(settings.model_storage_dir)
        os.environ["SKIP_MODEL_BOOTSTRAP"] = "1"
    except Exception as e:
        logger.warning(f"Automatic model download check failed or bypassed: {e}")


CUSTOM_CSS = """
/* DocNeedle Design System Tokens */
:root {
    --dn-bg: #09090b;
    --dn-surface: #18181b;
    --dn-surface-card: #141416;
    --dn-border: #27272a;
    --dn-border-hover: #3f3f46;
    --dn-text-main: #f4f4f5;
    --dn-text-muted: #a1a1aa;
    --dn-text-subtle: #71717a;
    --dn-primary: #f97316;
    --dn-primary-hover: #ea580c;
    --dn-accent-blue: #256cd2;
    --dn-accent-cyan: #09bdce;
    --dn-success: #10b981;
    --dn-warning: #f59e0b;
    --dn-danger: #f43f5e;
}

/* Base styles */
:root, html, body {
    margin: 0 !important;
    padding: 0 !important;
    background-color: var(--dn-bg) !important;
    color: var(--dn-text-main) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    overflow-x: hidden !important;
    width: 100% !important;
}

.gradio-container {
    max-width: 1560px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    margin: 0 auto !important;
    padding: 16px !important;
    background-color: var(--dn-bg) !important;
    color: var(--dn-text-main) !important;
}

/* Accessible focus visible rings */
button:focus-visible,
input:focus-visible,
select:focus-visible,
textarea:focus-visible,
[tabindex="0"]:focus-visible {
    outline: 2px solid var(--dn-primary) !important;
    outline-offset: 2px !important;
}

/* Clean typography */
.gradio-container .prose *,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container p {
    color: var(--dn-text-main) !important;
    word-break: normal !important;
    overflow-wrap: break-word !important;
}

/* Main dashboard 3-column layout (1: Dosya Yükleme, 2: Alan Girdileri, 3: Sonuç) */
.dn-main-layout {
    display: flex !important;
    flex-direction: row !important;
    gap: 14px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    align-items: flex-start !important;
}

.dn-col-docs {
    flex: 3 1 0% !important;
    min-width: 0 !important;
}

.dn-col-fields {
    flex: 4 1 0% !important;
    min-width: 0 !important;
}

.dn-col-results {
    flex: 5 1 0% !important;
    min-width: 0 !important;
}

/* Responsive layout: stack columns vertically on screens <= 1100px */
@media (max-width: 1100px) {
    .gradio-container {
        padding: 10px 14px !important;
        max-width: 100% !important;
    }
    .gradio-container .main {
        max-width: 100% !important;
        width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .gradio-container .contain,
    .gradio-container .wrap {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    .dn-main-layout {
        flex-direction: column !important;
        width: 100% !important;
        gap: 12px !important;
        margin: 0 !important;
    }
    .dn-col-docs,
    .dn-col-results,
    .dn-col-fields,
    .dn-left-panel,
    .dn-right-panel {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    .dn-card-group {
        padding: 12px 14px !important;
        margin-bottom: 12px !important;
    }
    .dn-header-banner {
        padding: 12px 14px !important;
        margin-bottom: 12px !important;
    }
}

/* Card container groups */
.dn-card-group {
    background-color: var(--dn-surface) !important;
    border: 1px solid var(--dn-border) !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
    margin-bottom: 14px !important;
    box-sizing: border-box !important;
    width: 100% !important;
}

.dn-panel-header {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 10px !important;
}
.dn-panel-header * {
    margin: 0 !important;
}

/* Section header with flex wrap */
.dn-section-header {
    display: flex !important;
    flex-direction: row !important;
    justify-content: space-between !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    width: 100% !important;
    margin-bottom: 10px !important;
    background: transparent !important;
    border: none !important;
}

.dn-section-title {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    flex: 1 1 200px !important;
    min-width: 0 !important;
}
.dn-section-title * {
    margin: 0 !important;
    word-break: normal !important;
}

.dn-header-actions {
    display: flex !important;
    flex-direction: row !important;
    gap: 6px !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    justify-content: flex-end !important;
    flex: 0 0 auto !important;
}

.dn-action-btn {
    background-color: var(--dn-border) !important;
    border: 1px solid var(--dn-border-hover) !important;
    color: #e4e4e7 !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 5px 10px !important;
    min-width: 0 !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
}
.dn-action-btn:hover {
    background-color: var(--dn-border-hover) !important;
    color: #ffffff !important;
    border-color: #52525b !important;
}

.dn-clear-btn {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--dn-text-muted) !important;
}
.dn-clear-btn:hover {
    color: var(--dn-danger) !important;
    background-color: rgba(244, 63, 94, 0.1) !important;
    border-color: rgba(244, 63, 94, 0.2) !important;
}

@media (max-width: 520px) {
    .dn-section-header {
        flex-direction: column !important;
        align-items: stretch !important;
    }
    .dn-header-actions {
        width: 100% !important;
        justify-content: stretch !important;
    }
    .dn-header-actions > * {
        flex: 1 1 0% !important;
        text-align: center !important;
    }
}

/* Quick Presets Box & Grid */
.dn-preset-box {
    background-color: #141416 !important;
    border: 1px solid var(--dn-border) !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    margin-bottom: 12px !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

.dn-preset-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 4px;
    font-size: 11px;
    font-weight: 700;
    color: var(--dn-text-muted);
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}

.dn-preset-sub {
    font-size: 10px;
    font-weight: 400;
    color: var(--dn-text-subtle);
}

.dn-preset-grid {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 8px !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

@media (max-width: 550px) {
    .dn-preset-grid {
        grid-template-columns: 1fr !important;
    }
}

.dn-preset-btn {
    width: 100% !important;
    min-width: 0 !important;
    background-color: #27272a !important;
    border: 1px solid #3f3f46 !important;
    color: #d4d4d8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 8px 6px !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
    text-align: center !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.dn-preset-btn:hover {
    background-color: #3f3f46 !important;
    border-color: var(--dn-primary) !important;
    color: #ffffff !important;
}

/* Field Card */
.dn-field-card {
    background-color: var(--dn-surface-card) !important;
    border: 1px solid var(--dn-border) !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    margin-bottom: 8px !important;
    box-sizing: border-box !important;
    width: 100% !important;
    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out !important;
}
.dn-field-card:hover {
    border-color: var(--dn-border-hover) !important;
}
.dn-field-card:focus-within {
    border-color: #52525b !important;
    box-shadow: 0 0 0 1px rgba(249, 115, 22, 0.2) !important;
}

.dn-field-row-top {
    display: flex !important;
    gap: 8px !important;
    width: 100% !important;
    align-items: flex-start !important;
}

.dn-field-row-bottom {
    display: flex !important;
    gap: 8px !important;
    width: 100% !important;
}

@media (max-width: 520px) {
    .dn-field-row-bottom {
        flex-direction: column !important;
    }
}

/* Add Field dashed button */
.dn-add-field-btn {
    border: 1px dashed var(--dn-border-hover) !important;
    background-color: transparent !important;
    color: var(--dn-text-muted) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    width: 100% !important;
    padding: 10px 14px !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
}
.dn-add-field-btn:hover {
    border-color: var(--dn-text-subtle) !important;
    color: var(--dn-text-main) !important;
    background-color: var(--dn-border) !important;
}

/* Delete button */
.dn-del-btn {
    background: transparent !important;
    border: none !important;
    color: var(--dn-text-subtle) !important;
    font-size: 14px !important;
    cursor: pointer !important;
    padding: 6px !important;
    transition: color 0.15s ease-in-out, transform 0.1s ease-in-out !important;
}
.dn-del-btn:hover {
    color: var(--dn-danger) !important;
    transform: scale(1.1) !important;
}

/* Results header */
.dn-results-header {
    display: flex !important;
    flex-direction: row !important;
    justify-content: space-between !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-bottom: 8px !important;
    width: 100% !important;
}

/* Copy button */
.dn-copy-btn {
    background: #27272a !important;
    border: 1px solid #3f3f46 !important;
    color: #e4e4e7 !important;
    padding: 4px 9px !important;
    border-radius: 4px !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    flex-shrink: 0 !important;
}
.dn-copy-btn:hover {
    background: #3f3f46 !important;
    color: #ffffff !important;
    border-color: #52525b !important;
}

/* Secondary toolbar button */
.dn-secondary-btn {
    background-color: var(--dn-surface) !important;
    border: 1px solid var(--dn-border-hover) !important;
    color: #cbd5e1 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    padding: 6px 12px !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
}
.dn-secondary-btn:hover {
    background-color: var(--dn-border) !important;
    border-color: var(--dn-text-subtle) !important;
    color: var(--dn-text-main) !important;
}

.dn-raw-hint {
    font-size: 11px !important;
    color: var(--dn-text-muted) !important;
    margin-bottom: 6px !important;
}

.dn-raw-actions {
    display: flex !important;
    gap: 8px !important;
    width: 100% !important;
    margin-top: 6px !important;
}
@media (max-width: 480px) {
    .dn-raw-actions {
        flex-direction: column !important;
    }
    .dn-raw-actions > * {
        width: 100% !important;
    }
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
    box-shadow: 0 2px 8px rgba(249, 115, 22, 0.25) !important;
    transition: all 0.2s ease-in-out !important;
    width: 100% !important;
    margin-top: 10px !important;
    min-height: 46px !important;
}
.dn-primary-btn:hover {
    background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%) !important;
    box-shadow: 0 4px 16px rgba(234, 88, 12, 0.45) !important;
    transform: translateY(-1px) !important;
}
.dn-primary-btn:active {
    transform: translateY(0) !important;
}

/* File upload container & preview */
.gradio-container .file-preview-holder,
.gradio-container .upload-container,
.gradio-container [data-testid="dropzone"] {
    background-color: var(--dn-surface-card) !important;
    border: 1px dashed var(--dn-border-hover) !important;
    color: var(--dn-text-main) !important;
    border-radius: 8px !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

/* Output Tabs */
.gradio-container .tab-nav {
    border-bottom: 1px solid var(--dn-border) !important;
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
}
.gradio-container .tab-nav button {
    color: var(--dn-text-muted) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-bottom: 2px solid transparent !important;
    white-space: nowrap !important;
}
.gradio-container .tab-nav button.selected {
    color: var(--dn-text-main) !important;
    border-bottom-color: var(--dn-primary) !important;
    font-weight: 600 !important;
}

/* JSON viewer */
.gradio-container .json-holder {
    background-color: var(--dn-bg) !important;
    border: 1px solid var(--dn-border) !important;
    border-radius: 6px !important;
    max-width: 100% !important;
    overflow-x: auto !important;
}

/* Hide Gradio footer completely */
footer, .gradio-container footer, .gradio-container .footer {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Responsive breakpoint rules for mobile (<= 600px) */
@media (max-width: 600px) {
    .gradio-container {
        padding: 6px 8px !important;
    }
    .dn-card-group {
        padding: 10px 10px !important;
        margin-bottom: 10px !important;
    }
    .dn-header-banner {
        padding: 10px 12px !important;
        margin-bottom: 10px !important;
    }
}
"""

HEADER_HTML = """
<style>
/* Direct global layout overrides to eliminate excessive side margins when stacked */
html, body {
    margin: 0 !important;
    padding: 0 !important;
}
@media (max-width: 1100px) {
    .gradio-container .main.app {
        max-width: 100% !important;
        width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .gradio-container .contain,
    .gradio-container .wrap {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    .gradio-container {
        padding: 10px 14px !important;
    }
}
@media (max-width: 600px) {
    .gradio-container {
        padding: 6px 8px !important;
    }
}
.gradio-container .html-container:has(.dn-header-banner) {
    padding: 0 !important;
}
</style>
<div class="dn-header-banner" style="display: flex; align-items: center; justify-content: flex-start; padding: 14px 18px; background: #18181b; border: 1px solid #27272a; border-radius: 8px; margin-bottom: 16px; width: 100%; box-sizing: border-box;">
    <div style="display: flex; align-items: center; gap: 14px; min-width: 0;">
        <svg role="img" aria-label="DocNeedle Logo" viewBox="170 170 684 684" style="width: 36px; height: 36px; flex-shrink: 0;" fill="none" xmlns="http://www.w3.org/2000/svg">
            <title>DocNeedle Logo</title>
            <g>
                <path d="M 727 271 L 418 270 L 294 400 L 297 708 L 320 704 L 328 704 L 304 723 L 294 729 L 294 763 L 297 762 L 343 717 L 399 657 L 469 576 L 639 371 Z" fill="#2E46AA" />
                <path d="M 418 270 L 294 400 L 418 400 Z" fill="#256CD2" />
                <path d="M 728 444 L 671 501 L 552 611 L 464 695 L 385 775 L 330 834 L 317 850 L 341 849 L 349 838 L 367 818 L 368 822 L 363 850 L 729 849 L 729 506 L 723 506 L 693 513 L 688 512 L 728 480 Z" fill="#09BDCE" />
                <path d="M 741 338 L 733 335 L 723 335 L 704 343 L 679 363 L 658 385 L 563 502 L 296 819 L 372 748 L 601 528 L 688 448 L 721 415 L 743 386 L 752 364 L 751 350 L 749 346 Z M 726 367 L 725 378 L 719 389 L 709 402 L 685 425 L 677 430 L 671 432 L 665 432 L 659 425 L 660 416 L 665 406 L 678 389 L 693 374 L 705 365 L 715 361 L 719 361 Z" fill="#FFFFFF" stroke="#0F172A" stroke-width="8" stroke-linejoin="round" />
            </g>
        </svg>
        <div style="min-width: 0;">
            <div style="font-weight: 700; font-size: 18px; color: #f4f4f5; letter-spacing: -0.02em; line-height: 1.2;">DocNeedle</div>
            <div style="font-size: 12px; color: #a1a1aa; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">Document Extraction Microservice</div>
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

DEFAULT_EXAMPLE_SCHEMA: list[dict[str, str]] = [
    {
        "key": "example_field",
        "description": "Describe what you want to extract from the document",
        "type": "string",
    }
]

SCHEMA_PRESETS: dict[str, list[dict[str, str]]] = {
    "invoice": [
        {"key": "invoice_id", "description": "Invoice number or identifier", "type": "id"},
        {"key": "invoice_date", "description": "Date of the invoice", "type": "date"},
        {
            "key": "vendor_name",
            "description": "Name of the supplier or issuing organization",
            "type": "organization",
        },
        {
            "key": "total_amount",
            "description": "Total amount payable including taxes",
            "type": "money",
        },
        {"key": "vat_amount", "description": "Tax or VAT (KDV) amount", "type": "money"},
    ],
    "receipt": [
        {"key": "merchant_name", "description": "Merchant or store name", "type": "organization"},
        {"key": "receipt_date", "description": "Transaction date", "type": "date"},
        {"key": "total_amount", "description": "Total purchase amount", "type": "money"},
        {
            "key": "payment_method",
            "description": "Payment method (Cash, Credit Card, etc.)",
            "type": "string",
        },
    ],
    "id_card": [
        {"key": "full_name", "description": "Full legal name of the person", "type": "person"},
        {
            "key": "id_number",
            "description": "National ID, passport, or identification number",
            "type": "id",
        },
        {"key": "birth_date", "description": "Date of birth", "type": "date"},
        {"key": "expiry_date", "description": "Card or document expiration date", "type": "date"},
    ],
    "contract": [
        {
            "key": "contract_parties",
            "description": "Names of the participating parties",
            "type": "list",
        },
        {
            "key": "effective_date",
            "description": "Effective start date of the agreement",
            "type": "date",
        },
        {
            "key": "contract_value",
            "description": "Total monetary value or consideration",
            "type": "money",
        },
        {"key": "governing_law", "description": "Governing law or jurisdiction", "type": "string"},
    ],
}


def apply_preset_to_form_updates(
    preset_key: str,
) -> tuple[list[bool], list[Any], list[Any], list[Any], list[Any], str]:
    """Retrieve preset schema items and return standard form component updates."""
    items = SCHEMA_PRESETS.get(preset_key, [])
    return apply_items_to_form_updates(items)


def resolve_field_type(raw_type: Any) -> str:
    """Normalize raw field type string against TYPE_MAP and FIELD_TYPE_KEYS."""
    t = str(raw_type).strip().lower() if raw_type else "string"
    if t in ("none", "n/a", ""):
        return "string"
    resolved = TYPE_MAP.get(t, t)
    return resolved if resolved in FIELD_TYPE_KEYS else "string"


def collect_form_items(
    q_vals: list[Any] | tuple[Any, ...],
    k_vals: list[Any] | tuple[Any, ...],
    t_vals: list[Any] | tuple[Any, ...],
    vis_vals: list[bool] | None = None,
) -> list[dict[str, str]]:
    """Collect normalized extraction item dicts from form builder inputs."""
    items: list[dict[str, str]] = []
    num_fields = min(len(q_vals), len(k_vals), len(t_vals))
    for i in range(num_fields):
        q = str(q_vals[i]).strip() if q_vals[i] else ""
        k = str(k_vals[i]).strip() if k_vals[i] else ""
        is_vis = vis_vals[i] if vis_vals is not None and i < len(vis_vals) else True
        if not is_vis and not (q or k):
            continue
        if q or k:
            items.append(
                {
                    "key": k or q or f"item_{len(items) + 1}",
                    "description": q or k,
                    "type": resolve_field_type(t_vals[i]),
                }
            )
    return items


def unpack_field_args(
    field_args: tuple[Any, ...],
    num_fields: int = MAX_FIELDS,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    """Unpack flattened Gradio form arguments into (questions, keys, types) slices."""
    q_vals = field_args[0:num_fields]
    k_vals = field_args[num_fields : 2 * num_fields]
    t_vals = field_args[2 * num_fields : 3 * num_fields]
    return q_vals, k_vals, t_vals


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
        resolved_t = resolve_field_type(item.get("type", "string"))

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
    q_vals: list[Any] | tuple[Any, ...],
    k_vals: list[Any] | tuple[Any, ...],
    t_vals: list[Any] | tuple[Any, ...],
    vis_vals: list[bool] | None = None,
) -> str:
    """Convert form fields to formatted JSON string."""
    items = collect_form_to_items_or_default(q_vals, k_vals, t_vals, vis_vals)
    return json.dumps(items, indent=2, ensure_ascii=False)


def collect_form_to_items_or_default(
    q_vals: list[Any] | tuple[Any, ...],
    k_vals: list[Any] | tuple[Any, ...],
    t_vals: list[Any] | tuple[Any, ...],
    vis_vals: list[bool] | None = None,
) -> list[dict[str, str]]:
    """Collect form items or return the default example schema when empty."""
    items = collect_form_items(q_vals, k_vals, t_vals, vis_vals)
    return items if items else list(DEFAULT_EXAMPLE_SCHEMA)


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
        items if items else DEFAULT_EXAMPLE_SCHEMA,
        indent=2,
        ensure_ascii=False,
    )

    return new_vis, group_updates, q_updates, k_updates, t_updates, json_str


def render_results_html(response_dict: dict[str, Any] | None) -> str:
    """Render extracted results matching frontend card design with enhanced UX."""
    if not response_dict:
        return """
        <div style="border: 1px dashed #3f3f46; border-radius: 12px; padding: 48px 24px; text-align: center; color: #a1a1aa; background: #141416;">
            <div style="width: 52px; height: 52px; margin: 0 auto 16px; border-radius: 50%; background: rgba(37, 108, 210, 0.12); display: flex; align-items: center; justify-content: center; border: 1px solid rgba(37, 108, 210, 0.25);">
                <svg style="width: 26px; height: 26px; color: #60a5fa;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                </svg>
            </div>
            <div style="font-weight: 700; font-size: 16px; color: #f4f4f5; letter-spacing: -0.01em;">Ready for Document Extraction</div>
            <div style="font-size: 13px; margin: 6px auto 20px; color: #a1a1aa; max-width: 440px; line-height: 1.5;">
                Upload your documents on the left and define target fields using Quick Presets or the custom Form Builder.
            </div>
            <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: #1c1c20; border: 1px solid #2e2e33; color: #d4d4d8; font-size: 11px; padding: 5px 10px; border-radius: 6px;">📄 PDF / Scanned</span>
                <span style="background: #1c1c20; border: 1px solid #2e2e33; color: #d4d4d8; font-size: 11px; padding: 5px 10px; border-radius: 6px;">📝 DOCX / Word</span>
                <span style="background: #1c1c20; border: 1px solid #2e2e33; color: #d4d4d8; font-size: 11px; padding: 5px 10px; border-radius: 6px;">📊 XLSX / Excel</span>
                <span style="background: #1c1c20; border: 1px solid #2e2e33; color: #d4d4d8; font-size: 11px; padding: 5px 10px; border-radius: 6px;">🖼️ Images (PNG, JPG, WebP)</span>
            </div>
            <div style="border-top: 1px solid #27272a; padding-top: 14px; font-size: 11px; color: #71717a; display: flex; align-items: center; justify-content: center; gap: 6px;">
                <span>🔒</span>
                <span>Zero disk persistence · Pure in-memory vector store · Privacy guaranteed</span>
            </div>
        </div>
        """

    if "error" in response_dict:
        err_msg = html.escape(str(response_dict["error"]))
        return f"""
        <div role="alert" style="border: 1px solid #991b1b; background: rgba(153, 27, 27, 0.15); color: #fecdd3; border-radius: 10px; padding: 18px; font-size: 13px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 18px;">⚠️</span>
                <strong style="color: #fda4af; font-size: 14px;">Extraction Notice</strong>
            </div>
            <div style="color: #fecdd3; line-height: 1.5; margin-bottom: 12px; padding: 8px 12px; background: rgba(0, 0, 0, 0.3); border-radius: 6px; border-left: 3px solid #f43f5e; font-family: monospace; font-size: 12px;">
                {err_msg}
            </div>
            <div style="font-size: 12px; color: #cbd5e1; background: #141416; border: 1px solid #27272a; border-radius: 6px; padding: 10px 12px;">
                <span style="font-weight: 600; color: #f4f4f5; display: block; margin-bottom: 4px;">Troubleshooting Tips:</span>
                <ul style="margin: 0; padding-left: 18px; line-height: 1.6; color: #a1a1aa;">
                    <li>Verify you have uploaded at least one supported document (PDF, DOCX, XLSX, Images).</li>
                    <li>Ensure at least one extraction field has a valid question or target key defined.</li>
                    <li>You can click one of the <strong>Quick Presets</strong> on the left to quickly populate standard fields.</li>
                    <li>Check that the total upload size does not exceed the allowed limit (50 MB).</li>
                </ul>
            </div>
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
            '<span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 3px 8px; border-radius: 5px; font-size: 11px; font-weight: 600;">✓ Found</span>'
            if is_found
            else '<span style="background: #27272a; color: #a1a1aa; border: 1px solid #3f3f46; padding: 3px 8px; border-radius: 5px; font-size: 11px; font-weight: 600;">✕ Not Found</span>'
        )

        if is_found:
            conf_color = "#10b981" if conf >= 80 else ("#f59e0b" if conf >= 50 else "#f43f5e")
            conf_label = "High" if conf >= 80 else ("Medium" if conf >= 50 else "Low")
            conf_html = f"""
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 11px; color: #a1a1aa;">{conf_label}:</span>
                <div style="width: 50px; height: 6px; background: #27272a; border-radius: 3px; overflow: hidden; border: 1px solid #3f3f46;">
                    <div style="width: {conf}%; height: 100%; background: {conf_color};"></div>
                </div>
                <span style="font-family: monospace; font-size: 11px; font-weight: 600; color: {conf_color};">%{conf}</span>
            </div>
            """
        else:
            conf_html = ""

        evidence_html = ""
        if evidence:
            doc_name = html.escape(str(evidence.get("document", "")))
            page_num = evidence.get("page")
            raw_snippet = evidence.get("snippet", "")
            snippet = html.escape(str(raw_snippet)) if raw_snippet else ""
            page_badge = (
                f'<span style="background: #27272a; border: 1px solid #3f3f46; color: #d4d4d8; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 600;">• Page {int(page_num)}</span>'
                if isinstance(page_num, int)
                else ""
            )

            snippet_box = (
                f'<div style="margin-top: 6px; padding: 8px 12px; background: #09090b; border-left: 3px solid #256cd2; border-radius: 0 6px 6px 0; font-size: 12px; color: #cbd5e1; font-style: italic; line-height: 1.4;">"{snippet}"</div>'
                if snippet
                else ""
            )

            evidence_html = f"""
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #27272a; font-size: 12px; color: #a1a1aa;">
                <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <span style="font-weight: 600; color: #f4f4f5;">Source Document:</span>
                    <span style="background: #1c1c20; border: 1px solid #2e2e33; color: #e4e4e7; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-family: monospace;">📄 {doc_name}</span>
                    {page_badge}
                </div>
                {snippet_box}
            </div>
            """

        thinking_html = ""
        if thinking:
            safe_thinking = html.escape(str(thinking))
            thinking_html = f"""
            <details style="margin-top: 10px; font-size: 12px; color: #a1a1aa;">
                <summary style="cursor: pointer; font-weight: 500; color: #94a3b8; user-select: none; padding: 4px 0;">🧠 Model Reasoning & Grounding Steps</summary>
                <div style="margin-top: 6px; padding: 10px 12px; background: #09090b; border: 1px solid #27272a; border-radius: 6px; font-family: monospace; white-space: pre-wrap; color: #e2e8f0; font-size: 11px; line-height: 1.5;">{safe_thinking}</div>
            </details>
            """

        # Copy button for found values
        copy_btn_html = ""
        if is_found and raw_val is not None and str(raw_val).strip():
            safe_copy_val = html.escape(str(raw_val).replace('"', "&quot;"))
            copy_btn_html = f"""
            <button type="button" class="dn-copy-btn" data-clipboard="{safe_copy_val}" onclick="navigator.clipboard.writeText(this.getAttribute('data-clipboard')); const orig = this.innerHTML; this.innerHTML='✓ Copied'; setTimeout(() => this.innerHTML=orig, 1500);" title="Copy value to clipboard">📋 Copy</button>
            """

        val_style = (
            "color: #f4f4f5;"
            if is_found
            else "color: #71717a; font-style: italic; font-weight: normal;"
        )

        cards_html += f"""
        <div style="background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 14px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); width: 100%; box-sizing: border-box;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">
                <div style="display: flex; align-items: center; gap: 8px; min-width: 0; flex-wrap: wrap;">
                    <span style="font-family: monospace; font-size: 13px; font-weight: 700; color: #f4f4f5; background: #141416; padding: 2px 6px; border-radius: 4px; border: 1px solid #27272a; word-break: break-all;">{key}</span>
                    {badge_html}
                </div>
                {conf_html}
            </div>
            <div class="dn-value-box" style="background: #09090b; border: 1px solid #27272a; border-radius: 6px; padding: 10px 12px; font-size: 13px; font-weight: 600; {val_style} word-break: break-word; overflow-wrap: anywhere; display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; width: 100%; box-sizing: border-box;">
                <div style="flex: 1; min-width: 0; word-break: break-word; overflow-wrap: anywhere;">{val}</div>
                {copy_btn_html}
            </div>
            {evidence_html}
            {thinking_html}
        </div>
        """

    if found == total and total > 0:
        completion_badge = '<span style="background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.3);">All Items Found</span>'
    elif found > 0:
        completion_badge = '<span style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; border: 1px solid rgba(245, 158, 11, 0.3);">Partially Found</span>'
    else:
        completion_badge = '<span style="background: #27272a; color: #a1a1aa; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; border: 1px solid #3f3f46;">No Items Found</span>'

    summary_html = f"""
    <div class="dn-summary-box" style="background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.3); flex-wrap: wrap; gap: 12px; width: 100%; box-sizing: border-box;">
        <div class="dn-summary-stats" style="display: flex; gap: 24px; align-items: center; flex-wrap: wrap;">
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
            {completion_badge}
        </div>
    </div>
    """

    return summary_html + cards_html


def make_status_html(
    message: str,
    is_success: bool = False,
    is_loading: bool = False,
    is_error: bool = False,
) -> str:
    """Format inline status HTML notification with WCAG live region support."""
    border_color = (
        "#10b981"
        if is_success
        else ("#f43f5e" if is_error else ("#f97316" if is_loading else "#27272a"))
    )
    bg_color = (
        "rgba(16, 185, 129, 0.08)"
        if is_success
        else (
            "rgba(244, 63, 94, 0.08)"
            if is_error
            else ("rgba(249, 115, 22, 0.08)" if is_loading else "#141416")
        )
    )
    text_color = (
        "#34d399"
        if is_success
        else ("#fb7185" if is_error else ("#fb923c" if is_loading else "#a1a1aa"))
    )
    icon = "✅" if is_success else ("⚠️" if is_error else ("⏳" if is_loading else "📄"))
    return f"""
    <div role="status" aria-live="polite" style="padding: 10px 14px; background: {bg_color}; border: 1px solid {border_color}; border-radius: 6px; font-size: 12px; color: {text_color}; margin-top: 10px; display: flex; align-items: center; gap: 8px;">
        <span>{icon}</span>
        <span>{message}</span>
    </div>
    """


def _format_completion_status(found: int, total: int, total_time_ms: float) -> str:
    """Format extraction completion status message with elapsed time."""
    time_s = f"{total_time_ms / 1000:.2f} s"
    return (
        f"Extraction complete: {found} of {total} items found in {time_s} ({total_time_ms:.0f} ms)."
    )


async def _async_perform_extraction_ui(
    files: Any,
    session_chunks: list[Any] | None,
    active_tab: str,
    raw_json: str,
    vis: list[bool],
    *field_args: Any,
    start_time: float | None = None,
    progress: Any = None,
) -> tuple[str, dict[str, Any], list[Any], str]:
    """Execute complete document extraction pipeline from dynamic Gradio UI fields or raw JSON schema.

    Uses initial OCR/ingested chunks from session state to avoid redundant OCR runs on queries.
    """
    import time

    if start_time is None:
        start_time = time.perf_counter()

    if progress is not None and callable(progress):
        progress(0.05, desc="🔍 Validating input documents & extraction schema...")

    if not files and not session_chunks:
        err = {"error": "Please upload at least one document (PDF, DOCX, XLSX, or Image)."}
        return (
            render_results_html(err),
            err,
            [],
            make_status_html(
                "Please upload at least one document to start.", is_success=False, is_error=True
            ),
        )

    file_list = files if isinstance(files, list) else ([files] if files else [])
    if len(file_list) == 0 and not session_chunks:
        err = {"error": "Please upload at least one document."}
        return (
            render_results_html(err),
            err,
            [],
            make_status_html(
                "Please upload at least one document.", is_success=False, is_error=True
            ),
        )

    from docneedle.extraction.schemas import ExtractionItemQuery
    from docneedle.extraction.service import ExtractionService
    from docneedle.ingest.service import IngestionService
    from docneedle.retrieval.store import SessionVectorStore

    use_raw = active_tab == "raw"
    q_vals, k_vals, t_vals = unpack_field_args(field_args)
    form_queries = [
        ExtractionItemQuery(key=it["key"], description=it["description"], type=it["type"])
        for it in collect_form_items(q_vals, k_vals, t_vals, vis)
    ]

    raw_queries: list[ExtractionItemQuery] = []
    if raw_json and raw_json.strip():
        try:
            parsed_items = parse_raw_json_schema(raw_json)
            default_ex = DEFAULT_EXAMPLE_SCHEMA[0]
            for it in parsed_items:
                if (
                    it["key"] == default_ex["key"]
                    and it["description"] == default_ex["description"]
                ):
                    continue
                raw_queries.append(
                    ExtractionItemQuery(
                        key=it["key"],
                        description=it["description"],
                        type=it["type"],
                    )
                )
        except Exception as e:
            if use_raw:
                err = {"error": f"Invalid JSON schema: {e}"}
                return (
                    render_results_html(err),
                    err,
                    list(session_chunks) if session_chunks else [],
                    make_status_html(f"Schema Error: {e}", is_success=False, is_error=True),
                )

    queries = raw_queries if (use_raw and raw_queries) else (form_queries or raw_queries)

    if not queries:
        err = {"error": "Please define at least one field to extract via Form Builder or Raw JSON."}
        return (
            render_results_html(err),
            err,
            list(session_chunks) if session_chunks else [],
            make_status_html(
                "Please define at least one field to extract.", is_success=False, is_error=True
            ),
        )

    if len(queries) > settings.max_schema_items:
        err = {
            "error": (
                f"Schema contains {len(queries)} items, exceeding maximum allowed limit of "
                f"{settings.max_schema_items} items."
            )
        }
        return (
            render_results_html(err),
            err,
            list(session_chunks) if session_chunks else [],
            make_status_html(
                f"Limit Exceeded: Maximum {settings.max_schema_items} fields allowed per request.",
                is_success=False,
                is_error=True,
            ),
        )

    if len(file_list) > settings.max_upload_files:
        err = {
            "error": (
                f"Batch contains {len(file_list)} files, exceeding maximum allowed limit of "
                f"{settings.max_upload_files} files."
            )
        }
        return (
            render_results_html(err),
            err,
            list(session_chunks) if session_chunks else [],
            make_status_html(
                f"Limit Exceeded: Maximum {settings.max_upload_files} files allowed per request.",
                is_success=False,
                is_error=True,
            ),
        )

    max_file_bytes = settings.max_upload_size_mb * 1024 * 1024
    max_total_bytes = settings.max_total_upload_size_mb * 1024 * 1024
    total_bytes = 0
    for f in file_list:
        fpath = Path(f.name if hasattr(f, "name") else str(f))
        if fpath.exists():
            sz = fpath.stat().st_size
            total_bytes += sz
            if max_file_bytes > 0 and sz > max_file_bytes:
                err = {
                    "error": (
                        f"File '{fpath.name}' exceeds maximum allowed upload size of "
                        f"{settings.max_upload_size_mb} MB."
                    )
                }
                return (
                    render_results_html(err),
                    err,
                    list(session_chunks) if session_chunks else [],
                    make_status_html(
                        f"File size exceeded: {fpath.name} > {settings.max_upload_size_mb} MB",
                        is_success=False,
                        is_error=True,
                    ),
                )
    if max_total_bytes > 0 and total_bytes > max_total_bytes:
        err = {
            "error": (
                f"Total upload payload size ({total_bytes / (1024 * 1024):.1f} MB) exceeds "
                f"maximum allowed limit of {settings.max_total_upload_size_mb} MB."
            )
        }
        return (
            render_results_html(err),
            err,
            list(session_chunks) if session_chunks else [],
            make_status_html(
                f"Total payload size exceeded: {total_bytes / (1024 * 1024):.1f} MB > "
                f"{settings.max_total_upload_size_mb} MB",
                is_success=False,
                is_error=True,
            ),
        )

    try:
        from docneedle.cuda_compat import preload_cuda_runtime_libs

        preload_cuda_runtime_libs()
    except Exception:
        pass

    ingest_svc = IngestionService()
    extraction_svc = ExtractionService()
    store = SessionVectorStore()

    try:
        if progress is not None and callable(progress):
            progress(0.18, desc="📄 Parsing & inspecting document pages...")

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
                    "Could not extract readable text from documents.",
                    is_success=False,
                    is_error=True,
                ),
            )

        if progress is not None and callable(progress):
            progress(
                0.42,
                desc=f"⚡ Indexing {len(chunks_to_use)} text chunk(s) in in-memory vector store...",
            )

        logger.info(
            f"--> [PIPELINE] Indexing {len(chunks_to_use)} text chunks across {len(file_list)} files into in-memory vector store..."
        )
        store.add_chunks(chunks_to_use)

        if progress is not None and callable(progress):
            progress(
                0.68,
                desc=f"🧠 Extracting {len(queries)} field(s) with Gemma-4 Multimodal LLM...",
            )

        logger.info(
            f"--> [PIPELINE] Starting extraction pipeline for {len(queries)} target field query(s)..."
        )
        response = await extraction_svc.extract_items(queries, store)

        if progress is not None and callable(progress):
            progress(0.92, desc="✅ Verifying citations & calculating confidence scores...")

        resp_dict = response.model_dump()
        total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        resp_dict["processing_time_ms"] = total_time_ms
        status_msg = _format_completion_status(
            resp_dict.get("found_items", 0),
            resp_dict.get("total_items", 0),
            total_time_ms,
        )

        if progress is not None and callable(progress):
            progress(1.0, desc="Extraction completed")

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
            make_status_html(f"Error: {e}", is_success=False, is_error=True),
        )
    finally:
        store.clear()
        import gc

        gc.collect()
        if "torch" in sys.modules:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def _estimate_extraction_duration(
    files: Any = None,
    session_chunks: list[Any] | None = None,
    active_tab: str = "",
    raw_json: str = "",
    vis: list[bool] | None = None,
    progress: Any = None,
    *field_args: Any,
    **kwargs: Any,
) -> int:
    """Dynamically estimate ZeroGPU reservation duration based on schema field count."""
    try:
        if progress is not None and not isinstance(progress, gr.Progress):
            field_args = (progress,) + field_args

        num_items = 1
        if active_tab == "raw" and raw_json and raw_json.strip():
            try:
                parsed = json.loads(raw_json)
                if isinstance(parsed, list):
                    num_items = max(1, len(parsed))
            except Exception:
                num_items = 1
        elif field_args and vis:
            try:
                q_vals, k_vals, t_vals = unpack_field_args(field_args)
                items = collect_form_items(q_vals, k_vals, t_vals, vis)
                if items:
                    num_items = max(1, len(items))
                else:
                    num_items = max(1, sum(1 for v in vis if v))
            except Exception:
                num_items = max(1, sum(1 for v in vis if v))
        elif vis and isinstance(vis, list):
            num_items = max(1, sum(1 for v in vis if v))

        # Base duration covers initial multimodal prefill and model warmup (~67-75s)
        # RAG retrieval context size is bounded (max 2 images, 5 text chunks),
        # so LLM inference workload is independent of the number of uploaded files.
        base_duration = 75
        item_overhead = max(0, num_items - 1) * 3

        estimated = base_duration + item_overhead
        logger.info(
            "--> [ZeroGPU Duration] Estimated reservation: %ds (items=%d)",
            estimated,
            num_items,
        )
        return min(95, max(60, estimated))
    except Exception as e:
        logger.debug("Fallback to default ZeroGPU duration due to estimation error: %s", e)
        return 75


@gpu_decorator(duration=_estimate_extraction_duration)
def perform_extraction_ui(
    files: Any,
    session_chunks: list[Any] | None,
    active_tab: str,
    raw_json: str,
    vis: list[bool],
    progress: gr.Progress = gr.Progress(track_tqdm=True),
    *field_args: Any,
) -> tuple[str, dict[str, Any], list[Any], str]:
    """Execute complete document extraction pipeline from dynamic Gradio UI fields or raw JSON schema."""
    import asyncio
    import concurrent.futures
    import time

    # Support callers that pass positional field_args directly without a progress tracker
    if not isinstance(progress, gr.Progress):
        field_args = (progress,) + field_args
        progress = None

    start_time = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results_html, resp_dict, chunks_to_use, status_html = pool.submit(
                asyncio.run,
                _async_perform_extraction_ui(
                    files,
                    session_chunks,
                    active_tab,
                    raw_json,
                    vis,
                    *field_args,
                    start_time=start_time,
                    progress=progress,
                ),
            ).result()
    else:
        results_html, resp_dict, chunks_to_use, status_html = asyncio.run(
            _async_perform_extraction_ui(
                files,
                session_chunks,
                active_tab,
                raw_json,
                vis,
                *field_args,
                start_time=start_time,
                progress=progress,
            )
        )

    total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    if isinstance(resp_dict, dict) and "results" in resp_dict:
        resp_dict["processing_time_ms"] = total_time_ms
        results_html = render_results_html(resp_dict)
        status_msg = _format_completion_status(
            resp_dict.get("found_items", 0),
            resp_dict.get("total_items", 0),
            total_time_ms,
        )
        status_html = make_status_html(status_msg, is_success=True)

    return results_html, resp_dict, chunks_to_use, status_html


def check_system_health() -> dict[str, Any]:
    from docneedle.main import get_system_health

    return get_system_health().model_dump()


# 2. Build interactive Gradio Dashboard (dark mode & English)
with gr.Blocks(title="DocNeedle - Document Extraction") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(elem_classes=["dn-main-layout"]):
        # 1. Column: File Upload / Documents (Dosya Yükleme)
        with gr.Column(scale=3, elem_classes=["dn-col-docs"]):
            with gr.Group(elem_classes=["dn-card-group"]):
                gr.Markdown(
                    "### 📁 DOCUMENTS\n*Upload PDFs, scanned files, spreadsheets, or images*",
                    elem_classes=["dn-panel-header"],
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

                gr.HTML(
                    """
                    <div style="margin-top: 10px; padding: 10px 12px; background: rgba(0,0,0,0.25); border: 1px solid #27272a; border-radius: 6px; font-size: 11px; color: #a1a1aa; line-height: 1.5;">
                        <div style="font-weight: 600; color: #e4e4e7; margin-bottom: 4px;">📄 Supported Formats</div>
                        <div style="display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 8px;">
                            <span style="background: #27272a; padding: 2px 6px; border-radius: 4px; color: #d4d4d8;">PDF (Scanned/OCR-Free)</span>
                            <span style="background: #27272a; padding: 2px 6px; border-radius: 4px; color: #d4d4d8;">DOCX / XLSX</span>
                            <span style="background: #27272a; padding: 2px 6px; border-radius: 4px; color: #d4d4d8;">PNG / JPG / WEBP</span>
                        </div>
                        <div style="font-size: 10px; color: #71717a; border-top: 1px solid #27272a; padding-top: 6px;">
                            🔒 <strong>Privacy First:</strong> RAM-only vector store. Zero disk persistence.
                        </div>
                    </div>
                    """
                )

        # 2. Column: Field Inputs & Presets (Alan Girdileri)
        with gr.Column(scale=4, elem_classes=["dn-col-fields"]):
            with gr.Group(elem_classes=["dn-card-group"]):
                with gr.Row(elem_classes=["dn-section-header"]):
                    gr.Markdown(
                        "### 🎯 FIELDS TO EXTRACT\n*Define target fields via presets or form*",
                        elem_classes=["dn-section-title"],
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

                # Quick Presets Box (Fluid Grid & Full-Width Header)
                with gr.Group(elem_classes=["dn-preset-box"]):
                    gr.HTML(
                        '<div class="dn-preset-heading"><span>⚡ QUICK PRESETS</span><span class="dn-preset-sub">1-Click Templates</span></div>'
                    )
                    with gr.Row(elem_classes=["dn-preset-grid"]):
                        preset_invoice_btn = gr.Button(
                            "🧾 Invoice / Fatura", elem_classes=["dn-preset-btn"], size="sm"
                        )
                        preset_receipt_btn = gr.Button(
                            "🧾 Receipt / Fiş", elem_classes=["dn-preset-btn"], size="sm"
                        )
                        preset_id_btn = gr.Button(
                            "🆔 ID / Kimlik", elem_classes=["dn-preset-btn"], size="sm"
                        )
                        preset_contract_btn = gr.Button(
                            "📜 Contract / Sözleşme", elem_classes=["dn-preset-btn"], size="sm"
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
                                with gr.Row(elem_classes=["dn-field-row-top"]):
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
                                with gr.Row(elem_classes=["dn-field-row-bottom"]):
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
                        with gr.Row(elem_classes=["dn-raw-actions"]):
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
                    "🚀 Extract Data",
                    variant="primary",
                    elem_classes=["dn-primary-btn"],
                )

        # 3. Column: Results & Provenance (Sonuç)
        with gr.Column(scale=5, elem_classes=["dn-col-results"]):
            with gr.Group(elem_classes=["dn-card-group"]):
                with gr.Row(elem_classes=["dn-results-header"]):
                    gr.Markdown("### 📊 EXTRACTION RESULTS", elem_classes=["dn-section-title"])
                    with gr.Row(elem_classes=["dn-header-actions"]):
                        export_results_btn = gr.DownloadButton(
                            "📥 Export JSON",
                            elem_classes=["dn-action-btn"],
                            size="sm",
                            visible=False,
                        )

                last_response_state = gr.State({})

                with gr.Tabs(elem_classes=["dn-output-tabs"]):
                    with gr.Tab("📋 Formatted View", id="tab_formatted"):
                        results_html_view = gr.HTML(value=render_results_html(None))

                    with gr.Tab("📊 Raw JSON View", id="tab_raw"):
                        raw_json_output = gr.JSON(label="Complete JSON Extraction Response")

                with gr.Accordion("⚙️ System Health & API Info", open=False):
                    health_btn = gr.Button("Check System Health", size="sm")
                    health_json = gr.JSON(label="System Details")
                    health_btn.click(
                        check_system_health, outputs=health_json, api_name=False, queue=False
                    )
                    gr.Markdown(
                        """
                        * **Swagger API Documentation:** [/docs](/docs)
                        * **Extraction Endpoint:** `POST /v1/extract`
                        * **Health Endpoint:** `GET /healthz`
                        """
                    )

    # Event handlers for dynamic schema builder (all internal UI events with api_name=False)
    def on_add_field(vis: list[bool], *field_args: Any) -> list[Any]:
        new_vis = list(vis)
        q_vals = field_args[0:MAX_FIELDS]
        k_vals = field_args[MAX_FIELDS : 2 * MAX_FIELDS]
        for idx in range(len(new_vis)):
            q = str(q_vals[idx]).strip() if idx < len(q_vals) and q_vals[idx] else ""
            k = str(k_vals[idx]).strip() if idx < len(k_vals) and k_vals[idx] else ""
            if q or k:
                new_vis[idx] = True
        for idx in range(len(new_vis)):
            if not new_vis[idx]:
                new_vis[idx] = True
                break
        return [new_vis] + [gr.update(visible=v) for v in new_vis]

    add_field_btn.click(
        on_add_field,
        inputs=[vis_state] + q_inputs + k_inputs,
        outputs=[vis_state] + field_groups,
        api_name=False,
        queue=False,
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
            queue=False,
        )

    def on_clear_all() -> list[Any]:
        new_vis, grp_ups, q_ups, k_ups, t_ups, empty_json = apply_items_to_form_updates([])
        return [new_vis] + grp_ups + q_ups + k_ups + t_ups + [empty_json]

    clear_all_btn.click(
        on_clear_all,
        outputs=[vis_state] + field_groups + q_inputs + k_inputs + t_inputs + [raw_json_input],
        api_name=False,
        queue=False,
    )

    def make_preset_handler(preset_key: str):
        def on_apply_preset() -> list[Any]:
            new_vis, grp_ups, q_ups, k_ups, t_ups, json_str = apply_preset_to_form_updates(
                preset_key
            )
            count = len(SCHEMA_PRESETS.get(preset_key, []))
            gr.Info(f"Loaded '{preset_key.capitalize()}' preset with {count} fields.")
            return [new_vis] + grp_ups + q_ups + k_ups + t_ups + [json_str]

        return on_apply_preset

    preset_invoice_btn.click(
        make_preset_handler("invoice"),
        outputs=[vis_state] + field_groups + q_inputs + k_inputs + t_inputs + [raw_json_input],
        api_name=False,
        queue=False,
    )
    preset_receipt_btn.click(
        make_preset_handler("receipt"),
        outputs=[vis_state] + field_groups + q_inputs + k_inputs + t_inputs + [raw_json_input],
        api_name=False,
        queue=False,
    )
    preset_id_btn.click(
        make_preset_handler("id_card"),
        outputs=[vis_state] + field_groups + q_inputs + k_inputs + t_inputs + [raw_json_input],
        api_name=False,
        queue=False,
    )
    preset_contract_btn.click(
        make_preset_handler("contract"),
        outputs=[vis_state] + field_groups + q_inputs + k_inputs + t_inputs + [raw_json_input],
        api_name=False,
        queue=False,
    )

    def on_sync_form_to_raw(vis: list[bool], *field_args: Any) -> str:
        q_vals, k_vals, t_vals = unpack_field_args(field_args)
        return serialize_form_to_json(q_vals, k_vals, t_vals, vis)

    sync_form_to_raw_btn.click(
        on_sync_form_to_raw,
        inputs=[vis_state] + q_inputs + k_inputs + t_inputs,
        outputs=[raw_json_input],
        api_name=False,
        queue=False,
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
        queue=False,
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
        queue=False,
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
            q_vals, k_vals, t_vals = unpack_field_args(field_args)
            items = collect_form_to_items_or_default(q_vals, k_vals, t_vals, vis)

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
        queue=False,
    )

    tab_form_input.select(lambda: "form", outputs=[active_tab_state], api_name=False, queue=False)

    def on_select_raw_tab(vis: list[bool], *field_args: Any) -> tuple[str, str]:
        q_vals, k_vals, t_vals = unpack_field_args(field_args)
        json_str = serialize_form_to_json(q_vals, k_vals, t_vals, vis)
        return "raw", json_str

    tab_raw_input.select(
        on_select_raw_tab,
        inputs=[vis_state] + q_inputs + k_inputs + t_inputs,
        outputs=[active_tab_state, raw_json_input],
        api_name=False,
        queue=False,
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
        queue=False,
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

    def on_export_results(resp_dict: dict[str, Any] | None) -> str | None:
        if not resp_dict or not resp_dict.get("results"):
            gr.Warning("No extracted results to export.")
            return None
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="docneedle_results_",
            delete=False,
            encoding="utf-8",
        )
        json.dump(resp_dict, tmp, indent=2, ensure_ascii=False)
        tmp.close()
        return tmp.name

    export_results_btn.click(
        on_export_results,
        inputs=[last_response_state],
        outputs=[export_results_btn],
        api_name=False,
        queue=False,
    )

    def on_results_changed(resp_dict: Any) -> tuple[Any, Any]:
        has_results = isinstance(resp_dict, dict) and bool(resp_dict.get("results"))
        return resp_dict, gr.update(visible=has_results)

    raw_json_output.change(
        on_results_changed,
        inputs=[raw_json_output],
        outputs=[last_response_state, export_results_btn],
        api_name=False,
        queue=False,
    )

    # ZeroGPU keepalive and compatibility handler
    try:

        @gpu_decorator(duration=60)
        def _zerogpu_compat_probe() -> str:
            return ""

        _zg_btn = gr.Button("ZeroGPU Init", visible=False)
        _zg_btn.click(_zerogpu_compat_probe, outputs=[], api_name=False)
    except Exception:
        pass

# 3. Mount complete FastAPI REST API onto Gradio App
from gradio.routes import App  # noqa: E402

from docneedle.api.routes_extract import router as extract_router  # noqa: E402
from docneedle.api.routes_jobs import router as jobs_router  # noqa: E402
from docneedle.main import health_check  # noqa: E402

gradio_app = App(
    title=settings.app_name,
    version=settings.app_version,
    description="Privacy-first local RAG document extraction microservice",
    docs_url="/docs",
    redoc_url="/redoc",
)
gradio_app.include_router(extract_router)
gradio_app.include_router(jobs_router)
gradio_app.add_api_route("/healthz", health_check, methods=["GET"])
gradio_app.add_api_route("/health", health_check, methods=["GET"])

if __name__ == "__main__":
    app = gradio_app
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"--> [STARTUP] Starting DocNeedle on 0.0.0.0:{port}...")
    demo.launch(
        _app=gradio_app,
        server_name="0.0.0.0",
        server_port=port,
        footer_links=[],
        css=CUSTOM_CSS,
        ssr_mode=False,
    )
else:
    # Mount Gradio dashboard onto root path for external ASGI runners (e.g. uvicorn app:app)
    app = gr.mount_gradio_app(
        gradio_app,
        demo,
        path="/",
        footer_links=[],
        css=CUSTOM_CSS,
        ssr_mode=False,
    )
