"""AcroForm interactive form field extraction and formatting from PDF documents."""

import ctypes
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _extract_acroform_fields_by_page(doc: Any) -> dict[int, dict[str, str]]:
    """Extract interactive AcroForm field key-value pairs per page if present in PDF."""
    import pypdfium2.raw as pdfium_c

    try:
        if doc.get_formtype() == pdfium_c.FORMTYPE_NONE:
            return {}
    except Exception:
        return {}

    try:
        doc.init_forms()
    except Exception as e:
        logger.debug(f"Failed to initialize forms in PDF: {e}")
        return {}

    form_env = doc.formenv.raw if doc.formenv else None
    if not form_env:
        return {}

    def _read_wide_string(func: Any, annot: Any) -> str:
        try:
            n_bytes = func(form_env, annot, None, 0)
            if n_bytes <= 2:
                return ""
            buf = ctypes.create_string_buffer(n_bytes)
            ptr = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ushort))
            written = func(form_env, annot, ptr, n_bytes)
            if written <= 2:
                return ""
            safe_len = (written - 2) & ~1
            return bytes(buf)[:safe_len].decode("utf-16-le", errors="replace").strip()
        except Exception:
            return ""

    per_page: dict[int, dict[str, str]] = {}
    try:
        for p_idx in range(len(doc)):
            page_num = p_idx + 1
            page = doc[p_idx]
            try:
                annot_count = pdfium_c.FPDFPage_GetAnnotCount(page.raw)
                for a_idx in range(annot_count):
                    annot = pdfium_c.FPDFPage_GetAnnot(page.raw, a_idx)
                    try:
                        f_type = pdfium_c.FPDFAnnot_GetFormFieldType(form_env, annot)
                        if f_type == pdfium_c.FPDF_FORMFIELD_UNKNOWN:
                            continue
                        name = _read_wide_string(pdfium_c.FPDFAnnot_GetFormFieldName, annot)
                        alt = _read_wide_string(pdfium_c.FPDFAnnot_GetFormFieldAlternateName, annot)
                        val = _read_wide_string(pdfium_c.FPDFAnnot_GetFormFieldValue, annot)

                        if f_type in (
                            pdfium_c.FPDF_FORMFIELD_CHECKBOX,
                            pdfium_c.FPDF_FORMFIELD_RADIOBUTTON,
                        ):
                            is_checked = bool(pdfium_c.FPDFAnnot_IsChecked(form_env, annot))
                            if is_checked:
                                val = "Yes"
                            elif val in ("Off", "", "0", "None"):
                                val = ""

                        if alt and name and alt.lower() != name.lower():
                            field_key = f"{alt} ({name})"
                        else:
                            field_key = alt or name or f"field_{page_num}_{a_idx + 1}"

                        if val:
                            per_page.setdefault(page_num, {})[field_key] = val
                    finally:
                        pdfium_c.FPDFPage_CloseAnnot(annot)
            finally:
                page.close()
    except Exception as e:
        logger.debug(f"Error extracting AcroForm fields: {e}")
    finally:
        try:
            doc.close_forms()
        except Exception:
            pass

    return per_page


def _format_acroform_fields_markdown(fields: dict[str, str]) -> str:
    """Format AcroForm key-values into a high-visibility markdown table."""
    if not fields:
        return ""
    lines = ["### Dijital Form Alanları (AcroForm)\n| Alan Adı | Değer |", "| :--- | :--- |"]
    for k, v in fields.items():
        clean_k = k.replace("|", "-").replace("\n", " ").strip()
        clean_v = str(v).replace("|", "-").replace("\n", " ").strip()
        lines.append(f"| {clean_k} | {clean_v} |")
    return "\n".join(lines) + "\n\n"
