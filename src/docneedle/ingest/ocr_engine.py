"""Ephemeral on-demand OCR engine using RapidOCR (PP-OCRv6 via ONNX Runtime)."""

from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from docneedle.config import settings

logger = logging.getLogger(__name__)

DEFAULT_LONGEST_EDGE = 1500

_TURKISH_UNIQUE_CHARS = frozenset("ğĞşŞıİ")
_TURKISH_CONTEXT_KEYWORDS = (
    "T.C.",
    "TÜRKIYE",
    "TÜRKİYE",
    "CUMHURIYET",
    "KIMLIK",
    "KİMLİK",
    "BAKANLI",
    "MÜDÜRLÜ",
    "BELGES",
    "NOTER",
    "VERGI",
    "FATURA",
    "SAYFA",
    "DOGUM",
    "DOĞUM",
)
_TURKISH_ARTIFACT_MAP = str.maketrans(
    {
        "Č": "Ğ",
        "č": "ğ",
        "Ġ": "Ğ",
        "ġ": "ğ",
        "Í": "İ",
        "ę": "e",
    }
)


def fit_image(image: Image.Image, longest_edge: int | None = DEFAULT_LONGEST_EDGE) -> Image.Image:
    """Downscale image so max(width, height) <= longest_edge. Never upscales."""
    if longest_edge is None or longest_edge <= 0:
        return image
    w, h = image.size
    scale = longest_edge / max(w, h)
    if scale >= 1:
        return image
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def has_turkish_context(text: str) -> bool:
    """Detect if transcribed text belongs to a Turkish document without language parameters."""
    if any(ch in _TURKISH_UNIQUE_CHARS for ch in text):
        return True
    upper = text.upper()
    return any(kw in upper for kw in _TURKISH_CONTEXT_KEYWORDS)


def normalize_turkish_ocr_artifacts(text: str) -> str:
    """Fix look-alike Latin/Eastern-European glyph confusions when text is in Turkish context."""
    if not text or not has_turkish_context(text):
        return text
    return text.translate(_TURKISH_ARTIFACT_MAP)


# OCR Layout Geometry & Clustering Constants
OCR_BOX_HORIZONTAL_OVERLAP_RATIO = 0.4
OCR_LINE_VERTICAL_TOLERANCE_RATIO = 0.5
OCR_COLUMN_GAP_CHAR_RATIO = 1.0
OCR_COLUMN_GAP_MIN_PX = 8.0
OCR_COLUMN_GAP_SECONDARY_RATIO = 0.8
OCR_PARAGRAPH_VERTICAL_GAP_RATIO = 0.85


@dataclass(slots=True)
class BoundingBoxProps:
    """Geometric bounding box properties of an OCR-transcribed text token."""

    text: str
    left: float
    right: float
    top: float
    bottom: float
    width: float
    height: float
    center_y: float
    char_w: float

    def overlaps_horizontally_with(
        self,
        other: BoundingBoxProps,
        min_overlap_ratio: float = OCR_BOX_HORIZONTAL_OVERLAP_RATIO,
    ) -> bool:
        """Check if this box shares significant horizontal X span with another box."""
        overlap_x = min(self.right, other.right) - max(self.left, other.left)
        min_w = min(self.width, other.width)
        return min_w > 0 and (overlap_x / min_w) > min_overlap_ratio


@dataclass(slots=True)
class LayoutLine:
    """A clustered reading line composed of horizontally distributed bounding boxes."""

    items: list[BoundingBoxProps] = field(default_factory=list)
    sum_cy: float = 0.0
    sum_h: float = 0.0
    top: float = float("inf")
    bottom: float = float("-inf")

    @property
    def mean_center_y(self) -> float:
        return self.sum_cy / len(self.items) if self.items else 0.0

    @property
    def mean_height(self) -> float:
        return self.sum_h / len(self.items) if self.items else 0.0

    def add_item(self, item: BoundingBoxProps) -> None:
        self.items.append(item)
        self.sum_cy += item.center_y
        self.sum_h += item.height
        self.top = min(self.top, item.top)
        self.bottom = max(self.bottom, item.bottom)

    def contains_horizontal_overlap(
        self,
        item: BoundingBoxProps,
        min_overlap_ratio: float = OCR_BOX_HORIZONTAL_OVERLAP_RATIO,
    ) -> bool:
        return any(
            item.overlaps_horizontally_with(existing, min_overlap_ratio) for existing in self.items
        )


def _extract_box_props(box: Any, text: str) -> BoundingBoxProps | None:
    """Extract geometric bounding box properties from a 4-point polygon."""
    try:
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
    except (TypeError, ValueError, IndexError):  # fmt: skip
        return None

    if len(xs) < 4 or len(ys) < 4:
        return None

    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    center_y = (top + bottom) / 2.0
    char_w = max(1.0, width / max(1, len(text)))
    return BoundingBoxProps(
        text=text,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        width=width,
        height=height,
        center_y=center_y,
        char_w=char_w,
    )


def _parse_bounding_boxes(boxes: Any, txts: Sequence[str]) -> list[BoundingBoxProps]:
    """Parse raw boxes and text sequence into valid geometric properties."""
    if not txts or boxes is None:
        return []
    items: list[BoundingBoxProps] = []
    try:
        if len(boxes) == len(txts):
            for box, raw_txt in zip(boxes, txts, strict=False):
                clean_txt = str(raw_txt).strip()
                if not clean_txt:
                    continue
                props = _extract_box_props(box, clean_txt)
                if props is not None:
                    items.append(props)
    except TypeError:
        return []
    return items


def _cluster_into_lines(items: list[BoundingBoxProps]) -> list[LayoutLine]:
    """Group sorted bounding boxes into reading lines based on vertical proximity
    and non-overlap.
    """
    lines: list[LayoutLine] = []
    for item in items:
        best_line: LayoutLine | None = None
        best_dy = float("inf")

        for line in lines:
            mean_cy = line.mean_center_y
            mean_h = line.mean_height
            ref_h = min(item.height, mean_h)
            dy = abs(item.center_y - mean_cy)

            if (
                dy <= OCR_LINE_VERTICAL_TOLERANCE_RATIO * ref_h
                and not line.contains_horizontal_overlap(item)
            ):
                if dy < best_dy:
                    best_dy = dy
                    best_line = line

        if best_line is None:
            new_line = LayoutLine()
            new_line.add_item(item)
            lines.append(new_line)
        else:
            best_line.add_item(item)

    lines.sort(key=lambda ln: ln.mean_center_y)
    return lines


def _render_line_text(line: LayoutLine) -> str:
    """Format single line text with column gap analysis."""
    line_items = sorted(line.items, key=lambda it: it.left)
    parts = [line_items[0].text]
    prev_item = line_items[0]

    for curr_item in line_items[1:]:
        gap = curr_item.left - prev_item.right
        avg_char_w = (prev_item.char_w + curr_item.char_w) / 2.0
        separator = (
            " | "
            if (
                gap > OCR_COLUMN_GAP_CHAR_RATIO * avg_char_w
                or (
                    gap >= OCR_COLUMN_GAP_MIN_PX
                    and gap > OCR_COLUMN_GAP_SECONDARY_RATIO * avg_char_w
                )
            )
            else " "
        )
        parts.append(separator)
        parts.append(curr_item.text)
        prev_item = curr_item

    return "".join(parts)


def _format_layout_lines(lines: list[LayoutLine]) -> str:
    """Assemble formatted lines and insert vertical blank line separators for paragraph breaks."""
    output_lines: list[str] = []
    prev_line: LayoutLine | None = None

    for line in lines:
        if prev_line is not None:
            v_gap = line.top - prev_line.bottom
            prev_h = prev_line.mean_height
            curr_h = line.mean_height
            if v_gap > OCR_PARAGRAPH_VERTICAL_GAP_RATIO * max(prev_h, curr_h):
                output_lines.append("")

        output_lines.append(_render_line_text(line))
        prev_line = line

    return "\n".join(output_lines)


def format_ocr_layout(boxes: Any, txts: Sequence[str]) -> str:
    """Reconstruct spatial reading order and column separation from OCR bounding boxes.

    Prevents adjacent boxes from gluing without spaces and prevents tilted multi-row forms
    from collapsing into a single garbled line.
    """
    items = _parse_bounding_boxes(boxes, txts)
    if not items:
        return ""

    items.sort(key=lambda it: (it.center_y, it.left))
    lines = _cluster_into_lines(items)
    return _format_layout_lines(lines)


def _extract_raw_ocr_text(res: Any) -> str:
    """Extract raw text string from RapidOCR output object or legacy tuple."""
    if res is None:
        return ""

    if hasattr(res, "txts"):
        if not res.txts:
            return ""
        boxes = getattr(res, "boxes", None)
        layout_text = format_ocr_layout(boxes, res.txts)
        if layout_text:
            return layout_text
        if hasattr(res, "to_markdown"):
            md = res.to_markdown()
            if isinstance(md, str) and md.strip():
                return md.strip()
        return "\n".join(str(t).strip() for t in res.txts if str(t).strip())

    if isinstance(res, tuple):
        items = res[0]
        if not items:
            return ""
        lines: list[str] = []
        for item in items:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                txt = str(item[1]).strip()
            else:
                txt = str(item).strip()
            if txt:
                lines.append(txt)
        return "\n".join(lines)

    return ""


def _resolve_ocr_use_cuda(use_cuda_override: bool | None = None) -> bool:
    """Determine whether RapidOCR should run on ONNX Runtime CUDAExecutionProvider."""
    want_cuda = (
        use_cuda_override
        if use_cuda_override is not None
        else (settings.gpu_enabled and settings.ocr_use_cuda)
    )
    if not want_cuda:
        return False

    try:
        from docneedle.cuda_compat import preload_cuda_runtime_libs

        preload_cuda_runtime_libs()
    except Exception as e:
        logger.debug("CUDA runtime preload before OCR check failed: %s", e)

    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(directory="")
            except Exception as e:
                logger.debug("onnxruntime.preload_dlls() notice: %s", e)

        has_cuda_ep = (
            ort.get_device() == "GPU" and "CUDAExecutionProvider" in ort.get_available_providers()
        )
        if not has_cuda_ep:
            logger.info(
                "RapidOCR GPU requested, but CUDAExecutionProvider is unavailable "
                "(device=%s, providers=%s); running on CPUExecutionProvider.",
                ort.get_device(),
                ort.get_available_providers(),
            )
        return has_cuda_ep
    except Exception as e:
        logger.debug("Could not verify ONNX Runtime CUDA provider: %s", e)
        return False


class EphemeralRapidOCREngine:
    """Manages ephemeral in-process RapidOCR inference with immediate RAM/VRAM deallocation.

    Loads the ONNX Runtime detection, direction classification, and recognition models
    on entering context, performs zero-parameter multilingual transcription in RAM/VRAM,
    and frees all session references and native memory on exit.
    """

    def __init__(
        self,
        text_score: float | None = None,
        use_cuda: bool | None = None,
    ) -> None:
        self.text_score = settings.ocr_text_score if text_score is None else text_score
        self.use_cuda = use_cuda
        self.active_use_cuda: bool = False
        self.ocr: Any = None
        self._lock = threading.Lock()

    def __enter__(self) -> EphemeralRapidOCREngine:
        from rapidocr import ModelType, RapidOCR

        det_tier = ModelType(settings.ocr_det_model_type.lower())
        rec_tier = ModelType(settings.ocr_rec_model_type.lower())
        use_cuda = _resolve_ocr_use_cuda(self.use_cuda)
        self.active_use_cuda = use_cuda
        ep_label = (
            f"CUDAExecutionProvider (device={settings.ocr_gpu_device_id})"
            if use_cuda
            else "CPUExecutionProvider"
        )
        logger.info(
            "Loading ephemeral RapidOCR engine (ONNX Runtime %s, det=%s, rec=%s)...",
            ep_label,
            det_tier.value,
            rec_tier.value,
        )
        params: dict[str, Any] = {
            "Global.text_score": self.text_score,
            "Global.max_side_len": settings.ocr_longest_edge,
            "Global.use_cls": False,
            "Det.model_type": det_tier,
            "Rec.model_type": rec_tier,
            "Rec.rec_batch_num": settings.ocr_rec_batch_num if use_cuda else 6,
            "Det.limit_side_len": settings.ocr_det_limit_side_len,
            "Det.box_thresh": settings.ocr_det_box_thresh,
            "Det.unclip_ratio": settings.ocr_det_unclip_ratio,
            "EngineConfig.onnxruntime.use_cuda": use_cuda,
        }
        if use_cuda:
            params.update(
                {
                    "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": settings.ocr_gpu_device_id,
                    "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": (
                        settings.ocr_cuda_conv_algo_search
                    ),
                    "EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy": (
                        settings.ocr_cuda_arena_extend_strategy
                    ),
                }
            )
        try:
            self.ocr = RapidOCR(params=params)
        except TypeError:
            self.ocr = RapidOCR()
        self.ocr.text_score = self.text_score
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        logger.info("Evicting ephemeral RapidOCR engine from RAM/VRAM...")
        self.ocr = None
        self.active_use_cuda = False
        gc.collect()
        logger.info("Ephemeral RapidOCR memory successfully reclaimed.")

    def transcribe_image(self, image: Image.Image | np.ndarray) -> str:
        """Transcribe text from an image or rendered PDF page into clean markdown."""
        if self.ocr is None:
            raise RuntimeError("EphemeralRapidOCREngine must be used within a 'with' block.")

        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        pil_img = fit_image(pil_img, longest_edge=settings.ocr_longest_edge)
        img_array = np.array(pil_img)

        with self._lock:
            res = self.ocr(img_array)

        raw_text = _extract_raw_ocr_text(res)
        return normalize_turkish_ocr_artifacts(raw_text.strip())


RapidOCREngine = EphemeralRapidOCREngine
