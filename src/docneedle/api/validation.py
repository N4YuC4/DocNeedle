"""Request validation and parsing utilities for DocNeedle API endpoints."""

import json
from typing import Final

from fastapi import HTTPException, UploadFile

from docneedle.config import settings
from docneedle.extraction.schemas import ExtractionItemQuery

BYTES_PER_MEGABYTE: Final[int] = 1024 * 1024


def parse_extraction_schema(items_schema: str) -> list[ExtractionItemQuery]:
    """Parse and validate extraction schema JSON payload into query models."""
    try:
        raw_items = json.loads(items_schema)
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        items = [ExtractionItemQuery.model_validate(it) for it in raw_items]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid schema JSON format: {e}") from e

    if not items:
        raise HTTPException(
            status_code=400,
            detail="At least one extraction item must be provided in schema.",
        )
    return items


def validate_file_size(filename: str, content: bytes, max_bytes: int) -> None:
    """Enforce maximum upload payload constraint per file."""
    if max_bytes > 0 and len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File '{filename}' exceeds maximum allowed upload size of "
                f"{settings.max_upload_size_mb} MB."
            ),
        )


async def read_and_validate_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    """Read uploaded files into memory and validate non-empty list and per-file size limits."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one document file must be uploaded.")

    max_bytes = settings.max_upload_size_mb * BYTES_PER_MEGABYTE
    files_data: list[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        filename = upload.filename or "uploaded_document"
        validate_file_size(filename, content, max_bytes)
        files_data.append((filename, content))
    return files_data

