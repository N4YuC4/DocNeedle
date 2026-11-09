"""Request validation and parsing utilities for DocNeedle API endpoints."""

import json
from typing import Any, Final

from fastapi import HTTPException, UploadFile

from docneedle.config import settings
from docneedle.extraction.schemas import ExtractionItemQuery

BYTES_PER_MEGABYTE: Final[int] = 1024 * 1024


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


def parse_extraction_schema(items_schema: str) -> list[ExtractionItemQuery]:
    """Parse and validate extraction schema JSON payload into query models."""
    try:
        raw_items = json.loads(items_schema)
        if isinstance(raw_items, dict):
            if "$defs" in raw_items or "definitions" in raw_items:
                raw_items = dereference_json_schema(raw_items, raw_items)
            raw_items = [raw_items]
        elif isinstance(raw_items, list):
            deref_items = []
            for it in raw_items:
                if isinstance(it, dict) and isinstance(it.get("items_schema"), dict):
                    it_copy = dict(it)
                    it_copy["items_schema"] = dereference_json_schema(
                        it["items_schema"], it["items_schema"]
                    )
                    deref_items.append(it_copy)
                else:
                    deref_items.append(it)
            raw_items = deref_items
        items = [ExtractionItemQuery.model_validate(it) for it in raw_items]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid schema JSON format: {e}") from e

    if not items:
        raise HTTPException(
            status_code=400,
            detail="At least one extraction item must be provided in schema.",
        )

    if len(items) > settings.max_schema_items:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Schema contains {len(items)} items, exceeding maximum allowed limit of "
                f"{settings.max_schema_items} items per request."
            ),
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

    if len(files) > settings.max_upload_files:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch contains {len(files)} files, exceeding maximum allowed limit of "
                f"{settings.max_upload_files} files per request."
            ),
        )

    max_file_bytes = settings.max_upload_size_mb * BYTES_PER_MEGABYTE
    max_total_bytes = settings.max_total_upload_size_mb * BYTES_PER_MEGABYTE
    total_bytes = 0

    files_data: list[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        total_bytes += len(content)
        if max_total_bytes > 0 and total_bytes > max_total_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Total upload payload size exceeds maximum allowed limit of "
                    f"{settings.max_total_upload_size_mb} MB."
                ),
            )
        filename = upload.filename or "uploaded_document"
        validate_file_size(filename, content, max_file_bytes)
        files_data.append((filename, content))
    return files_data
