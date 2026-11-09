import asyncio
import logging
from typing import Any

from docneedle.llm.local_engine import LLMExtractionResult, get_local_engine

logger = logging.getLogger(__name__)


class LLMClient:
    """In-process LLM client using local GGUF model."""

    async def extract_item(
        self,
        query_key: str,
        query_description: str,
        context: str,
        image_bytes_list: list[bytes] | None = None,
        expected_type: str | None = None,
        items_schema: dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
    ) -> LLMExtractionResult:
        """Extract item directly via in-process local neural model."""
        local_engine = get_local_engine()
        return await asyncio.to_thread(
            local_engine.extract_item,
            query_key=query_key,
            query_description=query_description,
            context=context,
            image_bytes_list=image_bytes_list,
            expected_type=expected_type,
            items_schema=items_schema,
            enable_thinking=enable_thinking,
        )

    async def extract_schema(
        self,
        fields: list[dict[str, Any]],
        context: str,
        image_bytes_list: list[bytes] | None = None,
        enable_thinking: bool | None = None,
    ) -> dict[str, LLMExtractionResult]:
        """Extract multiple schema items in a single pass over consolidated context."""
        local_engine = get_local_engine()
        return await asyncio.to_thread(
            local_engine.extract_schema,
            fields=fields,
            context=context,
            image_bytes_list=image_bytes_list,
            enable_thinking=enable_thinking,
        )
