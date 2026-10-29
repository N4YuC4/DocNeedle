import asyncio
import logging

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
        enable_thinking: bool | None = None,
    ) -> LLMExtractionResult:
        """Extract item directly via in-process local neural model."""
        local_engine = get_local_engine()
        return local_engine.extract_item(
            query_key=query_key,
            query_description=query_description,
            context=context,
            image_bytes_list=image_bytes_list,
            expected_type=expected_type,
            enable_thinking=enable_thinking,
        )
