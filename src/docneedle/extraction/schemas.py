"""Information extraction and verification schemas."""

from pydantic import BaseModel, Field


class ExtractionItemQuery(BaseModel):
    """Specification of an item to extract from the ingested documents."""

    key: str = Field(description="Unique identifier key for the requested information")
    description: str = Field(description="Natural language description or question")
    type: str = Field(
        default="string",
        description=(
            "Expected type: string, number, money, date, time, email, "
            "phone, id, iban, percentage, person, organization, address, boolean, list"
        ),
    )


class ExtractionEvidence(BaseModel):
    """Source document provenance."""

    document: str
    page: int | None = None
    snippet: str | None = None


class ExtractionItemResult(BaseModel):
    """Result of extraction for a single requested item."""

    key: str
    found: bool
    value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: ExtractionEvidence | None = None
    thinking: str | None = None


class ExtractionResponse(BaseModel):
    """Full extraction response returning all extracted items and execution metadata."""

    status: str = "completed"
    processing_time_ms: float
    total_items: int
    found_items: int
    results: list[ExtractionItemResult]
