"""DocNeedle asynchronous job models and schemas."""

from enum import Enum

from pydantic import BaseModel, Field

from docneedle.extraction.schemas import ExtractionItemResult, ExtractionResponse


class JobStatus(str, Enum):
    """Lifecycle states of an extraction job."""

    PENDING = "pending"
    PARSING = "parsing"
    INDEXING = "indexing"
    EXTRACTING = "extracting"
    COMPLETED = "completed"
    FAILED = "failed"


class JobProgress(BaseModel):
    """Detailed real-time execution progress of an extraction job."""

    current_step: str = Field(default="Queued in memory", description="Human-readable current step")
    total_items: int = Field(default=0, description="Total target fields to extract")
    completed_items: int = Field(default=0, description="Number of fields already extracted")
    current_key: str | None = Field(
        default=None, description="Key of the field currently being evaluated"
    )
    percent: float = Field(default=0.0, description="Overall completion percentage (0.0 to 100.0)")


class JobCreateResponse(BaseModel):
    """Response returned immediately upon accepting an asynchronous extraction job."""

    job_id: str
    status: JobStatus
    poll_url: str
    events_url: str
    created_at: float


class JobStatusResponse(BaseModel):
    """Detailed state of an extraction job including progress and final or partial results."""

    job_id: str
    status: JobStatus
    progress: JobProgress
    created_at: float
    completed_at: float | None = None
    processing_time_ms: float | None = None
    results: list[ExtractionItemResult] = Field(default_factory=list)
    response: ExtractionResponse | None = None
    error: str | None = None
