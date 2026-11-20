"""Public API schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class Segment(BaseModel):
    start: float = Field(ge=0, description="Segment start offset in seconds.")
    end: float = Field(ge=0, description="Segment end offset in seconds.")
    text: str = Field(description="Whitespace-normalized segment text.")


class TranscriptionResponse(BaseModel):
    id: str = Field(description="Request ID. Identifies this response only; not retrievable later.")
    status: Literal["completed"] = "completed"
    transcript: str = Field(description="Segments joined with single spaces; whitespace-normalized only.")
    language: str | None = Field(
        description="Language hint if supplied, else the detected language; null when no speech was found."
    )
    duration_seconds: float = Field(description="Audio duration measured by FFmpeg.")
    processing_time_seconds: float = Field(
        description="Server wall time from receiving the parsed upload to building this response."
    )
    word_count: int = Field(ge=0, description="Whitespace-delimited word count of `transcript`.")
    model: str = Field(description="Identifier of the speech-to-text model used.")
    segments: list[Segment]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool = Field(
        description="Whether a transcription provider is loaded in this process. "
        "This is a liveness check; it does not run a test inference."
    )
