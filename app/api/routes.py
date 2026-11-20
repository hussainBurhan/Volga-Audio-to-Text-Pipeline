"""HTTP layer only: parse the request, call the service, serialize the result."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.schemas.transcription import ErrorResponse, HealthResponse, TranscriptionResponse
from app.services.transcription import TranscriptionService

router = APIRouter()


def get_service(request: Request) -> TranscriptionService:
    return request.app.state.service


ServiceDep = Annotated[TranscriptionService, Depends(get_service)]

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Missing, empty, or undecodable audio."},
    413: {"model": ErrorResponse, "description": "Upload larger than the configured limit."},
    415: {"model": ErrorResponse, "description": "Unsupported file extension or container."},
    422: {"model": ErrorResponse, "description": "Invalid language hint, or audio too long."},
    500: {"model": ErrorResponse, "description": "Unexpected transcription failure."},
    503: {"model": ErrorResponse, "description": "Model not loaded, or server busy."},
}


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(service: ServiceDep) -> HealthResponse:
    """Liveness check. Also reports whether the speech model is loaded (no test inference)."""
    return HealthResponse(model_loaded=service.model_loaded)


@router.post(
    "/api/v1/transcriptions",
    response_model=TranscriptionResponse,
    responses=_ERROR_RESPONSES,
    tags=["transcriptions"],
)
async def create_transcription(
    request: Request,
    service: ServiceDep,
    file: Annotated[UploadFile, File(description="Audio file: .wav, .mp3 or .m4a.")],
    language: Annotated[
        str | None,
        Form(description="Optional language hint such as `en`. Omit to auto-detect."),
    ] = None,
) -> TranscriptionResponse:
    """Transcribe one short audio file synchronously and return structured JSON."""
    return await service.transcribe_upload(
        file, file.filename, language, request.state.request_id
    )
