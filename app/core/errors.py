"""Typed application errors and the handlers that turn them into one JSON shape.

Every error response looks like:
    {"error": {"code": "...", "message": "...", "request_id": "..."}}
"""

import logging
import uuid
from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code = 500
    code = "INTERNAL_ERROR"
    message = "An unexpected error occurred."

    def __init__(self, message: str | None = None, headers: Mapping[str, str] | None = None):
        self.message = message or self.message
        self.headers = dict(headers or {})
        super().__init__(self.message)


class MissingFile(AppError):
    status_code = 400
    code = "MISSING_FILE"
    message = "A multipart form field named 'file' containing an audio file is required."


class EmptyFile(AppError):
    status_code = 400
    code = "EMPTY_FILE"
    message = "The uploaded file is empty."


class InvalidAudio(AppError):
    status_code = 400
    code = "INVALID_AUDIO"
    message = "The file could not be decoded as audio."


class FileTooLarge(AppError):
    status_code = 413
    code = "FILE_TOO_LARGE"
    message = "The uploaded file exceeds the maximum allowed size."


class UnsupportedFormat(AppError):
    status_code = 415
    code = "UNSUPPORTED_AUDIO_FORMAT"
    message = "Supported formats: WAV, MP3, M4A."


class InvalidLanguage(AppError):
    status_code = 422
    code = "INVALID_LANGUAGE"
    message = "Unsupported language hint. Use a Whisper language code such as 'en', or omit it."


class AudioTooLong(AppError):
    status_code = 422
    code = "AUDIO_TOO_LONG"
    message = "The audio exceeds the maximum allowed duration."


class TranscriptionFailed(AppError):
    status_code = 500
    code = "TRANSCRIPTION_FAILED"
    message = "Transcription failed unexpectedly."


class BackendUnavailable(AppError):
    status_code = 503
    code = "TRANSCRIPTION_UNAVAILABLE"
    message = "The transcription backend is not available."


class ServerBusy(AppError):
    status_code = 503
    code = "SERVER_BUSY"
    message = "The server is busy transcribing other audio. Retry shortly."


_HTTP_STATUS_CODES = {
    400: "BAD_REQUEST",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    413: FileTooLarge.code,
}


def error_response(
    request_id: str,
    status_code: int,
    code: str,
    message: str,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers={**(headers or {}), "X-Request-ID": request_id},
    )


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return error_response(
            _request_id(request), exc.status_code, exc.code, exc.message, exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [str(err["loc"][-1]) for err in exc.errors() if err.get("loc")]
        if "file" in fields:
            missing = MissingFile()
            return error_response(
                _request_id(request), missing.status_code, missing.code, missing.message
            )
        # Name the offending fields only; never echo submitted values back.
        detail = ", ".join(sorted(set(fields))) or "request"
        return error_response(
            _request_id(request), 422, "VALIDATION_ERROR", f"Invalid request field(s): {detail}."
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_STATUS_CODES.get(exc.status_code, "HTTP_ERROR")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return error_response(_request_id(request), exc.status_code, code, message)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # The traceback goes to server logs only; the client gets a generic message.
        logger.error(
            "unhandled error request_id=%s type=%s",
            _request_id(request),
            type(exc).__name__,
            exc_info=exc,
        )
        return error_response(
            _request_id(request), 500, AppError.code, AppError.message
        )
