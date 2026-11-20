"""The transcription pipeline: validate -> stage -> inspect -> transcribe -> post-process."""

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.errors import (
    AppError,
    BackendUnavailable,
    InvalidLanguage,
    ServerBusy,
    TranscriptionFailed,
)
from app.core.languages import SUPPORTED_LANGUAGES
from app.providers.base import ProviderResult, TranscriptionProvider
from app.schemas.transcription import TranscriptionResponse
from app.services.audio import (
    AsyncReadable,
    AudioInfo,
    inspect_audio,
    save_upload,
    validated_extension,
)
from app.services.postprocess import assemble_transcript, build_segments, count_words

logger = logging.getLogger(__name__)

AudioInspector = Callable[..., AudioInfo]


def validate_language(language: str | None) -> str | None:
    """Normalize the optional hint. Blank means 'detect'; unknown codes are rejected."""
    if language is None or not language.strip():
        return None
    code = language.strip().lower()
    if code not in SUPPORTED_LANGUAGES:
        raise InvalidLanguage()
    return code


class TranscriptionService:
    def __init__(
        self,
        settings: Settings,
        provider: TranscriptionProvider | None,
        inspector: AudioInspector = inspect_audio,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._inspector = inspector
        # Inference is CPU/GPU-bound; this caps how many run at once in this process.
        self._slots = asyncio.Semaphore(settings.max_concurrent_transcriptions)
        if settings.temp_dir is not None:
            settings.temp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_loaded(self) -> bool:
        return self._provider is not None

    async def transcribe_upload(
        self, upload: AsyncReadable, filename: str | None, language: str | None, request_id: str
    ) -> TranscriptionResponse:
        started = time.perf_counter()
        provider = self._provider
        if provider is None:
            raise BackendUnavailable()

        # Cheap checks first, before any bytes are written to disk.
        language = validate_language(language)
        suffix = validated_extension(filename)

        path = await save_upload(
            upload,
            suffix=suffix,
            max_bytes=self._settings.max_upload_bytes,
            directory=self._settings.temp_dir,
        )
        try:
            info = await run_in_threadpool(
                self._inspector,
                path,
                max_duration_seconds=self._settings.max_audio_seconds,
                timeout_seconds=self._settings.ffmpeg_timeout_seconds,
            )
            result = await self._run_inference(provider, path, language, request_id)
        finally:
            # Single cleanup point for every path: success, validation error, model failure.
            path.unlink(missing_ok=True)

        segments = build_segments(result.segments)
        transcript = assemble_transcript(segments)
        elapsed = time.perf_counter() - started
        logger.info(
            "transcription completed request_id=%s audio_seconds=%.2f segments=%d elapsed_seconds=%.2f",
            request_id, info.duration_seconds, len(segments), elapsed,
        )
        return TranscriptionResponse(
            id=request_id,
            transcript=transcript,
            # A language "detected" from audio with no speech is noise, so report none.
            language=language or (result.language if segments else None),
            duration_seconds=info.duration_seconds,
            processing_time_seconds=round(elapsed, 3),
            word_count=count_words(transcript),
            model=provider.model_name,
            segments=segments,
        )

    async def _run_inference(
        self, provider: TranscriptionProvider, path: Path, language: str | None, request_id: str
    ) -> ProviderResult:
        # Backpressure: wait a bounded time for an inference slot, then shed load with 503
        # instead of letting requests (and their temp files) pile up without limit.
        try:
            await asyncio.wait_for(
                self._slots.acquire(), timeout=self._settings.queue_timeout_seconds
            )
        except TimeoutError:
            raise ServerBusy(headers={"Retry-After": "5"}) from None
        try:
            # Off the event loop, so /health and other requests stay responsive.
            return await run_in_threadpool(provider.transcribe, path, language)
        except AppError:
            raise
        except Exception:
            # Full detail to server logs; a generic, safe error to the client.
            logger.exception("transcription provider failed request_id=%s", request_id)
            raise TranscriptionFailed() from None
        finally:
            self._slots.release()
