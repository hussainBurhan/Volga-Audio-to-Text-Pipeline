"""The transcription pipeline: validate -> stage -> inspect -> transcribe -> post-process.

Two entry points share it: a synchronous one for short clips, and background jobs for
long audio, where holding an HTTP connection open for the whole inference is not sensible.
"""

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
from app.services.jobs import Job, JobStore
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
        # Synchronous requests and background jobs compete for the same slots.
        self._slots = asyncio.Semaphore(settings.max_concurrent_transcriptions)
        self._jobs = JobStore(settings.max_active_jobs, settings.job_ttl_seconds)
        self._tasks: set[asyncio.Task[None]] = set()  # keeps running jobs referenced
        if settings.temp_dir is not None:
            settings.temp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_loaded(self) -> bool:
        return self._provider is not None

    # --- synchronous: short clips ------------------------------------------------------

    async def transcribe_upload(
        self, upload: AsyncReadable, filename: str | None, language: str | None, request_id: str
    ) -> TranscriptionResponse:
        started = time.perf_counter()
        provider = self._require_provider()
        path, info, language = await self._stage(
            upload,
            filename,
            language,
            max_bytes=self._settings.max_upload_bytes,
            max_seconds=self._settings.max_audio_seconds,
        )
        try:
            return await self._transcribe_staged(
                provider, path, info, language, request_id, started,
                queue_timeout=self._settings.queue_timeout_seconds,
            )
        finally:
            path.unlink(missing_ok=True)

    # --- asynchronous: long audio ------------------------------------------------------

    async def submit_job(
        self, upload: AsyncReadable, filename: str | None, language: str | None, request_id: str
    ) -> Job:
        """Validate and stage the upload now; transcribe in the background.

        Bad input still fails fast with a normal 4xx. Only the slow part is deferred.
        """
        started = time.perf_counter()
        provider = self._require_provider()
        job = self._jobs.create(request_id)  # reserves capacity before accepting bytes
        try:
            path, info, language = await self._stage(
                upload,
                filename,
                language,
                max_bytes=self._settings.max_job_upload_bytes,
                max_seconds=self._settings.max_job_audio_seconds,
            )
        except BaseException:
            self._jobs.discard(job.id)
            raise
        task = asyncio.create_task(self._run_job(job, provider, path, info, language, started))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def get_job(self, job_id: str) -> Job:
        return self._jobs.get(job_id)

    async def _run_job(
        self,
        job: Job,
        provider: TranscriptionProvider,
        path: Path,
        info: AudioInfo,
        language: str | None,
        started: float,
    ) -> None:
        def mark_processing() -> None:
            job.status = "processing"

        try:
            # No queue timeout: a job is allowed to wait its turn.
            result = await self._transcribe_staged(
                provider, path, info, language, job.id, started,
                queue_timeout=None, on_start=mark_processing,
            )
            self._jobs.complete(job, result)
        except AppError as exc:
            self._jobs.fail(job, exc.code, exc.message)
        except Exception:
            logger.exception("transcription job failed job_id=%s", job.id)
            self._jobs.fail(job, AppError.code, AppError.message)
        except asyncio.CancelledError:  # shutdown
            self._jobs.fail(job, AppError.code, "The server stopped before this job finished.")
            raise
        finally:
            path.unlink(missing_ok=True)

    # --- shared steps ------------------------------------------------------------------

    def _require_provider(self) -> TranscriptionProvider:
        if self._provider is None:
            raise BackendUnavailable()
        return self._provider

    async def _stage(
        self,
        upload: AsyncReadable,
        filename: str | None,
        language: str | None,
        *,
        max_bytes: int,
        max_seconds: float,
    ) -> tuple[Path, AudioInfo, str | None]:
        """Validate the request and leave a checked audio file on disk. Caller deletes it."""
        # Cheap checks first, before any bytes are written to disk.
        language = validate_language(language)
        suffix = validated_extension(filename)
        path = await save_upload(
            upload, suffix=suffix, max_bytes=max_bytes, directory=self._settings.temp_dir
        )
        try:
            info = await run_in_threadpool(
                self._inspector,
                path,
                max_duration_seconds=max_seconds,
                timeout_seconds=self._settings.ffmpeg_timeout_seconds,
            )
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path, info, language

    async def _transcribe_staged(
        self,
        provider: TranscriptionProvider,
        path: Path,
        info: AudioInfo,
        language: str | None,
        request_id: str,
        started: float,
        *,
        queue_timeout: float | None,
        on_start: Callable[[], None] | None = None,
    ) -> TranscriptionResponse:
        result = await self._run_inference(
            provider, path, language, request_id, queue_timeout, on_start
        )
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
        self,
        provider: TranscriptionProvider,
        path: Path,
        language: str | None,
        request_id: str,
        queue_timeout: float | None,
        on_start: Callable[[], None] | None,
    ) -> ProviderResult:
        # Backpressure: a synchronous request waits a bounded time for an inference slot,
        # then is shed with 503 instead of piling up without limit.
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=queue_timeout)
        except TimeoutError:
            raise ServerBusy(headers={"Retry-After": "5"}) from None
        try:
            if on_start:
                on_start()
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
