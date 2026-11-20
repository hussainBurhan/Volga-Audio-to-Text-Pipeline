"""In-memory store for asynchronous transcription jobs.

Deliberately simple: one process, no persistence. Jobs vanish on restart and are not
shared between worker processes. A production version would put this in a database
behind the same interface.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app.core.errors import JobNotFound, ServerBusy
from app.schemas.transcription import ErrorDetail, TranscriptionResponse

JobStatus = Literal["queued", "processing", "completed", "failed"]
_ACTIVE: tuple[JobStatus, ...] = ("queued", "processing")


@dataclass
class Job:
    id: str
    status: JobStatus = "queued"
    result: TranscriptionResponse | None = None
    error: ErrorDetail | None = None
    finished_at: float | None = None


class JobStore:
    def __init__(
        self, max_active: int, ttl_seconds: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._max_active = max_active
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._jobs: dict[str, Job] = {}

    def create(self, job_id: str) -> Job:
        """Reserve a job slot, or shed load if too many jobs are already waiting."""
        self._purge_expired()
        if sum(job.status in _ACTIVE for job in self._jobs.values()) >= self._max_active:
            raise ServerBusy(
                "Too many transcription jobs are in progress. Retry shortly.",
                headers={"Retry-After": "30"},
            )
        job = self._jobs[job_id] = Job(id=job_id)
        return job

    def get(self, job_id: str) -> Job:
        self._purge_expired()
        try:
            return self._jobs[job_id]
        except KeyError:
            raise JobNotFound() from None

    def discard(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def complete(self, job: Job, result: TranscriptionResponse) -> None:
        job.status, job.result, job.finished_at = "completed", result, self._clock()

    def fail(self, job: Job, code: str, message: str) -> None:
        job.error = ErrorDetail(code=code, message=message, request_id=job.id)
        job.status, job.finished_at = "failed", self._clock()

    def _purge_expired(self) -> None:
        # Finished results (which contain transcript text) are dropped after the TTL.
        cutoff = self._clock() - self._ttl_seconds
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.finished_at is not None and job.finished_at < cutoff
        ]
        for job_id in expired:
            del self._jobs[job_id]
