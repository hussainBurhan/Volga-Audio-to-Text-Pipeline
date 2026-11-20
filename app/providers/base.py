"""The seam between the pipeline and any speech-to-text engine.

The pipeline only knows this protocol, so the local Whisper model can be replaced by a
hosted API — or by a fake in tests — without touching routes or services.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class RawSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class ProviderResult:
    segments: list[RawSegment]
    language: str | None


class TranscriptionProvider(Protocol):
    model_name: str

    def transcribe(self, audio_path: Path, language: str | None) -> ProviderResult:
        """Blocking transcription of one audio file. Called from a worker thread."""
        ...
