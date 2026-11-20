"""faster-whisper implementation of TranscriptionProvider."""

from pathlib import Path

from app.core.config import Settings
from app.providers.base import ProviderResult, RawSegment


class WhisperProvider:
    def __init__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        beam_size: int,
        vad_filter: bool,
    ) -> None:
        # Imported lazily so the fast test suite never imports the ML stack.
        from faster_whisper import WhisperModel

        # Downloads weights from the Hugging Face Hub on first use, then reads the local cache.
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self.model_name = f"faster-whisper-{model_size}"

    @classmethod
    def from_settings(cls, settings: Settings) -> "WhisperProvider":
        return cls(
            model_size=settings.stt_model_size,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            beam_size=settings.stt_beam_size,
            vad_filter=settings.stt_vad_filter,
        )

    def transcribe(self, audio_path: Path, language: str | None) -> ProviderResult:
        segments, info = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=self._beam_size,
            # Voice-activity detection drops non-speech audio before decoding, which avoids
            # Whisper's tendency to hallucinate text on silence.
            vad_filter=self._vad_filter,
        )
        # `segments` is a lazy generator: inference actually runs while it is consumed.
        raw = [RawSegment(start=s.start, end=s.end, text=s.text) for s in segments]
        return ProviderResult(segments=raw, language=info.language)
