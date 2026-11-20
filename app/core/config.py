"""Environment-driven settings. Every limit here is our assumption, not an employer requirement."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    # Input limits
    max_upload_bytes: int = Field(20 * 1024 * 1024, gt=0)
    max_audio_seconds: float = Field(300.0, gt=0)

    # Speech-to-text model
    stt_model_size: str = "base"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_beam_size: int = Field(5, ge=1)
    stt_vad_filter: bool = True

    # Concurrency / backpressure
    max_concurrent_transcriptions: int = Field(1, ge=1)
    queue_timeout_seconds: float = Field(30.0, ge=0)

    # FFmpeg tooling
    ffmpeg_timeout_seconds: float = Field(30.0, gt=0)

    # Where uploads are staged; None means the system temp directory.
    temp_dir: Path | None = None

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
