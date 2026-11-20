"""Upload staging and media validation.

Nothing here trusts the client: the filename only contributes an allow-listed extension,
the size is counted while streaming, and the content is checked by actually probing and
decoding it with FFmpeg.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from starlette.concurrency import run_in_threadpool

from app.core.errors import (
    AudioTooLong,
    BackendUnavailable,
    EmptyFile,
    FileTooLarge,
    InvalidAudio,
    UnsupportedFormat,
)

SUPPORTED_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a"})
# ffprobe `format_name` tokens. M4A is reported as the family "mov,mp4,m4a,3gp,3g2,mj2".
SUPPORTED_CONTAINERS = frozenset({"wav", "mp3", "m4a"})

_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class AudioInfo:
    duration_seconds: float
    container: str
    codec: str


class AsyncReadable(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


def validated_extension(filename: str | None) -> str:
    """Return the allow-listed, lower-cased extension. The name is never used as a path."""
    extension = Path(filename or "").suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormat()
    return extension


async def save_upload(
    upload: AsyncReadable, *, suffix: str, max_bytes: int, directory: Path | None = None
) -> Path:
    """Stream an upload to a generated temp file, enforcing ``max_bytes`` as bytes arrive.

    The partial file is removed if anything goes wrong; on success the caller owns it.
    """
    fd, name = tempfile.mkstemp(prefix="upload-", suffix=suffix, dir=directory)
    path = Path(name)
    written = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(_CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    raise FileTooLarge()
                await run_in_threadpool(out.write, chunk)
        if written == 0:
            raise EmptyFile()
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def inspect_audio(path: Path, *, max_duration_seconds: float, timeout_seconds: float) -> AudioInfo:
    """Validate that ``path`` is supported, decodable audio within the duration limit.

    Blocking (runs FFmpeg subprocesses); call it from a worker thread.
    """
    probe = _probe(path, timeout_seconds)
    audio_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise InvalidAudio("The file contains no audio stream.")

    container = probe.get("format", {}).get("format_name", "")
    if not SUPPORTED_CONTAINERS.intersection(container.split(",")):
        raise UnsupportedFormat()

    header_duration = _to_float(probe.get("format", {}).get("duration")) or _to_float(
        audio_streams[0].get("duration")
    )
    # Cheap rejection first, based on what the container claims...
    if header_duration is not None and header_duration > max_duration_seconds:
        raise AudioTooLong()

    # ...then decode it for real: proves decodability and measures the true duration,
    # since header durations can be wrong or forged.
    decoded_duration = _decode(path, max_duration_seconds, timeout_seconds)
    duration = decoded_duration if decoded_duration is not None else header_duration
    if duration is None or duration <= 0:
        raise InvalidAudio()
    if duration > max_duration_seconds:
        raise AudioTooLong()

    return AudioInfo(
        duration_seconds=round(duration, 3),
        container=container,
        codec=audio_streams[0].get("codec_name", "unknown"),
    )


def _probe(path: Path, timeout_seconds: float) -> dict:
    result = _run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        timeout_seconds,
    )
    if result.returncode != 0:
        raise InvalidAudio()
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise InvalidAudio() from None


def _decode(path: Path, max_duration_seconds: float, timeout_seconds: float) -> float | None:
    """Decode the first audio stream to a null sink; return the decoded duration if reported.

    Decoding stops just past the limit, so an over-long file cannot make this run unbounded.
    """
    result = _run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-progress", "pipe:1",
            "-i", str(path), "-map", "0:a:0", "-t", str(max_duration_seconds + 1),
            "-f", "null", "-",
        ],
        timeout_seconds,
    )
    if result.returncode != 0:
        raise InvalidAudio()
    return _parse_decoded_seconds(result.stdout)


def _parse_decoded_seconds(progress_output: bytes) -> float | None:
    """Read the final ``out_time_us`` (microseconds) from ``ffmpeg -progress`` output."""
    seconds = None
    for line in progress_output.decode("utf-8", errors="replace").splitlines():
        key, _, value = line.partition("=")
        if key == "out_time_us" and value.strip().isdigit():
            seconds = int(value) / 1_000_000
    return seconds


def _run(command: list[str], timeout_seconds: float) -> subprocess.CompletedProcess[bytes]:
    # Argument list, never a shell string; subprocess.run kills the child on timeout.
    try:
        return subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise InvalidAudio("The audio could not be validated within the time limit.") from None
    except FileNotFoundError:
        raise BackendUnavailable("Audio tooling (FFmpeg) is not available on the server.") from None


def _to_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
