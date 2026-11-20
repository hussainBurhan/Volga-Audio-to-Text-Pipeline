import json
import subprocess
from pathlib import Path

import anyio
import pytest

from app.core.errors import (
    AudioTooLong,
    BackendUnavailable,
    EmptyFile,
    FileTooLarge,
    InvalidAudio,
    UnsupportedFormat,
)
from app.services import audio
from app.services.audio import inspect_audio, save_upload, validated_extension
from tests.conftest import generate_audio, requires_ffmpeg

LIMITS = {"max_duration_seconds": 300.0, "timeout_seconds": 30.0}


class ChunkedUpload:
    """Minimal async file: yields ``total`` bytes and records how many were requested."""

    def __init__(self, total: int) -> None:
        self.remaining = total
        self.served = 0

    async def read(self, size: int = -1) -> bytes:
        n = min(size, self.remaining)
        self.remaining -= n
        self.served += n
        return b"x" * n


# --- extension allow-list -------------------------------------------------------------


@pytest.mark.parametrize("name", ["a.wav", "A.WAV", "talk.mp3", "memo.M4a", "../../etc/x.wav"])
def test_supported_extensions_are_accepted(name):
    assert validated_extension(name) in {".wav", ".mp3", ".m4a"}


@pytest.mark.parametrize("name", [None, "", "notes.txt", "clip.ogg", "wav", "audio.wav.exe"])
def test_unsupported_extensions_are_rejected(name):
    with pytest.raises(UnsupportedFormat):
        validated_extension(name)


# --- bounded upload staging -----------------------------------------------------------


def test_save_upload_writes_to_a_generated_name(tmp_path):
    path = anyio.run(lambda: save_upload(ChunkedUpload(10), suffix=".wav", max_bytes=100, directory=tmp_path))
    assert path.parent == tmp_path and path.suffix == ".wav" and path.stat().st_size == 10


def test_save_upload_stops_at_the_limit_and_removes_the_partial_file(tmp_path):
    upload = ChunkedUpload(50 * 1024 * 1024)
    limit = 3 * 1024 * 1024
    with pytest.raises(FileTooLarge):
        anyio.run(lambda: save_upload(upload, suffix=".wav", max_bytes=limit, directory=tmp_path))
    # It gave up as soon as the limit was crossed rather than reading all 50 MiB.
    assert upload.served <= limit + 1024 * 1024
    assert list(tmp_path.iterdir()) == []


def test_save_upload_rejects_empty_files_and_cleans_up(tmp_path):
    with pytest.raises(EmptyFile):
        anyio.run(lambda: save_upload(ChunkedUpload(0), suffix=".wav", max_bytes=100, directory=tmp_path))
    assert list(tmp_path.iterdir()) == []


# --- real ffprobe/ffmpeg validation ---------------------------------------------------


@requires_ffmpeg
@pytest.mark.parametrize(
    ("filename", "codec_args"),
    [("tone.wav", ()), ("tone.mp3", ("-c:a", "libmp3lame")), ("tone.m4a", ("-c:a", "aac"))],
)
def test_supported_formats_pass_and_report_duration(tmp_path, filename, codec_args):
    try:
        path = generate_audio(tmp_path / filename, seconds=2.0, codec_args=codec_args)
    except subprocess.CalledProcessError:
        pytest.skip(f"this FFmpeg build cannot encode {filename}")
    info = inspect_audio(path, **LIMITS)
    assert info.duration_seconds == pytest.approx(2.0, abs=0.15)


@requires_ffmpeg
@pytest.mark.parametrize("payload", [b"this is not audio " * 200, bytes(range(256)) * 64])
@pytest.mark.parametrize("extension", [".wav", ".mp3", ".m4a"])
def test_garbage_with_a_valid_extension_is_rejected(tmp_path, payload, extension):
    path = tmp_path / f"fake{extension}"
    path.write_bytes(payload)
    with pytest.raises(InvalidAudio):
        inspect_audio(path, **LIMITS)


@requires_ffmpeg
def test_unsupported_container_behind_a_valid_extension_is_rejected(tmp_path):
    # Really FLAC, merely named .wav: decodable audio, but not a container we accept.
    flac = generate_audio(tmp_path / "tone.flac")
    disguised = flac.rename(tmp_path / "tone.wav")
    with pytest.raises(UnsupportedFormat):
        inspect_audio(disguised, **LIMITS)


@requires_ffmpeg
def test_audio_longer_than_the_limit_is_rejected(tmp_path):
    path = generate_audio(tmp_path / "tone.wav", seconds=2.0)
    with pytest.raises(AudioTooLong):
        inspect_audio(path, max_duration_seconds=1.0, timeout_seconds=30.0)


# --- mocked probe results -------------------------------------------------------------


def _fake_run(probe: dict, decode_stdout: bytes = b"", decode_returncode: int = 0):
    def run(command, timeout_seconds):
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, json.dumps(probe).encode(), b"")
        return subprocess.CompletedProcess(command, decode_returncode, decode_stdout, b"")

    return run


def _probe(duration: str | None = "10.0", fmt: str = "wav", codec_type: str = "audio") -> dict:
    return {
        "format": {"format_name": fmt, **({"duration": duration} if duration else {})},
        "streams": [{"codec_type": codec_type, "codec_name": "pcm_s16le"}],
    }


def test_over_duration_header_is_rejected_before_decoding(monkeypatch):
    commands = []

    def run(command, timeout_seconds):
        commands.append(command[0])
        return _fake_run(_probe(duration="301.0"))(command, timeout_seconds)

    monkeypatch.setattr(audio, "_run", run)
    with pytest.raises(AudioTooLong):
        inspect_audio(Path("x.wav"), **LIMITS)
    assert commands == ["ffprobe"]


def test_forged_short_header_is_caught_by_the_decoded_duration(monkeypatch):
    # Header claims 10 s, but decoding produced 301 s of audio.
    monkeypatch.setattr(audio, "_run", _fake_run(_probe("10.0"), b"out_time_us=301000000\n"))
    with pytest.raises(AudioTooLong):
        inspect_audio(Path("x.wav"), **LIMITS)


def test_decoded_duration_is_preferred_over_the_header(monkeypatch):
    monkeypatch.setattr(
        audio, "_run", _fake_run(_probe("10.0"), b"out_time_us=N/A\nout_time_us=12500000\nprogress=end\n")
    )
    assert inspect_audio(Path("x.wav"), **LIMITS).duration_seconds == 12.5


def test_file_without_an_audio_stream_is_rejected(monkeypatch):
    monkeypatch.setattr(audio, "_run", _fake_run(_probe(codec_type="video")))
    with pytest.raises(InvalidAudio):
        inspect_audio(Path("x.m4a"), **LIMITS)


def test_decode_failure_is_rejected(monkeypatch):
    monkeypatch.setattr(audio, "_run", _fake_run(_probe(), decode_returncode=1))
    with pytest.raises(InvalidAudio):
        inspect_audio(Path("x.wav"), **LIMITS)


def test_unknown_duration_is_rejected(monkeypatch):
    monkeypatch.setattr(audio, "_run", _fake_run(_probe(duration=None)))
    with pytest.raises(InvalidAudio):
        inspect_audio(Path("x.wav"), **LIMITS)


def test_probe_timeout_becomes_a_client_error(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(InvalidAudio):
        inspect_audio(Path("x.wav"), **LIMITS)


def test_missing_ffmpeg_is_reported_as_backend_unavailable(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(BackendUnavailable):
        inspect_audio(Path("x.wav"), **LIMITS)
