"""Shared fixtures. Nothing here loads a speech model or touches the network.

Tiny audio fixtures are synthesized with FFmpeg (a sine tone), so no recordings are needed.
"""

import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.providers.base import ProviderResult, RawSegment

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")

DEFAULT_RESULT = ProviderResult(
    segments=[
        RawSegment(start=0.0, end=1.5, text=" Hello   world."),
        RawSegment(start=1.5, end=3.0, text=" This is a test. "),
    ],
    language="en",
)


class FakeProvider:
    """Stands in for the speech model. Records calls and peak concurrency."""

    model_name = "fake-model"

    def __init__(
        self,
        result: ProviderResult = DEFAULT_RESULT,
        error: Exception | None = None,
        delay: float = 0.0,
        describe_file: bool = False,
    ) -> None:
        self.result = result
        self.error = error
        self.delay = delay
        self.describe_file = describe_file
        self.calls: list[tuple[Path, str | None]] = []
        self.max_active = 0
        self._active = 0
        self._lock = threading.Lock()

    def transcribe(self, audio_path: Path, language: str | None) -> ProviderResult:
        with self._lock:
            self.calls.append((audio_path, language))
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.error:
                raise self.error
            if self.describe_file:
                # Output derived from the staged file, so mixed-up files would be visible.
                size = audio_path.stat().st_size
                return ProviderResult([RawSegment(0.0, 1.0, f"bytes {size}")], "en")
            return self.result
        finally:
            with self._lock:
                self._active -= 1


def generate_audio(path: Path, seconds: float = 1.0, codec_args: tuple[str, ...] = ()) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={seconds}", "-ac", "1", "-ar", "16000",
            *codec_args, str(path),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return path


@pytest.fixture(scope="session")
def audio_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("audio")


@pytest.fixture(scope="session")
def wav_bytes(audio_dir: Path) -> bytes:
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not installed")
    return generate_audio(audio_dir / "tone.wav").read_bytes()


@pytest.fixture
def upload_dir(tmp_path: Path) -> Path:
    return tmp_path / "uploads"


@pytest.fixture
def make_settings(upload_dir: Path) -> Callable[..., Settings]:
    def _make(**overrides: object) -> Settings:
        return Settings(_env_file=None, temp_dir=upload_dir, **overrides)

    return _make


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def client(make_settings: Callable[..., Settings], provider: FakeProvider) -> Iterator[TestClient]:
    with TestClient(create_app(make_settings(), provider), raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def make_client(make_settings: Callable[..., Settings]) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def _make(provider: FakeProvider | None = None, inspector=None, **overrides: object) -> TestClient:
        app = create_app(make_settings(**overrides), provider or FakeProvider(), inspector)
        c = TestClient(app, raise_server_exceptions=False)
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)
