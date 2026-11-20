"""Real-model smoke test. Opt-in: `pytest -m integration`.

Loads faster-whisper (downloading weights on first run) and transcribes the speech sample
through the full HTTP pipeline. Assertions are deliberately loose: exact wording can vary
between model versions and decoding settings.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.conftest import generate_audio, requires_ffmpeg

pytestmark = [pytest.mark.integration, requires_ffmpeg]

SAMPLE = Path(__file__).parent.parent / "samples" / "manual-tests" / "01_valid_speech.wav"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = Settings(_env_file=None, temp_dir=tmp_path_factory.mktemp("uploads"))
    with TestClient(create_app(settings)) as c:  # lifespan loads the real model
        assert c.get("/health").json()["model_loaded"] is True
        yield c


def test_real_speech_sample_is_transcribed(client):
    response = client.post(
        "/api/v1/transcriptions", files={"file": ("sample.wav", SAMPLE.read_bytes(), "audio/wav")}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["language"] == "en"
    assert body["model"] == "faster-whisper-base"
    assert body["word_count"] > 10
    assert body["segments"] and all(0 <= s["start"] <= s["end"] for s in body["segments"])
    assert body["segments"][-1]["end"] <= body["duration_seconds"] + 1.0
    transcript = body["transcript"].lower()
    for keyword in ("transcription", "pipeline", "friday"):
        assert keyword in transcript


def test_silence_returns_an_empty_completed_result(client, tmp_path):
    silence = tmp_path / "silence.wav"
    generate_audio(silence, seconds=3.0, codec_args=("-af", "volume=0"))

    response = client.post(
        "/api/v1/transcriptions", files={"file": ("silence.wav", silence.read_bytes(), "audio/wav")}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["transcript"], body["segments"], body["word_count"]) == ("", [], 0)
    assert body["language"] is None
