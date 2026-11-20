"""Real-model smoke test. Opt-in: `pytest -m integration`.

Loads faster-whisper (downloading weights on first run) and transcribes the speech sample
through the full HTTP pipeline. Assertions are deliberately loose: exact wording can vary
between model versions and decoding settings.
"""

import subprocess
import time
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


def test_long_audio_is_transcribed_as_a_background_job(client, tmp_path):
    # ~6 minutes: the sample looped 36 times. Too long for the synchronous endpoint.
    long_audio = tmp_path / "long.mp3"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-stream_loop", "35", "-i", str(SAMPLE), str(long_audio)],
        check=True, stdin=subprocess.DEVNULL,
    )
    files = {"file": ("long.mp3", long_audio.read_bytes(), "audio/mpeg")}

    too_long = client.post("/api/v1/transcriptions", files=files)
    assert too_long.status_code == 422
    assert too_long.json()["error"]["code"] == "AUDIO_TOO_LONG"

    accepted = client.post("/api/v1/transcription-jobs", files=files)
    assert accepted.status_code == 202, accepted.text
    url = accepted.headers["Location"]

    deadline = time.monotonic() + 600
    while (job := client.get(url).json())["status"] in ("queued", "processing"):
        assert time.monotonic() < deadline, "job timed out"
        assert client.get("/health").status_code == 200  # server stays responsive
        time.sleep(1)

    assert job["status"] == "completed", job
    result = job["result"]
    assert result["duration_seconds"] > 300
    # Timestamps are global across the model's 30-second windows, and stay in order.
    starts = [s["start"] for s in result["segments"]]
    assert starts == sorted(starts)
    assert result["segments"][-1]["end"] > 300
    assert result["word_count"] > 800  # 36 repetitions of a 29-word passage
    print(f"\nlong audio: {result['duration_seconds']}s audio, "
          f"{result['processing_time_seconds']}s processing, "
          f"{len(result['segments'])} segments, {result['word_count']} words")
