"""Asynchronous transcription jobs (the long-audio path), with a fake provider."""

import time

import pytest

from app.core.errors import JobNotFound, ServerBusy
from app.schemas.transcription import TranscriptionResponse
from app.services.audio import AudioInfo
from app.services.jobs import JobStore
from tests.conftest import FakeProvider

JOBS_URL = "/api/v1/transcription-jobs"


def wav(content: bytes, name: str = "clip.wav") -> dict:
    return {"file": (name, content, "audio/wav")}


def wait_for_job(client, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"{JOBS_URL}/{job_id}").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.01)
    raise AssertionError(f"job did not finish: {body}")


def fake_inspector(path, **kwargs) -> AudioInfo:
    return AudioInfo(duration_seconds=1800.0, container="wav", codec="pcm")


def assert_no_temp_files(upload_dir) -> None:
    assert not upload_dir.exists() or list(upload_dir.iterdir()) == []


# --- API ------------------------------------------------------------------------------


def test_job_is_accepted_then_completes_with_the_standard_result(client, wav_bytes, upload_dir):
    response = client.post(JOBS_URL, files=wav(wav_bytes), data={"language": "en"})

    assert response.status_code == 202, response.text
    accepted = response.json()
    assert accepted["status"] in ("queued", "processing")
    assert accepted["result"] is None and accepted["error"] is None
    assert response.headers["Location"] == f"{JOBS_URL}/{accepted['id']}"

    done = wait_for_job(client, accepted["id"])
    assert done["status"] == "completed" and done["error"] is None
    result = TranscriptionResponse.model_validate(done["result"])
    assert result.id == accepted["id"]
    assert result.transcript == "Hello world. This is a test."
    assert_no_temp_files(upload_dir)


def test_job_returns_before_inference_finishes(make_client, wav_bytes):
    client = make_client(FakeProvider(delay=0.4))

    started = time.monotonic()
    response = client.post(JOBS_URL, files=wav(wav_bytes))
    accepted_after = time.monotonic() - started

    assert response.status_code == 202
    assert accepted_after < 0.3  # did not wait for the 0.4 s "inference"
    assert client.get(f"{JOBS_URL}/{response.json()['id']}").json()["status"] == "processing"
    assert wait_for_job(client, response.json()["id"])["status"] == "completed"


def test_each_path_enforces_its_own_duration_limit(make_client):
    limits = []

    def recording_inspector(path, *, max_duration_seconds, timeout_seconds):
        limits.append(max_duration_seconds)
        return AudioInfo(duration_seconds=1800.0, container="wav", codec="pcm")

    client = make_client(inspector=recording_inspector)
    client.post("/api/v1/transcriptions", files=wav(b"x" * 100))
    job = client.post(JOBS_URL, files=wav(b"x" * 100)).json()
    wait_for_job(client, job["id"])

    assert limits == [300.0, 3600.0]  # 5 minutes synchronous, 1 hour as a job


def test_job_path_has_its_own_larger_upload_limit(make_client):
    client = make_client(inspector=fake_inspector, max_upload_bytes=1024, max_job_upload_bytes=500_000)
    payload = b"x" * 200_000

    assert client.post("/api/v1/transcriptions", files=wav(payload)).status_code == 413
    assert client.post(JOBS_URL, files=wav(payload)).status_code == 202
    assert client.post(JOBS_URL, files=wav(b"x" * 600_000)).status_code == 413


@pytest.mark.parametrize(
    ("files", "data", "status", "code"),
    [
        ({"file": ("notes.txt", b"hello", "text/plain")}, {}, 415, "UNSUPPORTED_AUDIO_FORMAT"),
        ({"file": ("a.wav", b"not audio " * 50, "audio/wav")}, {}, 400, "INVALID_AUDIO"),
        ({"file": ("a.wav", b"", "audio/wav")}, {}, 400, "EMPTY_FILE"),
        ({"file": ("a.wav", b"x", "audio/wav")}, {"language": "klingon"}, 422, "INVALID_LANGUAGE"),
    ],
)
def test_invalid_input_fails_immediately_and_creates_no_job(
    make_client, upload_dir, files, data, status, code
):
    client = make_client(max_active_jobs=1)

    response = client.post(JOBS_URL, files=files, data=data)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert_no_temp_files(upload_dir)
    # The rejected request did not use up the only job slot.
    assert client.post(JOBS_URL, files=wav(b"x"), data={"language": "xx"}).status_code == 422


def test_provider_failure_marks_the_job_failed_safely(make_client, wav_bytes, upload_dir):
    client = make_client(FakeProvider(error=RuntimeError("secret internal detail")))

    job_id = client.post(JOBS_URL, files=wav(wav_bytes)).json()["id"]
    done = wait_for_job(client, job_id)

    assert done["status"] == "failed" and done["result"] is None
    assert done["error"] == {
        "code": "TRANSCRIPTION_FAILED",
        "message": "Transcription failed unexpectedly.",
        "request_id": job_id,
    }
    assert_no_temp_files(upload_dir)


def test_jobs_queue_for_the_single_inference_slot(make_client, wav_bytes):
    provider = FakeProvider(delay=0.15)
    client = make_client(provider, inspector=fake_inspector)

    ids = [client.post(JOBS_URL, files=wav(b"x" * 10)).json()["id"] for _ in range(3)]
    statuses = [client.get(f"{JOBS_URL}/{i}").json()["status"] for i in ids]

    assert statuses == ["processing", "queued", "queued"]
    assert [wait_for_job(client, i)["status"] for i in ids] == ["completed"] * 3
    assert provider.max_active == 1


def test_too_many_active_jobs_are_refused(make_client, upload_dir):
    client = make_client(FakeProvider(delay=0.2), inspector=fake_inspector, max_active_jobs=2)

    first, second = (client.post(JOBS_URL, files=wav(b"x" * 10)) for _ in range(2))
    third = client.post(JOBS_URL, files=wav(b"x" * 10))

    assert (first.status_code, second.status_code) == (202, 202)
    assert third.status_code == 503
    assert third.json()["error"]["code"] == "SERVER_BUSY"
    assert "Retry-After" in third.headers
    for response in (first, second):
        wait_for_job(client, response.json()["id"])
    assert client.post(JOBS_URL, files=wav(b"x" * 10)).status_code == 202  # capacity is back


def test_unknown_job_is_404(client):
    response = client.get(f"{JOBS_URL}/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "JOB_NOT_FOUND"


# --- store ----------------------------------------------------------------------------


def _result(job_id: str) -> TranscriptionResponse:
    return TranscriptionResponse(
        id=job_id, transcript="", language=None, duration_seconds=1.0,
        processing_time_seconds=0.1, word_count=0, model="fake", segments=[],
    )


def test_finished_jobs_expire_after_the_ttl_but_active_ones_do_not():
    now = [0.0]
    store = JobStore(max_active=5, ttl_seconds=60, clock=lambda: now[0])
    finished, running = store.create("finished"), store.create("running")
    store.complete(finished, _result("finished"))

    now[0] = 59
    assert store.get("finished").status == "completed"
    now[0] = 61
    with pytest.raises(JobNotFound):
        store.get("finished")
    assert store.get("running") is running


def test_only_active_jobs_count_towards_capacity():
    store = JobStore(max_active=1, ttl_seconds=60)
    job = store.create("a")
    with pytest.raises(ServerBusy):
        store.create("b")
    store.fail(job, "X", "failed")
    store.create("b")
