"""API tests against the real app with a fake provider (no model, no network)."""

import asyncio
import uuid

import httpx
import pytest

from app.main import create_app
from app.providers.base import ProviderResult, RawSegment
from app.schemas.transcription import TranscriptionResponse
from app.services.audio import AudioInfo
from tests.conftest import FakeProvider, generate_audio

URL = "/api/v1/transcriptions"


def wav(content: bytes, name: str = "clip.wav") -> dict:
    return {"file": (name, content, "audio/wav")}


def assert_error(response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert set(error) == {"code", "message", "request_id"}
    assert error["code"] == code
    assert error["request_id"] == response.headers["X-Request-ID"]


def assert_no_temp_files(upload_dir) -> None:
    assert not upload_dir.exists() or list(upload_dir.iterdir()) == []


# --- health / docs --------------------------------------------------------------------


def test_health_reports_ok_and_model_state(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_openapi_docs_are_served(client):
    assert client.get("/docs").status_code == 200
    assert URL in client.get("/openapi.json").json()["paths"]


def test_unknown_route_uses_the_error_shape(client):
    assert_error(client.get("/nope"), 404, "NOT_FOUND")


# --- happy path -----------------------------------------------------------------------


def test_valid_upload_returns_schema_compliant_json(client, provider, wav_bytes, upload_dir):
    response = client.post(URL, files=wav(wav_bytes))

    assert response.status_code == 200, response.text
    body = response.json()
    TranscriptionResponse.model_validate(body)  # conforms to the published schema
    assert body["status"] == "completed"
    assert body["transcript"] == "Hello world. This is a test."
    assert body["word_count"] == 6
    assert body["language"] == "en"
    assert body["model"] == "fake-model"
    assert body["duration_seconds"] == pytest.approx(1.0, abs=0.1)
    assert body["processing_time_seconds"] >= 0
    assert body["segments"] == [
        {"start": 0.0, "end": 1.5, "text": "Hello world."},
        {"start": 1.5, "end": 3.0, "text": "This is a test."},
    ]
    uuid.UUID(body["id"])
    assert body["id"] == response.headers["X-Request-ID"]
    assert_no_temp_files(upload_dir)


def test_upload_is_staged_under_a_generated_name_not_the_client_filename(
    client, provider, wav_bytes, upload_dir
):
    response = client.post(URL, files=wav(wav_bytes, name="../../../etc/passwd.wav"))

    assert response.status_code == 200
    staged_path, _ = provider.calls[0]
    assert staged_path.parent == upload_dir
    assert "passwd" not in staged_path.name
    assert_no_temp_files(upload_dir)


def test_language_hint_is_normalized_and_passed_to_the_provider(client, provider, wav_bytes):
    response = client.post(URL, files=wav(wav_bytes), data={"language": " EN "})

    assert response.status_code == 200
    assert provider.calls[0][1] == "en"
    assert response.json()["language"] == "en"


def test_blank_language_hint_means_auto_detect(client, provider, wav_bytes):
    # Swagger UI submits an empty string for an untouched optional field.
    response = client.post(URL, files=wav(wav_bytes), data={"language": ""})

    assert response.status_code == 200
    assert provider.calls[0][1] is None


def test_no_speech_returns_a_completed_empty_result(make_client, wav_bytes):
    silent = FakeProvider(result=ProviderResult(segments=[], language="en"))
    response = make_client(silent).post(URL, files=wav(wav_bytes))

    assert response.status_code == 200
    body = response.json()
    assert (body["status"], body["transcript"], body["segments"], body["word_count"]) == (
        "completed", "", [], 0,
    )
    assert body["language"] is None


def test_out_of_order_provider_segments_are_sorted(make_client, wav_bytes):
    shuffled = FakeProvider(
        result=ProviderResult(
            [RawSegment(2.0, 3.0, "two"), RawSegment(0.0, 1.0, "zero"), RawSegment(1.0, 2.0, "one")],
            "en",
        )
    )
    body = make_client(shuffled).post(URL, files=wav(wav_bytes)).json()

    assert [s["start"] for s in body["segments"]] == [0.0, 1.0, 2.0]
    assert body["transcript"] == "zero one two"


# --- rejected input -------------------------------------------------------------------


def test_missing_file_is_rejected(client, provider):
    assert_error(client.post(URL, data={"language": "en"}), 400, "MISSING_FILE")
    assert provider.calls == []


def test_zero_byte_file_is_rejected(client, provider, upload_dir):
    assert_error(client.post(URL, files=wav(b"")), 400, "EMPTY_FILE")
    assert provider.calls == []
    assert_no_temp_files(upload_dir)


def test_unsupported_extension_is_rejected(client, provider, wav_bytes, upload_dir):
    response = client.post(URL, files={"file": ("notes.txt", wav_bytes, "audio/wav")})

    assert_error(response, 415, "UNSUPPORTED_AUDIO_FORMAT")
    assert provider.calls == []
    assert_no_temp_files(upload_dir)


def test_corrupt_file_with_a_valid_extension_is_rejected(client, provider, upload_dir):
    response = client.post(URL, files=wav(b"definitely not audio " * 100))

    assert_error(response, 400, "INVALID_AUDIO")
    assert provider.calls == []
    assert_no_temp_files(upload_dir)


def test_unsupported_container_behind_a_valid_extension_is_rejected(
    client, provider, tmp_path, upload_dir
):
    flac_bytes = generate_audio(tmp_path / "tone.flac").read_bytes()

    assert_error(client.post(URL, files=wav(flac_bytes)), 415, "UNSUPPORTED_AUDIO_FORMAT")
    assert provider.calls == []
    assert_no_temp_files(upload_dir)


def test_oversized_file_is_rejected_while_streaming(make_client, wav_bytes, upload_dir):
    provider = FakeProvider()
    client = make_client(provider, max_upload_bytes=1024)
    assert len(wav_bytes) > 1024

    assert_error(client.post(URL, files=wav(wav_bytes)), 413, "FILE_TOO_LARGE")
    assert provider.calls == []
    assert_no_temp_files(upload_dir)


def test_oversized_body_is_rejected_from_content_length(make_client, upload_dir):
    client = make_client(max_upload_bytes=1024)

    assert_error(client.post(URL, files=wav(b"x" * 200_000)), 413, "FILE_TOO_LARGE")
    assert_no_temp_files(upload_dir)


def test_oversized_chunked_body_without_content_length_is_rejected(make_client, upload_dir):
    client = make_client(max_upload_bytes=1024)
    boundary = "testboundary"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + b"x" * 200_000 + f"\r\n--{boundary}--\r\n".encode()

    def chunks():
        yield body

    response = client.post(
        URL,
        content=chunks(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    assert "content-length" not in response.request.headers
    assert_error(response, 413, "FILE_TOO_LARGE")
    assert_no_temp_files(upload_dir)


def test_over_duration_audio_is_rejected_via_mocked_probe(make_client, wav_bytes, upload_dir):
    def too_long(path, *, max_duration_seconds, timeout_seconds):
        from app.core.errors import AudioTooLong

        assert max_duration_seconds == 300.0
        raise AudioTooLong()

    provider = FakeProvider()
    response = make_client(provider, inspector=too_long).post(URL, files=wav(wav_bytes))

    assert_error(response, 422, "AUDIO_TOO_LONG")
    assert provider.calls == []
    assert_no_temp_files(upload_dir)


def test_over_duration_audio_is_rejected_with_real_probe(make_client, wav_bytes, upload_dir):
    client = make_client(max_audio_seconds=0.5)  # the fixture tone is 1 s long

    assert_error(client.post(URL, files=wav(wav_bytes)), 422, "AUDIO_TOO_LONG")
    assert_no_temp_files(upload_dir)


@pytest.mark.parametrize("language", ["xx", "english", "en-US", "../etc"])
def test_invalid_language_hint_is_rejected(client, provider, wav_bytes, language, upload_dir):
    response = client.post(URL, files=wav(wav_bytes), data={"language": language})

    assert_error(response, 422, "INVALID_LANGUAGE")
    assert provider.calls == []
    assert_no_temp_files(upload_dir)


# --- failure behavior -----------------------------------------------------------------


def test_provider_exception_gives_a_safe_error_and_cleans_up(make_client, wav_bytes, upload_dir):
    secret = "secret internal detail /models/private/path"
    response = make_client(FakeProvider(error=RuntimeError(secret))).post(URL, files=wav(wav_bytes))

    assert_error(response, 500, "TRANSCRIPTION_FAILED")
    assert secret not in response.text
    assert "Traceback" not in response.text
    assert_no_temp_files(upload_dir)


def test_unexpected_error_outside_the_provider_is_also_safe(make_client, wav_bytes, upload_dir):
    def broken_inspector(path, **kwargs):
        raise RuntimeError("secret inspector detail")

    response = make_client(inspector=broken_inspector).post(URL, files=wav(wav_bytes))

    assert_error(response, 500, "INTERNAL_ERROR")
    assert "secret" not in response.text
    assert_no_temp_files(upload_dir)


def test_model_not_loaded_returns_503_and_health_says_so(make_settings, wav_bytes):
    from fastapi.testclient import TestClient

    # No lifespan (no `with`), so no model load is attempted: the service has no provider.
    client = TestClient(create_app(make_settings(), provider=None))

    assert client.get("/health").json() == {"status": "ok", "model_loaded": False}
    assert_error(client.post(URL, files=wav(wav_bytes)), 503, "TRANSCRIPTION_UNAVAILABLE")


# --- concurrency ----------------------------------------------------------------------


def fake_inspector(path, **kwargs) -> AudioInfo:
    return AudioInfo(duration_seconds=float(path.stat().st_size), container="wav", codec="pcm")


async def post_many(app, payloads: list[bytes]) -> list[httpx.Response]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        return await asyncio.gather(*(http.post(URL, files=wav(p)) for p in payloads))


def test_concurrent_requests_do_not_cross_contaminate(make_settings, upload_dir):
    provider = FakeProvider(delay=0.02, describe_file=True)
    app = create_app(make_settings(max_concurrent_transcriptions=4), provider, fake_inspector)
    payloads = [b"x" * (100 + i) for i in range(8)]  # every upload has a unique size

    responses = asyncio.run(post_many(app, payloads))

    assert [r.status_code for r in responses] == [200] * 8
    bodies = [r.json() for r in responses]
    assert len({b["id"] for b in bodies}) == 8
    assert len({path for path, _ in provider.calls}) == 8
    for payload, body in zip(payloads, bodies):
        # Both the probe result and the transcript came from this request's own file.
        assert body["transcript"] == f"bytes {len(payload)}"
        assert body["duration_seconds"] == float(len(payload))
    assert_no_temp_files(upload_dir)


def test_inference_concurrency_is_capped(make_settings):
    provider = FakeProvider(delay=0.05)
    app = create_app(make_settings(max_concurrent_transcriptions=1), provider, fake_inspector)

    responses = asyncio.run(post_many(app, [b"x" * 100] * 4))

    assert [r.status_code for r in responses] == [200] * 4
    assert provider.max_active == 1


def test_busy_server_sheds_load_with_503(make_settings, upload_dir):
    provider = FakeProvider(delay=0.3)
    settings = make_settings(max_concurrent_transcriptions=1, queue_timeout_seconds=0.05)
    app = create_app(settings, provider, fake_inspector)

    responses = asyncio.run(post_many(app, [b"x" * 100] * 3))

    statuses = sorted(r.status_code for r in responses)
    assert statuses == [200, 503, 503]
    busy = next(r for r in responses if r.status_code == 503)
    assert busy.json()["error"]["code"] == "SERVER_BUSY"
    assert busy.headers["Retry-After"] == "5"
    assert_no_temp_files(upload_dir)


def test_health_stays_responsive_during_inference(make_settings):
    provider = FakeProvider(delay=0.5)
    app = create_app(make_settings(), provider, fake_inspector)

    async def scenario() -> float:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            job = asyncio.create_task(http.post(URL, files=wav(b"x" * 100)))
            await asyncio.sleep(0.1)  # inference is now running in a worker thread
            loop = asyncio.get_running_loop()
            started = loop.time()
            health = await http.get("/health")
            waited = loop.time() - started
            assert health.status_code == 200
            assert (await job).status_code == 200
            return waited

    assert asyncio.run(scenario()) < 0.25  # not blocked behind the 0.5 s inference
