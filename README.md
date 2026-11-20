# Audio-to-Text Transcription Pipeline

A small FastAPI service that accepts a short audio upload, transcribes it with a pretrained
speech-to-text model ([faster-whisper](https://github.com/SYSTRAN/faster-whisper)), and returns
structured JSON — transcript, timestamped segments, language, duration, word count — that
another service can display, search, or subtitle.

Built for the Volga Partners technical assessment.

## The assignment, and what is my choice

The brief asks for *"a simple transcription pipeline that converts audio input into text and
processes the result for downstream use"*, using any existing STT library or API, with the
focus on engineering decisions rather than model training.

Everything beyond that sentence is a design decision of mine, not a requirement from Volga
Partners: Python/FastAPI, faster-whisper, the REST contract, the supported formats, the
20 MiB / 300 s limits, Docker, and the test strategy.

**Deliberate scope:** one synchronous endpoint for short clips, stateless, no database,
no authentication, no frontend. A working, tested vertical slice rather than a platform.

## Quick start

Prerequisites:

- **Python 3.11+** (developed and tested on 3.12.5)
- **FFmpeg** with `ffprobe` on `PATH` (tested with 9.0.1) — `brew install ffmpeg` / `apt install ffmpeg`
- Internet access on the **first start only**: the `base` model (about 141 MB) is downloaded
  from the Hugging Face Hub into `~/.cache/huggingface` and reused afterwards.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8000
```

Then, in another terminal:

```bash
curl -s -F "file=@samples/manual-tests/01_valid_speech.wav" http://localhost:8000/api/v1/transcriptions
```

- Swagger UI: <http://localhost:8000/docs>
- Health: <http://localhost:8000/health>

Configuration is optional; every setting has a default. To change one, copy
[.env.example](.env.example) to `.env`.

### Real response

This is the actual output for `samples/manual-tests/01_valid_speech.wav` on my machine (not an illustration):

```json
{
  "id": "089741ea-ca87-479d-a67a-77cd914cf96a",
  "status": "completed",
  "transcript": "This is a short test recording for the audio transcription pipeline. We need to finalize the deployment by Friday. Please send the meeting notes to the team after lunch.",
  "language": "en",
  "duration_seconds": 9.961,
  "processing_time_seconds": 1.698,
  "word_count": 29,
  "model": "faster-whisper-base",
  "segments": [
    {"start": 0.0, "end": 4.0, "text": "This is a short test recording for the audio transcription pipeline."},
    {"start": 4.0, "end": 7.0, "text": "We need to finalize the deployment by Friday."},
    {"start": 7.0, "end": 10.0, "text": "Please send the meeting notes to the team after lunch."}
  ]
}
```

### Docker

```bash
docker build -t transcription-pipeline .
```

```bash
docker run --rm -p 8000:8000 -v whisper-cache:/home/app/.cache/huggingface transcription-pipeline
```

The image bundles FFmpeg. The named volume keeps the downloaded model between runs.
Verified on Docker 29.5 (x86_64): the image builds (about 1.5 GB, mostly the ML runtime), the
container starts, downloads the model, and transcribes `samples/manual-tests/01_valid_speech.wav` correctly.

## API

### `POST /api/v1/transcriptions`

`multipart/form-data` with:

| Field | Required | Description |
|---|---|---|
| `file` | yes | Audio file: `.wav`, `.mp3` or `.m4a` |
| `language` | no | Whisper language code such as `en`. Omit (or leave blank) to auto-detect. |

Returns `200` with the JSON shown above. Field notes:

- `transcript` is the segment texts joined with single spaces. The only text processing is
  whitespace trimming/collapsing — no paraphrasing, punctuation fixing, or summarizing.
- `segments` are sorted by start time with `0 <= start <= end`, in seconds, rounded to
  milliseconds. They are the model's own timestamps, so the last `end` can slightly exceed
  `duration_seconds` (visible above: 10.0 vs 9.961).
- `word_count` is a whitespace-delimited count, not a linguistic token count. It undercounts
  languages written without spaces (Chinese, Japanese, Thai).
- `duration_seconds` is measured by decoding the audio with FFmpeg.
- `processing_time_seconds` is server wall time from the parsed upload to the response:
  staging, validation, any wait for an inference slot, inference, and post-processing.
- `id` equals the `X-Request-ID` response header and appears in the server logs. It
  identifies this response only; **results are not stored and cannot be fetched later**.
- No confidence score is returned, because Whisper does not provide a calibrated one.
- **No speech:** if the file is valid audio but contains no detectable speech, the result is
  still `200 completed` with `transcript: ""`, `segments: []`, `word_count: 0`, and
  `language: null` (a language "detected" from silence would be noise).

### `GET /health`

`{"status": "ok", "model_loaded": true}`. This is a liveness check plus one honest fact:
whether a model is loaded in this process. It does not run a test inference. If the model
fails to load at startup the process stays up, `model_loaded` is `false`, and uploads get `503`.

### Errors

Every error has one shape:

```json
{"error": {"code": "UNSUPPORTED_AUDIO_FORMAT", "message": "Supported formats: WAV, MP3, M4A.", "request_id": "bcbcadb5-ff8b-4b12-aade-7140837c8d27"}}
```

| Status | Code | When |
|---|---|---|
| 400 | `MISSING_FILE` | No `file` part in the form |
| 400 | `EMPTY_FILE` | Zero-byte upload |
| 400 | `INVALID_AUDIO` | Not decodable as audio, or no audio stream |
| 413 | `FILE_TOO_LARGE` | Larger than `APP_MAX_UPLOAD_BYTES` (default 20 MiB) |
| 415 | `UNSUPPORTED_AUDIO_FORMAT` | Extension or actual container is not WAV/MP3/M4A |
| 422 | `INVALID_LANGUAGE` | Unknown language hint |
| 422 | `AUDIO_TOO_LONG` | Longer than `APP_MAX_AUDIO_SECONDS` (default 300 s) |
| 500 | `TRANSCRIPTION_FAILED` / `INTERNAL_ERROR` | Model or server failure; details go to logs only |
| 503 | `TRANSCRIPTION_UNAVAILABLE` | Model not loaded, or FFmpeg missing on the server |
| 503 | `SERVER_BUSY` | No inference slot became free in time (`Retry-After` header set) |

## Architecture

```text
client ── multipart upload ──▶ RequestContextMiddleware   request ID, one log line per request
                               BodySizeLimitMiddleware    hard cap on bytes received
                               api/routes.py              thin: parse, delegate, serialize
                               services/transcription.py  the pipeline (below)
                                 1. validate language hint and file extension
                                 2. services/audio.py     stream to generated temp file, counting bytes
                                 3. services/audio.py     ffprobe: container, audio stream, duration
                                                          ffmpeg: full decode, true duration
                                 4. providers/            transcribe in a worker thread,
                                                          behind a semaphore (default 1)
                                 5. services/postprocess.py  pure functions -> typed response
                                 finally: delete the temp file
```

| Module | Responsibility |
|---|---|
| [app/main.py](app/main.py) | App factory, model lifecycle, wiring |
| [app/api/routes.py](app/api/routes.py) | HTTP only — no business logic |
| [app/services/transcription.py](app/services/transcription.py) | Pipeline orchestration, concurrency limit, cleanup |
| [app/services/audio.py](app/services/audio.py) | Bounded upload staging, FFmpeg validation |
| [app/services/postprocess.py](app/services/postprocess.py) | Deterministic text/segment processing |
| [app/providers/base.py](app/providers/base.py) | `TranscriptionProvider` protocol |
| [app/providers/whisper_provider.py](app/providers/whisper_provider.py) | faster-whisper implementation |
| [app/core/](app/core) | Settings, typed errors and handlers, middleware, language codes |

### Design decisions

**Local faster-whisper instead of a hosted STT API.** A reviewer can run it with no account,
key, or cost, and audio never leaves the machine. The price is a model download, CPU-bound
latency, and owning the inference infrastructure. `base` is the default because it downloads
quickly and runs faster than real time on a laptop CPU; it is noticeably less accurate than
`small`/`medium` on noisy or accented speech. Model size, device and compute type are settings.
*Alternative considered:* a hosted API (OpenAI, Deepgram, AssemblyAI) — less code and better
accuracy per unit of effort, but adds a secret, a cost, and a network dependency to reviewing.

**A provider protocol.** The pipeline depends on a two-member interface
(`model_name`, `transcribe(path, language)`), not on Whisper. Swapping in a hosted API is one
new class; tests inject a fake through `create_app(provider=...)`, so the fast suite never
imports the ML stack. Cost: slightly more code than a single script.

**Synchronous endpoint for short clips.** The simplest contract to run, test and review: one
request, one response. It ties up an HTTP connection for the length of inference, which is why
the duration limit exists. Long audio needs the asynchronous design under
[Production evolution](#production-evolution).

**Model loaded once; inference off the event loop; explicit backpressure.** The model is
created once per process at startup. Inference is blocking and CPU-heavy, so it runs in a
worker thread — `/health` and uploads keep being served meanwhile (there is a test for this).
A semaphore limits concurrent inference (default 1, because parallel jobs on one CPU mostly
just slow each other down). A request waits up to `APP_QUEUE_TIMEOUT_SECONDS` for a slot and
then gets `503 SERVER_BUSY` with `Retry-After`, rather than queueing without bound. This is
honest load-shedding, not scalability: throughput is one clip at a time per process.

**Validate by decoding, not by trusting the client.** The extension must be on the allow-list
*and* `ffprobe` must report a supported container with an audio stream *and* `ffmpeg` must
decode it. `Content-Type` is ignored. Duration is checked twice: the header value for a cheap
early rejection, then the decoded duration, because headers can be wrong or forged (decoding
stops just past the limit, so a huge file cannot run unbounded). Cost: one extra decode pass
per request — about 125 ms measured for the 10 s sample (two FFmpeg process launches).

**Two size limits.** A middleware counts request-body bytes as they arrive and rejects with
`413` before the multipart parser spools anything; `Content-Length` is used only as an early
hint, since it is client-supplied and absent on chunked uploads. The pipeline then counts the
file's own bytes while copying it to the staging file.

**Stateless temp-file pipeline.** The upload is written to a `mkstemp`-generated name (the
client filename contributes only its allow-listed extension) and deleted in a single `finally`
block that covers success, validation errors and model failures. Nothing is persisted, so
there is no history and no retrieval by ID.

**Deterministic post-processing only.** Whitespace normalization, ordering, assembly and
counting are pure functions. The transcript is treated as evidence: the service never
rewrites what the model heard. It therefore does not fix recognition mistakes either.

**Voice-activity detection on by default.** Whisper tends to hallucinate short phrases on
silence. faster-whisper's bundled VAD drops non-speech audio first, which is what makes the
"no speech → empty result" policy hold in practice. It can clip very quiet speech; turn it off
with `APP_STT_VAD_FILTER=false`.

### Security and privacy

- Generated temp filenames; subprocesses use argument lists (no shell) with timeouts.
- Logs contain request ID, method, path, status and timings — never transcript text, original
  filenames, or audio bytes. Server-side tracebacks are logged; clients get a generic message.
- No audio or transcript is retained by the application. Two unavoidable caches exist: the
  model weights in the Hugging Face cache, and Starlette's own spooled copy of the upload,
  which it deletes when the request ends.
- **Not implemented: authentication, authorization, rate limiting, TLS.** These are
  prerequisites before exposing this service publicly or sending it sensitive recordings.

## Testing

### Fast suite — no model, no network

```bash
pytest
```

72 tests, about 4 seconds. They need `ffmpeg`/`ffprobe` (tiny sine-tone fixtures are
synthesized on the fly; those tests skip if FFmpeg is absent). Coverage by case:

| # | Case | Where |
|---|---|---|
| 1 | Valid upload → schema-compliant JSON | `test_api.py::test_valid_upload_returns_schema_compliant_json` |
| 2 | Missing / zero-byte file | `test_missing_file_is_rejected`, `test_zero_byte_file_is_rejected` |
| 3 | Unsupported type/container | `test_unsupported_extension_is_rejected`, `test_unsupported_container_behind_a_valid_extension_is_rejected` |
| 4 | Corrupt file with valid extension | `test_corrupt_file_with_a_valid_extension_is_rejected`, `test_audio.py::test_garbage_with_a_valid_extension_is_rejected` |
| 5 | Oversized upload, streaming limit | `test_oversized_*` (three paths), `test_audio.py::test_save_upload_stops_at_the_limit_and_removes_the_partial_file` |
| 6 | Over-duration via mocked probe | `test_over_duration_audio_is_rejected_via_mocked_probe`, plus real-probe and forged-header variants |
| 7 | No speech → empty result | `test_no_speech_returns_a_completed_empty_result` |
| 8 | Segment ordering, assembly | `test_postprocess.py`, `test_out_of_order_provider_segments_are_sorted` |
| 9 | Normalization does not paraphrase | `test_normalize_changes_nothing_but_whitespace` |
| 10 | Invalid language hint | `test_invalid_language_hint_is_rejected` |
| 11 | Provider exception → safe error + cleanup | `test_provider_exception_gives_a_safe_error_and_cleans_up` |
| 12 | Concurrent requests isolated | `test_concurrent_requests_do_not_cross_contaminate`, `test_inference_concurrency_is_capped` |
| 13 | `/health` | `test_health_reports_ok_and_model_state` |

Every rejection test also asserts that the staging directory is empty afterwards.

### Real-model smoke test

```bash
pytest -m integration
```

Loads the real `base` model and sends `samples/manual-tests/01_valid_speech.wav` and a generated silent file through
the full HTTP pipeline. Assertions are loose (language, key words, timestamp sanity) because
exact wording can vary between model versions.

### What was actually measured

Environment: laptop with an Intel Core i7-9750H (12 logical cores), macOS 14.6.1, Python 3.12.5,
FFmpeg 9.0.1, faster-whisper 1.2.1, model `base`, CPU, `int8`, beam size 5, VAD on.

| Check | Result |
|---|---|
| Fast suite | 72 passed |
| Integration suite | 2 passed (speech sample; silence → empty result) |
| `01_valid_speech.wav` via `curl` against a running server | 9.96 s audio → 1.70 s processing |
| Same clip as MP3 (with `language=en`) | 1.09 s processing, identical transcript |
| Same clip as M4A | 1.61 s processing, identical transcript |
| Validation step alone (ffprobe + decode) on `01_valid_speech.wav` | about 125 ms |
| Corrupt `.wav`, `.txt` upload, bad language, no file, 25 MiB file | 400 / 415 / 422 / 400 / 413, as documented |
| Staged uploads left in the temp directory afterwards | 0 |
| Clean copy: `python3 -m venv`, `pip install -r requirements-dev.txt`, both suites | pass |
| Docker image: build, run, transcribe `01_valid_speech.wav` | pass (2.09 s processing in the container) |

**On accuracy:** the transcript of the sample matched its reference text
([samples/manual-tests/01_valid_speech.txt](samples/manual-tests/01_valid_speech.txt)) word for word. That is one clip of clean synthetic
speech (generated with the macOS `say` command), so it demonstrates that the integration works —
it is **not** an accuracy measurement. No WER benchmark on natural, noisy, or non-English speech
has been run, and I make no accuracy or language-coverage claim. Latency figures are three
single requests on one laptop, not a benchmark.

## Known limitations

- One clip at a time per process; clips limited to 20 MiB / 300 s by default.
- The client waits for the whole transcription; a dropped connection does not cancel inference.
- No persistence, retrieval by ID, authentication, or rate limiting.
- Segment-level timestamps only (no word-level timestamps, no speaker diarization).
- The upload briefly exists on disk twice (the framework's spooled copy and the staging file).
- `GET /health` does not prove the model can transcribe, only that it loaded.
- The language allow-list is the Whisper set; a different provider would need its own.
- Dependency note: faster-whisper 1.2.1 is incompatible with PyAV 19, so `requirements.txt`
  pins `av<17` (16.1.0 verified).
- P1 items not built: `.txt`/`.srt` export, WER benchmark.

## Production evolution

For long recordings or real concurrency, the synchronous endpoint becomes a job system. This
is described, not built:

1. The client uploads directly to object storage with short-lived, authenticated credentials.
2. The API records a job in a database and returns `202 Accepted` with a job ID.
3. A durable queue feeds workers whose concurrency matches the hardware (one job per GPU, say).
4. Workers decode, split long audio into overlapping windows, transcribe, restore global
   timestamps, and persist the structured result.
5. The client polls `GET /transcriptions/{id}` or receives a webhook.
6. Around that: retries with idempotent jobs, a dead-letter queue, retention and deletion
   policy, access control, quotas, metrics, and cost tracking.

The pieces here that carry over unchanged are the provider protocol, the validation module,
and the post-processing functions; the route handler and the in-process semaphore are what get
replaced.
