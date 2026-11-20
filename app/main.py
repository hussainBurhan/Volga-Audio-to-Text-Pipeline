"""Application factory and wiring."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from app.providers.base import TranscriptionProvider
from app.providers.whisper_provider import WhisperProvider
from app.services.transcription import AudioInspector, TranscriptionService

logger = logging.getLogger(__name__)

# Room for multipart boundaries and the small form fields around the file itself.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024


def create_app(
    settings: Settings | None = None,
    provider: TranscriptionProvider | None = None,
    inspector: AudioInspector | None = None,
) -> FastAPI:
    """Build the app. Tests pass ``provider`` (and optionally ``inspector``) to avoid the model."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    service_kwargs = {"inspector": inspector} if inspector else {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if provider is None:
            # Load the model once per process, off the event loop. If loading fails the
            # process still starts: /health reports model_loaded=false and uploads get 503.
            try:
                loaded = await run_in_threadpool(WhisperProvider.from_settings, settings)
                app.state.service = TranscriptionService(settings, loaded, **service_kwargs)
                logger.info("speech model loaded model=%s", loaded.model_name)
            except Exception:
                logger.exception("speech model failed to load")
        yield

    app = FastAPI(
        title="Audio Transcription Pipeline",
        version="1.0.0",
        description="Upload short audio, get a structured, timestamped transcript.",
        lifespan=lifespan,
    )
    app.state.service = TranscriptionService(settings, provider, **service_kwargs)

    # Added last = outermost: the request ID exists before the body limit can reject.
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_body_bytes=settings.max_upload_bytes + _MULTIPART_OVERHEAD_BYTES,
    )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()
