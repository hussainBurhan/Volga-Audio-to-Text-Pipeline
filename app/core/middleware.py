"""Pure ASGI middleware: request IDs + access logging, and a hard request-body cap."""

import logging
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.errors import FileTooLarge, error_response

logger = logging.getLogger("app.access")


class RequestContextMiddleware:
    """Assigns a server-generated request ID and logs one line per request.

    Logs carry the ID, method, path, status and elapsed time only — never filenames,
    transcript text, or payload bytes.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            logger.info(
                "request_id=%s method=%s path=%s status=%d elapsed_ms=%.1f",
                request_id,
                scope["method"],
                scope["path"],
                status_code,
                (time.perf_counter() - started) * 1000,
            )


class BodySizeLimitMiddleware:
    """Rejects request bodies larger than ``max_body_bytes`` with 413.

    ``Content-Length`` gives a cheap early rejection, but it is client-supplied and absent
    for chunked uploads, so the bytes actually received are counted as well. This runs
    before the multipart parser spools anything to disk.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdigit() and int(declared) > self.max_body_bytes:
            request_id = scope.get("state", {}).get("request_id", "")
            response = error_response(
                request_id, FileTooLarge.status_code, FileTooLarge.code, FileTooLarge.message
            )
            await response(scope, receive, send)
            return

        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    # Raised inside the route's body parsing; FastAPI re-raises
                    # HTTPExceptions untouched, and our handler renders the 413.
                    raise HTTPException(
                        status_code=FileTooLarge.status_code, detail=FileTooLarge.message
                    )
            return message

        await self.app(scope, counting_receive, send)
