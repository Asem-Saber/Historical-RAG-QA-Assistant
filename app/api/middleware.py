import contextvars
import logging
import uuid

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


class RequestIDMiddleware:
    """Pure ASGI middleware that assigns a correlation ID to every HTTP request.

    - Reuses an incoming ``X-Request-ID`` header when present (cross-service tracing).
    - Stores the ID in a :class:`contextvars.ContextVar` so every logger in the
      call-stack can include it automatically via :class:`RequestIDFilter`.
    - Echoes the ID back in the ``X-Request-ID`` response header.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        incoming = headers.get(b"x-request-id", b"").decode()
        rid = incoming or uuid.uuid4().hex[:16]

        token = request_id_ctx.set(rid)

        async def send_with_rid(message):
            if message["type"] == "http.response.start":
                raw_headers = list(message.get("headers", []))
                raw_headers.append((b"x-request-id", rid.encode()))
                message["headers"] = raw_headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_rid)
        finally:
            request_id_ctx.reset(token)


class RequestIDFilter(logging.Filter):
    """Injects ``request_id`` into every log record so ``%(request_id)s``
    works in the formatter string."""

    def filter(self, record):
        record.request_id = request_id_ctx.get()
        return True
