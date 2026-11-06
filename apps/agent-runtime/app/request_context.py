"""Per-request correlation identifiers.

The browser or an upstream proxy may supply ``X-Request-ID``; otherwise the
server mints one. The value is echoed on the response, kept in a context
variable for the duration of the request, and written into every structured
log line produced by :func:`app.metrics.log`.

FastAPI is imported lazily so that installing the package without the ``web``
extra (CLI-only use) keeps working.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI, Request

REQUEST_ID_HEADER = "X-Request-ID"

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def current_request_id() -> str:
    """Return the correlation id of the request being handled, or ``""``."""
    return _request_id.get()


def install_request_id(app: FastAPI) -> None:
    """Register the correlation middleware on ``app``.

    Registered last so it becomes the outermost HTTP middleware: even requests
    rejected by the local-origin guard still receive an id.
    """

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        incoming = (request.headers.get(REQUEST_ID_HEADER) or "").strip()
        value = incoming[:128] if incoming else uuid4().hex
        token = _request_id.set(value)
        try:
            response = await call_next(request)
        finally:
            _request_id.reset(token)
        response.headers[REQUEST_ID_HEADER] = value
        return response
