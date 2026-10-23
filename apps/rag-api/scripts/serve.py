"""Local RAG API launcher.

On Windows, psycopg's async driver cannot run on the default ProactorEventLoop;
select the selector policy before uvicorn creates the loop. Production should use
the company deployment entrypoint instead of this development helper.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def selector_loop_factory(use_subprocess: bool = False) -> asyncio.AbstractEventLoop:
    """Return a selector loop; psycopg async rejects Proactor on Windows."""

    return asyncio.SelectorEventLoop()


import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        loop="scripts.serve:selector_loop_factory",
    )
