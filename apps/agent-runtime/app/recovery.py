"""Startup scan for sessions whose last turn never closed.

A crashed process leaves the append-only event log with ``turns > ends``.
This module only *detects and reports* those sessions; it never replays a
turn on its own, because the interrupted run may have produced file edits or
command side effects that a silent retry would duplicate.

The recovery contract is therefore:

* detect   → :func:`scan_interrupted`
* report   → caller logs ``recovery.interrupted_sessions`` and exposes the
             list (health endpoint / conversation list)
* decide   → the user explicitly starts a new turn on that conversation

Replay primitives already exist on :class:`app.session.JsonlSessionStore`:
``load_events`` / ``load_messages`` rebuild the conversation, and
``repair_tool_pairing`` closes dangling tool calls before the next turn.

Only files named ``session.jsonl`` are considered. The runtime workspace also
holds ``events.jsonl`` / ``judgments.jsonl`` / ``runs.jsonl`` with unrelated
schemas; a corrupt or foreign file is skipped, never fatal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .metrics import log
from .session import JsonlSessionStore

SESSION_FILENAME = "session.jsonl"


@dataclass(slots=True, frozen=True)
class InterruptedSession:
    """A session file whose last turn started but never ended."""

    path: Path
    conversation_id: str
    turns: int
    steps: int
    message_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "conversation_id": self.conversation_id,
            "path": str(self.path),
            "turns": self.turns,
            "steps": self.steps,
            "messages": self.message_count,
        }


def _session_files(directory: Path) -> list[Path]:
    """Session logs one or two levels deep: ``<c>/session.jsonl`` or the root."""

    found = {*directory.glob(SESSION_FILENAME), *directory.glob(f"*/{SESSION_FILENAME}")}
    return sorted(found)


async def scan_interrupted(sessions_dir: str | Path) -> list[InterruptedSession]:
    """Return every session under ``sessions_dir`` whose last turn is still open.

    The scan is read-only: nothing is repaired, truncated, or replayed here.
    Files that are not valid session logs are ignored.
    """

    directory = Path(sessions_dir)
    if not directory.is_dir():
        return []

    interrupted: list[InterruptedSession] = []
    for path in _session_files(directory):
        try:
            timeline = await JsonlSessionStore(path).load_timeline()
        except Exception as exc:  # partial write, foreign schema, busy file
            log(
                "recovery.session_unreadable",
                conversation_id=path.parent.name,
                error=type(exc).__name__,
            )
            continue
        if not timeline.last_turn_open:
            continue
        interrupted.append(
            InterruptedSession(
                path=path,
                conversation_id=path.parent.name,
                turns=timeline.turns,
                steps=timeline.steps,
                message_count=len(timeline.messages),
            )
        )
    return interrupted


async def report_interrupted(sessions_dir: str | Path) -> list[InterruptedSession]:
    """Scan and emit one structured log line. Never raises on a bad file."""

    try:
        interrupted = await scan_interrupted(sessions_dir)
    except Exception as exc:  # a corrupt directory must not block startup
        log("recovery.scan_failed", error=type(exc).__name__, detail=str(exc)[:200])
        return []

    if interrupted:
        log(
            "recovery.interrupted_sessions",
            count=len(interrupted),
            conversation_ids=[item.conversation_id for item in interrupted][:20],
            action="registered_only",
        )
    else:
        log("recovery.no_interrupted_sessions")
    return interrupted
