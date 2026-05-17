"""Per-conversation locking for chat append safety.

A drill background task can race a user chat turn to append to the same
``cmt_<id>.json`` file. Each path acquires the lock keyed on the
conversation's storage path before doing ``load -> append -> save`` so we
don't lose messages to a last-writer-wins overwrite.

In-process only; this is fine because the whole API runs in one process
today and the drill workers are threads.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


_REGISTRY_LOCK = threading.Lock()
_LOCKS: dict[str, threading.Lock] = {}


def _lock_for(key: str) -> threading.Lock:
    with _REGISTRY_LOCK:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


@contextmanager
def conversation_lock(key: str) -> Iterator[None]:
    """Acquire the lock for one conversation file path.

    Re-entrant across threads it isn't holding; **not** reentrant for the
    same thread (don't nest calls with the same key).
    """
    if not key:
        raise ValueError("conversation_lock requires a non-empty key")
    lock = _lock_for(key)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
