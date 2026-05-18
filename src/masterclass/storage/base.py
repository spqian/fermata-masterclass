from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable


class ObjectStorage(ABC):
    """Blob/file storage abstraction.

    Keys are logical ADLS-style keys using "/" separators. Implementations must
    reject path traversal and must not expose arbitrary host paths to callers.
    """

    @abstractmethod
    def exists(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def read_to_file(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.read_bytes(key))

    @abstractmethod
    def write_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        raise NotImplementedError

    def write_file(self, key: str, path: Path, *, content_type: str | None = None) -> None:
        self.write_bytes(key, path.read_bytes(), content_type=content_type)

    @abstractmethod
    def list_keys(self, prefix: str) -> Iterable[str]:
        raise NotImplementedError

    def delete_key(self, key: str) -> bool:
        """Delete a single object. Returns True if it existed and was removed,
        False if it didn't exist. Other errors propagate.

        Default implementation: list_keys + best-effort. Subclasses should
        override with the native delete operation."""
        raise NotImplementedError

    def delete_prefix(self, prefix: str) -> int:
        """Recursively delete every object under ``prefix``. Returns the count
        of keys actually deleted. Used by lesson/masterclass/drill cascades.

        Default implementation iterates list_keys and deletes one by one.
        Subclasses can override for atomicity / efficiency."""
        deleted = 0
        # Materialise the list first so iteration isn't disturbed by deletes.
        keys = list(self.list_keys(prefix))
        for key in keys:
            try:
                if self.delete_key(key):
                    deleted += 1
            except FileNotFoundError:
                # Concurrent delete or already gone; ignore.
                continue
        return deleted

    def read_json(self, key: str) -> Any:
        # ADLS upload (create+append+flush) has a brief window where the file
        # exists but is empty. Concurrent reads during a pipeline-stage save
        # used to crash callers with JSONDecodeError("Expecting value", ..., 0).
        # Retry up to 3 times with short backoff if we get empty bytes.
        import time as _time
        for attempt in range(3):
            data = self.read_bytes(key)
            if data:
                return json.loads(data.decode("utf-8"))
            if attempt < 2:
                _time.sleep(0.1 * (attempt + 1))
        # Empty after retries: surface the read so callers see the real key.
        raise json.JSONDecodeError("Expecting value (file was empty after retries)", "", 0)

    def write_json(self, key: str, data: Any) -> None:
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.write_bytes(key, payload, content_type="application/json")
