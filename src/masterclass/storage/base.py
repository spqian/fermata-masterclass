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

    def read_json(self, key: str) -> Any:
        return json.loads(self.read_bytes(key).decode("utf-8"))

    def write_json(self, key: str, data: Any) -> None:
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.write_bytes(key, payload, content_type="application/json")
