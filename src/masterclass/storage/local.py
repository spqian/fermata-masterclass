from __future__ import annotations

from pathlib import Path
import shutil
from typing import Iterable

from .base import ObjectStorage


class LocalObjectStorage(ObjectStorage):
    """Local filesystem backend that preserves the future ADLS key layout."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if "\\" in key:
            raise ValueError("storage keys must use '/' separators")
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"unsafe storage key: {key}")
        path = (self.root / key).resolve()
        if self.root not in path.parents and path != self.root:
            raise ValueError(f"storage key escapes root: {key}")
        return path

    def resolve_local_path(self, key: str) -> Path:
        """Return the local backing path for CLI-only workflows such as serving a player."""

        return self._path(key)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def read_to_file(self, key: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(key), target)

    def write_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        del content_type
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def write_file(self, key: str, source: Path, *, content_type: str | None = None) -> None:
        del content_type
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)

    def list_keys(self, prefix: str) -> Iterable[str]:
        base = self._path(prefix)
        if not base.exists():
            return []
        return (
            str(path.relative_to(self.root)).replace("\\", "/")
            for path in base.rglob("*")
            if path.is_file()
        )
