from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base import ObjectStorage


class AdlsObjectStorage(ObjectStorage):
    """Azure Data Lake Storage Gen2 implementation of ObjectStorage.

    This backend intentionally works with logical keys only. Authentication is
    delegated to Azure SDK credentials so production can use Managed Identity
    and local development can use DefaultAzureCredential.
    """

    def __init__(self, *, account_url: str, file_system: str, credential=None) -> None:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.storage.filedatalake import DataLakeServiceClient
        except ImportError as exc:
            raise RuntimeError("Install Azure dependencies with: pip install -e .[azure]") from exc

        self.account_url = account_url
        self.file_system_name = file_system
        self.credential = credential or DefaultAzureCredential()
        self._service = DataLakeServiceClient(account_url=account_url, credential=self.credential)
        self._fs = self._service.get_file_system_client(file_system=file_system)

    @staticmethod
    def _validate_key(key: str) -> str:
        if "\\" in key:
            raise ValueError("storage keys must use '/' separators")
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"unsafe storage key: {key}")
        return key

    @staticmethod
    def _translate_not_found(exc: Exception, key: str) -> Exception:
        """Map Azure SDK ResourceNotFoundError to the stdlib FileNotFoundError
        that the rest of the app expects from any ObjectStorage backend.

        Without this every consumer that does
            try: storage.read_*(key)
            except FileNotFoundError: ...create-on-miss...
        explodes with a 500 in cloud because Azure raises a different type
        than LocalObjectStorage. Keeping the boundary uniform here saves
        every caller a try/except in two flavors.
        """
        from azure.core.exceptions import ResourceNotFoundError
        if isinstance(exc, ResourceNotFoundError):
            return FileNotFoundError(key)
        return exc

    def exists(self, key: str) -> bool:
        key = self._validate_key(key)
        return self._fs.get_file_client(key).exists()

    def read_bytes(self, key: str) -> bytes:
        key = self._validate_key(key)
        try:
            return self._fs.get_file_client(key).download_file().readall()
        except Exception as exc:
            raise self._translate_not_found(exc, key) from exc

    def read_to_file(self, key: str, path: Path) -> None:
        key = self._validate_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            downloader = self._fs.get_file_client(key).download_file()
        except Exception as exc:
            raise self._translate_not_found(exc, key) from exc
        with path.open("wb") as handle:
            downloader.readinto(handle)

    def write_bytes(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        key = self._validate_key(key)
        file_client = self._fs.get_file_client(key)
        content_settings = None
        if content_type:
            try:
                from azure.storage.filedatalake import ContentSettings
                content_settings = ContentSettings(content_type=content_type)
            except ImportError:
                content_settings = None
        file_client.upload_data(data, overwrite=True, content_settings=content_settings)

    def write_file(self, key: str, path: Path, *, content_type: str | None = None) -> None:
        key = self._validate_key(key)
        file_client = self._fs.get_file_client(key)
        content_settings = None
        if content_type:
            try:
                from azure.storage.filedatalake import ContentSettings
                content_settings = ContentSettings(content_type=content_type)
            except ImportError:
                content_settings = None
        with path.open("rb") as handle:
            file_client.upload_data(handle, overwrite=True, content_settings=content_settings)

    def list_keys(self, prefix: str) -> Iterable[str]:
        prefix = self._validate_key(prefix.rstrip("/"))
        for path in self._fs.get_paths(path=prefix, recursive=True):
            if not path.is_directory:
                yield str(path.name)

