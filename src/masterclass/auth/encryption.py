from __future__ import annotations

import logging
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

_LOG = logging.getLogger(__name__)
_ENV_NAME = "MASTERCLASS_KEY_ENCRYPTION_KEY"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def ensure_key_encryption_key() -> str:
    """Return the Fernet key, creating a local-dev .env entry when safe."""

    key = (os.environ.get(_ENV_NAME) or "").strip()
    if key:
        return key
    if _truthy(os.environ.get("MASTERCLASS_PRODUCTION")):
        raise RuntimeError(
            f"{_ENV_NAME} is required when MASTERCLASS_PRODUCTION=true; "
            "generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )

    key = Fernet.generate_key().decode("ascii")
    os.environ[_ENV_NAME] = key
    env_file = _project_root() / ".env"
    existing = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    if _ENV_NAME not in {line.partition("=")[0].strip() for line in existing.splitlines() if "=" in line}:
        prefix = "" if not existing or existing.endswith("\n") else "\n"
        env_file.write_text(f"{existing}{prefix}{_ENV_NAME}={key}\n", encoding="utf-8")
    _LOG.warning("%s was missing; generated a local development key and wrote it to %s", _ENV_NAME, env_file)
    return key


class UserKeyCipher:
    def __init__(self, key: str | None = None) -> None:
        self._fernet = Fernet((key or ensure_key_encryption_key()).encode("ascii"))

    def encrypt(self, plaintext: str) -> str:
        value = plaintext.strip()
        if not value:
            raise ValueError("Gemini API key cannot be empty")
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> str:
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("Stored Gemini API key could not be decrypted; check MASTERCLASS_KEY_ENCRYPTION_KEY") from exc
