from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_LOG = logging.getLogger(__name__)
_ENV_NAME = "MASTERCLASS_KEY_ENCRYPTION_KEY"
_AESGCM_PREFIX = "gcm1:"  # versioned wire format: gcm1:<b64url(nonce|ciphertext+tag)>


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
    """Encrypts per-user secrets (Gemini API keys) with AES-256-GCM.

    The owner's stable id (``google_sub``) is passed as Associated Authenticated
    Data so a malicious admin with filesystem access cannot swap two users'
    encrypted blobs without the AEAD tag failing to verify.

    Legacy Fernet ciphertexts (written before this AAD upgrade) are still
    accepted on read for backwards compatibility; they should be rewritten on
    the next ``encrypt()`` call.
    """

    def __init__(self, key: str | None = None) -> None:
        raw = (key or ensure_key_encryption_key()).encode("ascii")
        self._fernet = Fernet(raw)
        # Fernet keys are 32 raw bytes encoded as urlsafe-b64; reuse those 32 bytes for AES-GCM.
        try:
            self._aes_key = base64.urlsafe_b64decode(raw)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"{_ENV_NAME} must be a Fernet-compatible urlsafe-base64 32-byte key") from exc
        if len(self._aes_key) != 32:
            raise ValueError(f"{_ENV_NAME} must decode to 32 bytes for AES-256-GCM")
        self._aesgcm = AESGCM(self._aes_key)

    def encrypt(self, plaintext: str, aad: str) -> str:
        value = plaintext.strip()
        if not value:
            raise ValueError("Gemini API key cannot be empty")
        if not aad:
            raise ValueError("encryption AAD (e.g. google_sub) is required")
        nonce = os.urandom(12)
        ct = self._aesgcm.encrypt(nonce, value.encode("utf-8"), aad.encode("utf-8"))
        blob = base64.urlsafe_b64encode(nonce + ct).decode("ascii")
        return f"{_AESGCM_PREFIX}{blob}"

    def decrypt(self, token: str, aad: str) -> str:
        if not aad:
            raise ValueError("decryption AAD (e.g. google_sub) is required")
        if token.startswith(_AESGCM_PREFIX):
            try:
                raw = base64.urlsafe_b64decode(token[len(_AESGCM_PREFIX):].encode("ascii"))
                if len(raw) < 13:
                    raise ValueError("ciphertext too short")
                nonce, ct = raw[:12], raw[12:]
                return self._aesgcm.decrypt(nonce, ct, aad.encode("utf-8")).decode("utf-8")
            except (InvalidTag, ValueError) as exc:
                # InvalidTag => either tampering or wrong AAD/user; surface as ValueError so
                # callers can return a stable error without leaking which case it was.
                raise ValueError(
                    "Stored Gemini API key could not be decrypted; check MASTERCLASS_KEY_ENCRYPTION_KEY "
                    "or whether the profile file was moved between users"
                ) from exc
        # Legacy Fernet ciphertext (no AAD).
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("Stored Gemini API key could not be decrypted; check MASTERCLASS_KEY_ENCRYPTION_KEY") from exc
