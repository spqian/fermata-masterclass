from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

from masterclass.auth.encryption import UserKeyCipher
from masterclass.storage.base import ObjectStorage

PreferredModel = Literal["gemini-2.5-pro", "gemini-2.5-flash"]
DEFAULT_MODEL: PreferredModel = "gemini-2.5-pro"
VALID_MODELS: set[str] = {"gemini-2.5-pro", "gemini-2.5-flash"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_google_sub(google_sub: str) -> str:
    value = (google_sub or "").strip()
    if not value or "/" in value or "\\" in value or ":" in value or ".." in value:
        raise ValueError("invalid Google subject")
    return value


@dataclass
class UserProfile:
    google_sub: str
    email: str
    display_name: str
    preferred_model: PreferredModel = DEFAULT_MODEL
    encrypted_gemini_key: str | None = None
    gemini_key_set_at: str | None = None
    created_at: str = ""
    last_login_at: str = ""

    @classmethod
    def from_json(cls, data: dict) -> "UserProfile":
        model = data.get("preferred_model") or DEFAULT_MODEL
        if model not in VALID_MODELS:
            model = DEFAULT_MODEL
        now = _now()
        return cls(
            google_sub=str(data.get("google_sub") or data.get("sub") or ""),
            email=str(data.get("email") or ""),
            display_name=str(data.get("display_name") or data.get("name") or ""),
            preferred_model=model,  # type: ignore[arg-type]
            encrypted_gemini_key=data.get("encrypted_gemini_key"),
            gemini_key_set_at=data.get("gemini_key_set_at"),
            created_at=str(data.get("created_at") or now),
            last_login_at=str(data.get("last_login_at") or now),
        )

    def to_json(self) -> dict:
        return asdict(self)

    def public_json(self) -> dict:
        return {
            "google_sub": self.google_sub,
            "email": self.email,
            "name": self.display_name,
            "display_name": self.display_name,
            "has_gemini_key": bool(self.encrypted_gemini_key),
            "gemini_key_set_at": self.gemini_key_set_at,
            "preferred_model": self.preferred_model,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
        }


class UserProfileStore:
    def __init__(self, storage: ObjectStorage, cipher: UserKeyCipher | None = None) -> None:
        self.storage = storage
        self._cipher = cipher

    def _key(self, google_sub: str) -> str:
        return f"user_profiles/{_safe_google_sub(google_sub)}.json"

    @property
    def cipher(self) -> UserKeyCipher:
        if self._cipher is None:
            self._cipher = UserKeyCipher()
        return self._cipher

    def load(self, google_sub: str) -> UserProfile:
        return UserProfile.from_json(self.storage.read_json(self._key(google_sub)))

    def upsert(self, profile: UserProfile) -> UserProfile:
        _safe_google_sub(profile.google_sub)
        if not profile.created_at:
            profile.created_at = _now()
        if not profile.last_login_at:
            profile.last_login_at = profile.created_at
        self.storage.write_json(self._key(profile.google_sub), profile.to_json())
        return profile

    def upsert_oauth_user(self, *, google_sub: str, email: str, display_name: str) -> UserProfile:
        now = _now()
        try:
            profile = self.load(google_sub)
            profile.email = email
            profile.display_name = display_name
            profile.last_login_at = now
        except FileNotFoundError:
            profile = UserProfile(
                google_sub=_safe_google_sub(google_sub),
                email=email,
                display_name=display_name,
                created_at=now,
                last_login_at=now,
            )
        return self.upsert(profile)

    def set_preferred_model(self, google_sub: str, model: str) -> UserProfile:
        if model not in VALID_MODELS:
            raise ValueError("preferred_model must be gemini-2.5-pro or gemini-2.5-flash")
        profile = self.load(google_sub)
        profile.preferred_model = model  # type: ignore[assignment]
        return self.upsert(profile)

    def set_gemini_key(self, google_sub: str, plaintext: str) -> UserProfile:
        profile = self.load(google_sub)
        profile.encrypted_gemini_key = self.cipher.encrypt(plaintext)
        profile.gemini_key_set_at = _now()
        return self.upsert(profile)

    def clear_gemini_key(self, google_sub: str) -> UserProfile:
        profile = self.load(google_sub)
        profile.encrypted_gemini_key = None
        profile.gemini_key_set_at = None
        return self.upsert(profile)

    def get_gemini_key_plain(self, google_sub: str) -> str | None:
        profile = self.load(google_sub)
        if not profile.encrypted_gemini_key:
            return None
        return self.cipher.decrypt(profile.encrypted_gemini_key)
