from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from urllib.parse import urlencode

from itsdangerous import BadSignature, URLSafeSerializer

from masterclass.auth.encryption import ensure_key_encryption_key

COOKIE_NAME = "fermata_session"
OAUTH_COOKIE_NAME = "fermata_oauth"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_SCOPES = "openid email profile"


@dataclass(frozen=True)
class GoogleIdentity:
    google_sub: str
    email: str
    display_name: str


def _secret() -> str:
    return (os.environ.get("MASTERCLASS_SESSION_SECRET") or ensure_key_encryption_key()).strip()


def _serializer(salt: str) -> URLSafeSerializer:
    return URLSafeSerializer(_secret(), salt=salt)


def _is_localhost(hostname: str | None) -> bool:
    return (hostname or "").lower() in {"127.0.0.1", "localhost", "::1"}


def cookie_secure_for_request(request) -> bool:
    return not _is_localhost(getattr(request.url, "hostname", None))


def sign_session_user_id(user_id: str) -> str:
    return _serializer("fermata-session").dumps({"user_id": user_id})


def get_session_user_id(request) -> str | None:
    raw = request.cookies.get(COOKIE_NAME)
    if not raw:
        return None
    try:
        payload = _serializer("fermata-session").loads(raw)
    except BadSignature:
        return None
    user_id = (payload or {}).get("user_id")
    return str(user_id).strip() or None


def set_session_cookie(response, request, user_id: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        sign_session_user_id(user_id),
        httponly=True,
        secure=cookie_secure_for_request(request),
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
        path="/",
    )


def clear_session_cookie(response, request) -> None:
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, secure=cookie_secure_for_request(request), samesite="lax")


def oauth_configured() -> bool:
    return bool(os.environ.get("GOOGLE_OAUTH_CLIENT_ID") and os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"))


def build_oauth_client():
    """Build the Authlib Starlette OAuth client registry for Google."""

    try:
        from authlib.integrations.starlette_client import OAuth
    except ImportError as exc:
        raise RuntimeError("Install API auth dependencies with: pip install -e .[api]") from exc

    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=os.environ.get("GOOGLE_OAUTH_CLIENT_ID"),
        client_secret=os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": GOOGLE_SCOPES},
    )
    return oauth


def friendly_oauth_missing_html() -> str:
    return """<!doctype html><html><head><title>Google OAuth not configured</title></head>
<body style='font-family:system-ui;margin:3rem;max-width:760px;line-height:1.5'>
<h1>Google Sign-In is not configured</h1>
<p>Server admin must configure Google OAuth credentials before users can sign in.</p>
<ol>
<li>Create an OAuth client in Google Cloud Console.</li>
<li>Set <code>GOOGLE_OAUTH_CLIENT_ID</code>, <code>GOOGLE_OAUTH_CLIENT_SECRET</code>, and <code>GOOGLE_OAUTH_REDIRECT_URI</code>.</li>
<li>Restart the Fermata Masterclass API.</li>
</ol>
<p><a href='/'>Return home</a></p>
</body></html>"""


def _same_origin_path(next_url: str | None) -> str:
    value = (next_url or "/").strip() or "/"
    if not value.startswith("/") or value.startswith("//") or "\\" in value:
        return "/"
    return value


def _code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _id_token_nonce(id_token: str) -> str | None:
    """Extract the ``nonce`` claim from an OIDC ID token (JWS compact form).

    The token already arrived over TLS straight from Google's token endpoint,
    so we trust the transport for authenticity and only need the claim value
    to compare against the cookie-stored nonce. We do not re-verify the
    signature here.
    """
    try:
        parts = id_token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1]
        padding = "=" * (-len(payload_b64) % 4)
        payload = base64.urlsafe_b64decode(payload_b64 + padding)
        claims = json.loads(payload.decode("utf-8"))
        nonce = claims.get("nonce")
        return str(nonce) if nonce else None
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def build_login_redirect(request, next_url: str | None = None):
    from fastapi.responses import HTMLResponse, RedirectResponse

    if not oauth_configured():
        return HTMLResponse(friendly_oauth_missing_html(), status_code=503)

    # Instantiate Authlib's Starlette client so configuration follows the documented pattern.
    build_oauth_client()

    redirect_uri = os.environ.get("GOOGLE_OAUTH_REDIRECT_URI") or str(request.url_for("auth_callback"))
    verifier = secrets.token_urlsafe(64)
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(24)
    oauth_state = _serializer("fermata-oauth").dumps({
        "state": state,
        "code_verifier": verifier,
        "next": _same_origin_path(next_url),
        "nonce": nonce,
    })
    params = {
        "client_id": os.environ["GOOGLE_OAUTH_CLIENT_ID"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "state": state,
        "nonce": nonce,
        "code_challenge": _code_challenge(verifier),
        "code_challenge_method": "S256",
        "access_type": "online",
        "prompt": "select_account",
    }
    response = RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}", status_code=302)
    response.set_cookie(
        OAUTH_COOKIE_NAME,
        oauth_state,
        httponly=True,
        secure=cookie_secure_for_request(request),
        samesite="lax",
        max_age=600,
        path="/",
    )
    return response


async def complete_google_callback(request) -> tuple[GoogleIdentity, str]:
    from fastapi import HTTPException
    from authlib.integrations.httpx_client import AsyncOAuth2Client

    error = request.query_params.get("error")
    if error:
        raise HTTPException(status_code=400, detail=f"Google sign-in failed: {error}")
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Google sign-in callback missing code or state")
    raw = request.cookies.get(OAUTH_COOKIE_NAME)
    if not raw:
        raise HTTPException(status_code=400, detail="Google sign-in state cookie expired; try again")
    try:
        oauth_state = _serializer("fermata-oauth").loads(raw)
    except BadSignature as exc:
        raise HTTPException(status_code=400, detail="Google sign-in state cookie is invalid; try again") from exc
    if oauth_state.get("state") != state:
        raise HTTPException(status_code=400, detail="Google sign-in state mismatch; try again")

    redirect_uri = os.environ.get("GOOGLE_OAUTH_REDIRECT_URI") or str(request.url_for("auth_callback"))
    async with AsyncOAuth2Client(
        client_id=os.environ.get("GOOGLE_OAUTH_CLIENT_ID"),
        client_secret=os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET"),
        scope=GOOGLE_SCOPES,
    ) as client:
        token = await client.fetch_token(
            GOOGLE_TOKEN_URL,
            code=code,
            redirect_uri=redirect_uri,
            code_verifier=oauth_state.get("code_verifier"),
        )
        # OpenID Connect: bind this callback to the original /auth/login by
        # verifying the nonce echoed inside Google's signed ID token. Combined
        # with the state cookie + PKCE verifier this closes the replay window.
        expected_nonce = oauth_state.get("nonce")
        if expected_nonce:
            id_token = token.get("id_token") if isinstance(token, dict) else None
            if not id_token or _id_token_nonce(id_token) != expected_nonce:
                raise HTTPException(status_code=400, detail="Google sign-in nonce mismatch; try again")
        resp = await client.get(GOOGLE_USERINFO_URL)
        resp.raise_for_status()
        info = resp.json()

    sub = str(info.get("sub") or "").strip()
    email = str(info.get("email") or "").strip()
    name = str(info.get("name") or email or "").strip()
    if not sub or not email:
        raise HTTPException(status_code=400, detail="Google did not return an account id and email")
    return GoogleIdentity(google_sub=sub, email=email, display_name=name), _same_origin_path(oauth_state.get("next"))
