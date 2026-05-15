# BYO Gemini API key

Fermata Masterclass uses a LibreChat-style bring-your-own-key model: each signed-in user supplies a Google AI Studio Gemini API key in **Settings**. Lesson generation, score-prep fallback, and web-backed MIDI search use that user's key when available.

## Why BYO keys

Google One / AI Pro / Ultra consumer subscriptions cannot be delegated to third-party API calls. Vertex AI OAuth would require broad `cloud-platform` scopes and a full GCP billing setup for every musician. A pasted AI Studio key is the least invasive, most common pattern.

## Get a key

1. Go to <https://aistudio.google.com/apikey>.
2. Create or choose a project and create an API key.
3. Paste the key into Fermata **Settings**.

For Gemini 2.5 Pro, enable billing/prepay at <https://aistudio.google.com/billing>. Free-tier accounts are generally better suited to Gemini 2.5 Flash.

## Models

- **Gemini 2.5 Pro** — default, better teacher quality, paid. Typical lesson estimate: about $0.15–$0.50 depending on media length and tool calls.
- **Gemini 2.5 Flash** — cheaper/free-tier friendly, weaker teacher quality. Typical lesson estimate: about $0.02–$0.08.

The model preference is stored on the user profile and applied to new lesson teacher runs.

## Encryption at rest

Keys are stored per user in `user_profiles/{google_sub}.json` under the configured object-storage root. The plaintext key is never written to disk. The `encrypted_gemini_key` field is a Fernet token encrypted with `MASTERCLASS_KEY_ENCRYPTION_KEY`.

Generate a key with:

```powershell
tools\python\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Local development auto-generates the encryption key into `.env` if it is missing. Production (`MASTERCLASS_PRODUCTION=true`) refuses to start without it.

## Server fallback

By default, lesson LLM calls require a per-user key. For local development only, set `ALLOW_SERVER_DEFAULT_KEY=true` and `GEMINI_API_KEY=...` to allow users without a saved key to use the server default key.
