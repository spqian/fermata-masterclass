# Roadmap

This is the next-up roadmap for Fermata Masterclass after the initial v2 productionization. Sequenced for dependency order; the first three can be tackled independently of the fourth.

| Sprint | Feature | Why it's next |
|---|---|---|
| **A** | Google Sign-In auth + per-user profiles | Foundation. Replaces the trust-the-header model and unlocks BYO key + chat. |
| **B** | Bring-your-own Gemini API key | Each user pays for their own usage. Required before any kind of public hosting. |
| **C** | Chat with the teacher (in lesson context) | The differentiating feature. Mostly wiring on top of the existing agentic loop. |
| **D** | Hosting on Azure | Last. Build everything else against `localhost`, then ship. |

Sprints A + B are tightly coupled (B depends on A). Sprint C is independent and can run in parallel. Sprint D is a separate phase.

---

## Sprint A — Google Sign-In auth

Today, every API endpoint trusts an `X-User-Id` header (`apps/api/main.py:tenant_from_header`). That's fine for a single-developer-on-localhost setup; it's a non-starter for anything public.

**Approach**

- OAuth 2.0 Authorization Code + PKCE flow against Google
- Scopes: `openid email profile` only — **never** request `cloud-platform` or any sensitive scope
- New endpoints: `GET /auth/login`, `GET /auth/callback`, `POST /auth/logout`, `GET /auth/me`
- Session: `httponly` `samesite=lax` signed cookie (itsdangerous), cookie content is just the user_id (Google `sub` claim)
- The existing `X-User-Id` header path **stays** so scripts and CI keep working — the new dependency accepts a session cookie OR a header
- `tenant_id == user_id == google_sub` for now (single-tenant per user)
- New module: `src/masterclass/auth/google_oauth.py`
- New persistent store: `user_profiles/{google_sub}.json` via the existing storage abstraction

**Out of scope for v1**: roles, multi-tenant teams, email/password fallback, account deletion (manual for now).

## Sprint B — Bring your own Gemini API key

**Why this matters.** Every lesson costs the operator (today: the developer running the server) ~$0.15–0.50 in Gemini API charges. If anyone ever shares a deploy URL with friends, those costs accrue to whoever pays the API bill. The fix is to make each user supply their own key.

**Why not OAuth-based billing delegation?**

The research is unambiguous:

1. **Google One / AI Pro / Ultra subscriptions** ($20/mo, $250/mo) provide consumer access to Gemini in the chatbot, Workspace apps, and AI Studio's web UI. They do **not** provide any developer API access, OAuth flow, or billing-delegation mechanism. The billing accounts for consumer subscriptions and developer APIs are entirely separate inside Google.
2. **Vertex AI** does support OAuth 2.0, but the required scope is `cloud-platform` — read/write/delete on **all** the user's Google Cloud resources. No reasonable user should grant that to a music app. It also requires the user to have set up a GCP project with billing enabled and the Vertex AI API enabled.
3. **Every comparable open-source AI tool** (LibreChat, Cline, Aider, Continue.dev, Open WebUI) uses the same pattern: the user pastes their API key into the app's settings. None of them implement OAuth-based billing delegation, because the option doesn't really exist.

So the plan is the standard pattern.

**Approach**

- Settings page where the user pastes their Google AI Studio API key
- Server stores the key Fernet-encrypted on the user profile (key from `MASTERCLASS_KEY_ENCRYPTION_KEY` env)
- 3-step onboarding deep-linking [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → enable billing ($10 minimum prepay required for Gemini 2.5 Pro; the free tier only supports 2.5 Flash) → paste key
- Optional model toggle in settings: Gemini 2.5 Pro (paid, ~$0.15–0.50/lesson) vs Gemini 2.5 Flash (free tier, ~$0.02–0.08/lesson, weaker reasoning)
- `LlmProvider` already accepts an `api_key` parameter — just thread it through from the user profile
- Server policy: refuse to operate without a per-user key UNLESS `ALLOW_SERVER_DEFAULT_KEY=true` (off in production, on locally for development)

## Sprint C — Chat with the teacher (in lesson context)

After a lesson is critiqued, let the user ask follow-up questions. The teacher answers in-context, with access to the same tools (re-listen to a passage, re-watch a video clip, inspect a bar's measurements).

The hard parts already exist: the agentic loop, the tool registry, the system-instruction templating, the score map, the audio Files API integration. This sprint is mostly wiring.

**Architecture**

- New endpoint: `POST /lessons/{session_id}/chat` (and `GET` / `DELETE` for history)
- Persistent conversations at `sessions/{session_id}/chat/{conversation_id}.json`
- Reuses `system_instruction_for_profile()` and `default_tool_registry()` with a chat-mode addendum that constrains the teacher to the current lesson and the original critique

**Guardrails (non-negotiable)**

The teacher's tools call the LLM, which costs money the user is paying for. So:

| Limit | Default | Why |
|---|---|---|
| Per-user-per-day turns | 50 | Stops runaway loops / abuse |
| Per-conversation turns | 20 | Conversations beyond this rarely add value |
| Per-message size | 2 KB | Music questions don't need novels |
| Tool calls per turn | 5 | Lower than the 15 the initial lesson gets |
| Topic guard | Gemini 2.5 Flash pre-check ("is this about music performance?") | Off-topic questions get a polite redirect without invoking expensive Pro |
| Cost meter | Per response | Show the user what they spent |

UI lives in a collapsible panel in the player's right column. No streaming for v1.

## Sprint D — Hosting on Azure

The codebase is already Azure-shaped: `storage/adls.py` uses `DefaultAzureCredential`, `toolchain/ffmpeg.py` does `shutil.which` fallback for Linux, `pyproject.toml` has an `[azure]` extra. The remaining work is packaging and provisioning.

**Recommended architecture**

```
GitHub repo → GitHub Actions (OIDC, no secrets) → ACR Basic
                                                      ↓
                                          Azure Container Apps
                                          (Consumption, min-replicas=1)
                                                      ↓
                                          ┌──────────┴──────────┐
                              Managed identity        Managed identity
                                          ↓                        ↓
                                   ADLS Gen2                  Key Vault
                                   (HNS=true)                 (Standard)
                                                                   ↓
                                                          GEMINI_API_KEY
                                                          (server default,
                                                           opt-in only)
```

**Why Container Apps and not the alternatives**

| Service | Why we're not using it |
|---|---|
| App Service Linux | Charges full instance rate 24×7. The workload is bursty (long-running lesson pipeline, then idle). Container Apps Consumption charges ~75% less when idle. |
| Azure Functions | Wrong shape. The app is ASGI with long-running background threads spawned per lesson; the Functions programming model fights this. |
| Azure Container Instances | No built-in HTTP routing, TLS, or auto-restart. Designed for batch jobs. |
| AKS | Massive overkill for a single-process app expecting tens of concurrent users. $150+/mo just for the system node pool. |

**What needs to be built**

- Multi-stage `Dockerfile` (`python:3.11-slim-bookworm` + `apt install ffmpeg openjdk-21-jre-headless` + `COPY` Audiveris JAR)
- Linux `install_tools.sh` sibling to the existing PowerShell installer
- Verify Audiveris JAR discovery on Linux (the ZIP layout differs from the Windows MSI)
- Wire env vars: `MASTERCLASS_STORAGE_BACKEND=adls`, `MASTERCLASS_ADLS_ACCOUNT_URL`, `MASTERCLASS_ADLS_FILE_SYSTEM`
- `.github/workflows/deploy.yml` using `azure/login@v2` (OIDC) + `azure/container-apps-deploy-action@v1`
- Custom domain + Container Apps managed TLS cert (free, auto-renewed)

**Estimated cost** (East US region, 500 lessons/month, 2 vCPU / 4 GiB):

| Component | Monthly |
|---|---|
| Container Apps (idle + active) | ~$42 |
| Azure Container Registry Basic | $5 |
| ADLS Gen2 (250 GB Hot LRS) | ~$5 |
| Key Vault Standard | <$1 |
| Outbound bandwidth | ~$1 |
| **Total** | **~$53** |

---

## Decisions still open

| | Question | Affects |
|---|---|---|
| D1 | Custom domain — `fermata.app` / `fermata-masterclass.com` / `fermata.music` / `learnfermata.com`? | Sprint D |
| D2 | Allow a server-default Gemini key for unauthenticated demo mode, or strictly require BYO from day one? | Sprint B |
| D3 | Chat scope — strictly current lesson, or also "previous takes of the same piece"? | Sprint C |
| D4 | Chat history retention — keep forever, or auto-prune after N days? | Sprint C |

Current defaults (until decided otherwise): D2 = strict BYO with a dev-only env override; D3 = current lesson only; D4 = keep forever.
