# Chat with the Teacher

After a lesson reaches `ready`, the player can open a lesson-scoped chat with the same agentic Gemini teacher that produced the original critique. The chat reuses the lesson teacher wiring: instrument profile, tool catalog, `default_tool_registry`, evidence digest, score note inventory, score images, compact lesson audio, and video frames.

## Endpoints

All endpoints require `TenantContext` through `X-User-Id` (or the current local-dev query fallback).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/lessons/{session_id}/chat` | Run one synchronous chat turn. Body: `{ "message": str, "conversation_id"?: str }`. Returns `{ conversation_id, reply, tool_calls, usage }`. |
| `GET` | `/lessons/{session_id}/chat` | List conversations for the lesson as `{ id, started_at, last_message_at, message_count }`. |
| `GET` | `/lessons/{session_id}/chat/{conv_id}` | Return a full persisted conversation. |
| `DELETE` | `/lessons/{session_id}/chat/{conv_id}` | Delete a conversation. |

`POST` is intentionally not a background job. It runs in the request thread and normally takes 5-30 seconds.

## Conversation schema

Stored at `tenant/{tenant}/users/{user}/sessions/{session_id}/chat/{conversation_id}.json`:

```json
{
  "schema_version": 1,
  "conversation_id": "abc123",
  "session_id": "a933ec4b...",
  "user_id": "pqian",
  "created_at": "2026-05-15T00:00:00Z",
  "updated_at": "2026-05-15T00:01:00Z",
  "messages": [
    {"role": "user", "content": "Can you explain bar 5 again?", "ts": "..."},
    {
      "role": "teacher",
      "content": "Yes — in bar 5...",
      "ts": "...",
      "tool_calls": [{"turn": 1, "tool": "listen", "args": {"start_sec": 42, "end_sec": 49}, "status": "ok"}],
      "usage": {"input_tokens": 12000, "output_tokens": 600, "estimated_cost_usd": 0.021, "model": "gemini-2.5-pro"}
    }
  ]
}
```

## Guardrails

- **Daily user cap:** 50 user messages per UTC day across all chats. Tracked under `sessions/_user_quotas/{user_id}_{YYYYMMDD}.json`; returns HTTP 429 when exhausted.
- **Conversation cap:** 20 user messages per conversation; returns HTTP 429.
- **Message size cap:** 2 KB UTF-8 per user message; returns HTTP 413.
- **Tool cap:** 5 teacher tool calls per chat turn.
- **Topic guard:** before Pro is invoked, a cheap Gemini 2.5 Flash yes/no check decides whether the question concerns music performance, music pedagogy, or the just-completed lesson. Off-topic requests return HTTP 422 with: `I'm here to help with this music lesson. Could you ask something about your performance, the score, or what we discussed?` The decision is cached per `(user_id, sha256(message))` for 1 hour. `DISABLE_TOPIC_GUARD=true` skips it for tests.

## Cost model

`src/masterclass/engine/teach_chat.py` contains the chat pricing table:

- Gemini 2.5 Pro: $1.25/M input tokens up to 200K, $5.00/M input tokens above 200K, $10.00/M output tokens.
- Gemini 2.5 Flash topic guard: $0.075/M input tokens, $0.30/M output tokens.

Each teacher response includes `usage = { input_tokens, output_tokens, estimated_cost_usd, model }`. Accepted chat turns include the topic-guard cost in the estimate when guard usage is available.

## BYOK seam

For now chat uses the same shared-key `LlmProvider` path as lesson teaching (`GEMINI_API_KEY`). The API endpoint has a TODO to switch construction to `use_for_user(user_id)` when the auth + BYO Gemini API key work lands.
