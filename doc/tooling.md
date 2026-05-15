# Bundled Toolchain & External Services

v2 is a single-process app that depends on bundled binaries. Everything ships under `tools/` (gitignored, ~1 GB total).

## Bundled binaries

| Tool | Version | Path | Size | Purpose |
|---|---|---|---|---|
| Python | 3.12 | `tools/python/python.exe` | ~250 MB w/ deps | Pipeline runtime |
| ffmpeg | static build | `tools/ffmpeg/bin/ffmpeg.exe` | ~150 MB | Audio/video extraction, clip encoding |
| Eclipse Temurin JDK | 21.0.11+10 | `tools/jre/` | ~280 MB | Audiveris runtime |
| Audiveris | 5.6.2 | `tools/audiveris/` | ~150 MB | OMR (PDF → MusicXML) |

All paths are resolved relative to the project root via `_project_root()` helpers (`agent_tools/_common.py`, `engine/audiveris_omr.py`, `toolchain/ffmpeg.py`). If a bundled binary isn't found, the helpers fall back to system PATH.

## Python dependencies

Pinned in `pyproject.toml`. Key deps:

- **fastapi** — web framework
- **uvicorn** — ASGI server
- **librosa** — audio analysis (CQT, chroma, onsets, pitch)
- **numpy / scipy** — numerical kernels
- **opencv-python (cv2)** — image processing for barline/staff detection
- **Pillow** — image I/O
- **pretty_midi** — MIDI parsing
- **pymupdf (fitz)** — PDF rasterization
- **music21** — MusicXML parsing
- **lxml** — XML parsing
- **google-genai** — Gemini API client (NEW SDK, not the old `google-generativeai`)

## External services

### Gemini API (Google)

- **Models used**: `gemini-2.5-pro` (teacher, score-prep fallback), `gemini-2.5-flash` (MIDI picker, query normalizer, `watch`/`listen` tools)
- **Auth**: `GEMINI_API_KEY` env var (loaded from `.env` at project root)
- **Endpoints**: standard `models.generate_content` + Files API
- **Rate limits**: 503 UNAVAILABLE happens during peak hours, especially on Pro. Built-in retry: 3 attempts with 5s/10s/15s backoff.
- **Cost**: typical lesson $0.30-2.00. Score-prep alone ~$0.50 if Gemini fallback fires.

### Mutopia Project (free)

- HTTP-only catalog of public-domain music: `https://www.mutopiaproject.org`
- Used for MIDI auto-find. We hit `cgibin/make-table.cgi?searchingfor={query}` and parse the HTML result table.
- No auth, no rate limits in practice.
- Coverage: ~2000 pieces. Mostly classical (Bach, Mozart, Beethoven, Chopin, etc.)
- File formats: ships `.ly` + `.mid` + `.pdf`. **No MusicXML** for most pieces (a recurring disappointment — would have eliminated the need for Audiveris if it had).

## MIDI auto-finder (`engine/midi_finder.py`)

Hybrid system: deterministic catalog search + LLM-grounded pick.

```
auto_attach_midi_to_masterclass()
    │
    ├── 1. mutopia_search(piece_name)
    │       Hit make-table.cgi with the user's exact piece description
    │       Regex-parse the nested HTML tables (one table per piece)
    │       Return list of candidates with {title, composer, opus, instrument, midi_url}
    │
    ├── 2. If 0 candidates OR <3 candidates:
    │       normalize_search_queries(piece_name) via Gemini Flash
    │       Returns 2-4 catalog-friendly queries (e.g. "bach bwv 1001", "chopin op 9 no 2")
    │       Retry mutopia_search with each query, accumulate
    │
    ├── 3. If 0 candidates after all retries → return found=False
    │
    ├── 4. gemini_pick_best(candidates) via Gemini Flash with structured output
    │       JSON schema constrained: must return one of the candidate midi_urls verbatim
    │       Returns {found, midi_url, confidence, reasoning}
    │
    ├── 5. download_midi(picked_url)
    │       Validate first 4 bytes are MThd (MIDI magic)
    │       Persist as reference/midi
    │       Stamp metadata: reference_midi_source, reference_midi_url, reference_midi_attribution
    │
    └── 6. Save audit to reference/midi_find.json
            All candidates considered, all queries tried, pick reasoning, token cost
```

### Mutopia HTML parser

Mutopia's HTML is one **nested table per piece** — not one `<tr>` per piece. Original parser was wrong (matched fields across pieces, paired Bach titles with Beethoven URLs). Fixed:

```python
_MUTOPIA_PIECE_BLOCK_RE = re.compile(
    r'<table\s+class="table-bordered\s+result-table">(?P<body>.*?)</table>',
    re.DOTALL | re.IGNORECASE,
)
```

Cell layout within each piece block (16 cells across 4 rows):
- 0: title, 1: "by COMPOSER", 2: opus, 3: spacer
- 4: "for INSTRUMENT", 5: year, 6: style, 7: spacer
- 8: source-edition, 9: license-link, 10: piece-info link, 11: date
- 12: .ly link, 13: .mid link, 14: preview, 15: ftp dir

Robust to occasional layout variations because we extract the .mid URL via separate regex (`href="..._.mid"`).

### Query normalization prompt

Original was too generic. Refined to:

> You are a search-keyword normalizer for the Mutopia Project (a public-domain music catalog with a strict keyword-AND search). Given a user-supplied piece description, produce 2-4 short search keyword strings, ordered most-specific first. Each string is 2-3 words, lowercase, no punctuation, no commas, no dashes. Fix obvious typos (e.g. 'nocture' -> 'nocturne'). Mutopia indexes catalog ids well — when a piece has a known catalog identifier, include it as one of the keyword strings: 'bach bwv 1001', 'mozart kv 545', 'beethoven op 27', 'chopin op 9 no 2'. Also include a generic fallback like 'composer-surname instrument-or-genre'. Avoid over-specific multi-word strings (Mutopia returns zero hits for 'bach violin sonata g minor').

Result: works correctly on user-supplied "Bach Violin Sonata No. 1 in G minor" → finds BWV 1001 Adagio.

## Files API quirks

The Gemini new SDK (`google-genai`, not `google-generativeai`) has a quirk where `genai.Client` may sometimes raise "client has been closed" mid-call. The fix in our wrappers:

```python
for attempt in range(2):
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(...)
        break
    except Exception as exc:
        if "client has been closed" in str(exc).lower():
            continue
        raise
```

A fresh client per retry. Works around the bug at minimal cost.

Files API uploads use `client.files.upload(file=BytesIO(bytes), config={"mime_type": "..."})`. Inline bytes via `Part.from_bytes` is preferred for clips < 15 MB (faster, no upload-then-fetch round-trip).

## Audiveris invocation

```python
cmd = [
    java_exe,
    "-jar", audiveris_jar,
    "-batch", "-export",
    "-output", str(tmp_outdir),
    str(tmp_pdf_path),
]
result = subprocess.run(cmd, capture_output=True, timeout=timeout_sec)
```

Audiveris writes `.mxl` files to the output directory. We pick the first `.mxl` found and read its bytes. Failure modes:

- Java not found → `RuntimeError("Java not found at ...")`
- Audiveris jar not found → similar
- Audiveris exits non-zero → `RuntimeError("Audiveris failed: ...stderr...")`
- Timeout → `subprocess.TimeoutExpired`

All failures are caught by `score_prep.py` and trigger the Gemini fallback path.

## Environment variables

Loaded from `.env` at project root via `python-dotenv` semantics (manual parse in scripts; FastAPI app uses python-dotenv).

| Var | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Gemini API auth |
| `MASTERCLASS_LLM_PROVIDER` | `gemini` | Set to `dry-run` to skip all LLM calls (offline dev) |
| `MASTERCLASS_LOCAL_ADLS_ROOT` | `./local_adls` | Override storage root |
| `MASTERCLASS_TEACH_MODEL` | `gemini-2.5-pro` | Teacher model override |
| `MASTERCLASS_SCORE_PREP_MODEL` | `gemini-2.5-pro` | Score-prep model (when Gemini fallback fires) |
| `MASTERCLASS_MIDI_FIND_MODEL` | `gemini-2.5-flash` | MIDI picker model |
| `MASTERCLASS_DISABLE_AUDIVERIS` | (unset) | Set to `1` to force Gemini score-prep path |
| `LISTEN_MAX_CLIP_SEC` | `60` | Max audio clip length for `listen` tool |
| `WATCH_MAX_CLIP_SEC` | `10` | Max video clip length for `watch` tool |
| `WATCH_INLINE_BYTES` | `15728640` (15 MB) | Threshold above which `watch` uses Files API instead of inline |
