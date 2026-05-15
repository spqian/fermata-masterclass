"""Hybrid MIDI auto-find for v2 masterclasses.

Two-stage flow:
  1. Deterministic search against public-domain catalog sites (Mutopia today;
     piano-midi.de can be added similarly). Returns a small candidate list of
     real, reachable MIDI URLs with title/composer/opus metadata.
  2. Gemini Flash picks the best candidate from the structured list. The model
     is grounded in actual catalog rows, so it cannot hallucinate URLs.

The chosen URL is then downloaded and verified to start with the ``MThd`` magic
bytes. The result + per-stage audit is persisted on the masterclass.
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from masterclass.agent.llm import LlmProvider, LlmUsage
from masterclass.core.masterclasses import MasterclassStore
from masterclass.core.models import MasterclassManifest
from masterclass.storage.base import ObjectStorage


HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MusicMasterclass/2.0)",
    "Accept": "text/html, audio/midi, */*",
}


@dataclass(frozen=True)
class MidiFinderConfig:
    model: str = "gemini-2.5-flash"
    catalog_timeout_sec: int = 15
    download_timeout_sec: int = 12
    pick_timeout_sec: int = 30
    overall_deadline_sec: int = 60
    max_candidates_per_source: int = 12
    max_midi_bytes: int = 5 * 1024 * 1024  # 5 MB cap


@dataclass
class MidiFindResult:
    found: bool
    midi_url: str | None = None
    midi_bytes: bytes | None = None
    source: str | None = None
    title: str | None = None
    composer: str | None = None
    attribution: str | None = None
    confidence: str | None = None
    reasoning: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    pick_reasoning: str | None = None
    usage: LlmUsage | None = None


# ----------------------------------------------------------------- Mutopia


_MUTOPIA_PIECE_BLOCK_RE = re.compile(
    r'<table\s+class="table-bordered\s+result-table">(?P<body>.*?)</table>',
    re.DOTALL | re.IGNORECASE,
)
_MUTOPIA_TD_RE = re.compile(r"<td[^>]*>(?P<cell>.*?)</td>", re.DOTALL | re.IGNORECASE)
_MUTOPIA_INFO_RE = re.compile(r'piece-info\.cgi\?id=(\d+)', re.IGNORECASE)
_MUTOPIA_MIDI_RE = re.compile(r'href="(https?://[^"]+?\.midi?)"', re.IGNORECASE)


def mutopia_search(piece_name: str, *, timeout: int) -> list[dict[str, Any]]:
    """Hit Mutopia's make-table.cgi search and parse the result table.

    Returns a list of candidate dicts with title/composer/opus/instrument/midi_url.
    Mutopia's HTML is regular enough that a small regex is reliable here; we
    don't pull in BeautifulSoup just for one search source.
    """

    query = urllib.parse.quote_plus(piece_name.strip())
    url = f"https://www.mutopiaproject.org/cgibin/make-table.cgi?searchingfor={query}"
    request = urllib.request.Request(url, headers=HTTP_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            html = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise RuntimeError(f"mutopia search failed: {exc}") from exc

    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for block in _MUTOPIA_PIECE_BLOCK_RE.finditer(html):
        body = block.group("body")
        cells = [_clean_html(m.group("cell")) for m in _MUTOPIA_TD_RE.finditer(body)]
        # Cell layout (fairly stable across Mutopia results):
        #   0: title, 1: "by COMPOSER", 2: opus, 3: (spacer)
        #   4: "for INSTRUMENT", 5: year, 6: style, 7: (spacer)
        #   8: source-edition, 9: license-link, 10: piece-info link, 11: date
        #  12: .ly link, 13: .mid link, 14: preview, 15: ftp dir
        if len(cells) < 14:
            continue
        midi_match = _MUTOPIA_MIDI_RE.search(body)
        if not midi_match:
            continue
        midi_url = midi_match.group(1).strip()
        if midi_url in seen_urls:
            continue
        seen_urls.add(midi_url)
        info_match = _MUTOPIA_INFO_RE.search(body)
        info_url = (
            f"https://www.mutopiaproject.org/cgibin/piece-info.cgi?id={info_match.group(1)}"
            if info_match else None
        )
        title = cells[0]
        composer = cells[1][3:].strip() if cells[1].lower().startswith("by ") else cells[1]
        opus = cells[2]
        instrument = cells[4][4:].strip() if cells[4].lower().startswith("for ") else cells[4]
        year = cells[5] if len(cells) > 5 else ""
        style = cells[6] if len(cells) > 6 else ""
        candidates.append({
            "source": "mutopia",
            "title": title,
            "composer": composer,
            "opus": opus,
            "instrument": instrument,
            "year": year,
            "style": style,
            "info_url": info_url,
            "midi_url": midi_url,
        })
    return candidates


def _clean_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value or "")
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- LLM pick


_PICK_SYSTEM_INSTRUCTION = (
    "You are a music librarian helping match a user-supplied piece description "
    "to one of several real catalog candidates. Each candidate is a concrete "
    "row from a public-domain music catalog (Mutopia, piano-midi.de, etc.) "
    "with a verified midi_url. Pick the candidate that best matches the piece. "
    "If no candidate matches, return midi_url as the empty string. Never "
    "invent URLs — only return one of the supplied candidate midi_urls verbatim."
)


def gemini_pick_best(
    *,
    provider: LlmProvider,
    piece_name: str,
    movement: str | None,
    instrument_profile: str | None,
    candidates: list[dict[str, Any]],
    model: str,
) -> tuple[dict[str, Any], LlmUsage]:
    if not candidates:
        return {"midi_url": "", "reasoning": "no candidates supplied"}, _empty_usage(model)

    prompt = (
        f"Piece description from user:\n"
        f"  piece_name: {piece_name!r}\n"
        f"  movement:   {movement!r}\n"
        f"  instrument: {instrument_profile!r}\n\n"
        f"Catalog candidates ({len(candidates)}):\n"
        + json.dumps(candidates, indent=2)
        + "\n\nReturn ONLY this JSON: {\"midi_url\": \"<one of the candidate midi_urls or empty>\", "
        "\"confidence\": \"high|medium|low\", \"reasoning\": \"one sentence\"}."
    )
    schema = {
        "type": "object",
        "properties": {
            "midi_url": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reasoning": {"type": "string"},
        },
        "required": ["midi_url"],
    }
    return provider.generate_json(
        model=model,
        system_instruction=_PICK_SYSTEM_INSTRUCTION,
        contents=[prompt],
        response_schema=schema,
    )


def _empty_usage(model: str) -> LlmUsage:
    return LlmUsage(provider="none", model=model, input_tokens=0, output_tokens=0, estimated_cost_usd=0.0)


# ---------------------------------------------------------------- Download


def download_midi(url: str, *, timeout: int, max_bytes: int) -> bytes:
    """Fetch a candidate URL and verify it is a real MIDI file."""

    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"non-http URL refused: {url!r}")

    request = urllib.request.Request(url, headers=HTTP_HEADERS)
    contexts = [None]
    if url.lower().startswith("https://"):
        contexts.append(ssl._create_unverified_context())

    last_error: Exception | None = None
    data = b""
    for ctx in contexts:
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=ctx) as response:
                data = response.read(max_bytes + 1)
                break
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"http {exc.code}") from exc
        except urllib.error.URLError as exc:
            last_error = exc
            reason = getattr(exc, "reason", exc)
            if ctx is None and isinstance(reason, ssl.SSLError):
                continue
            raise RuntimeError(f"url error: {reason}") from exc
    else:
        raise RuntimeError(f"url error: {last_error.reason if last_error else 'unknown'}")

    if not data:
        raise RuntimeError("empty response")
    if len(data) > max_bytes:
        raise RuntimeError(f"midi too large (>{max_bytes} bytes)")
    if not data.startswith(b"MThd"):
        raise RuntimeError("response does not start with MThd; not a MIDI file")
    return data


# ---------------------------------------------------------------- Driver


_QUERY_NORMALIZE_INSTRUCTION = (
    "You are a search-keyword normalizer for the Mutopia Project (a public-domain "
    "music catalog with a strict keyword-AND search). Given a user-supplied piece "
    "description, produce 2-4 short search keyword strings, ordered most-specific "
    "first. Each string is 2-3 words, lowercase, no punctuation, no commas, no dashes. "
    "Fix obvious typos (e.g. 'nocture' -> 'nocturne'). "
    "Mutopia indexes catalog ids well — when a piece has a known catalog identifier, "
    "include it as one of the keyword strings: 'bach bwv 1001', 'mozart kv 545', "
    "'beethoven op 27', 'chopin op 9 no 2'. Also include a generic fallback like "
    "'composer-surname instrument-or-genre'. Avoid over-specific multi-word strings "
    "(Mutopia returns zero hits for 'bach violin sonata g minor'). "
    "Return ONLY JSON: {\"queries\": [\"...\", ...]}."
)


def normalize_search_queries(
    *,
    provider: LlmProvider,
    piece_name: str,
    movement: str | None,
    instrument_profile: str | None,
    model: str,
) -> tuple[list[str], LlmUsage]:
    prompt = (
        f"Piece description from user:\n"
        f"  piece_name: {piece_name!r}\n"
        f"  movement:   {movement!r}\n"
        f"  instrument: {instrument_profile!r}\n"
    )
    schema = {
        "type": "object",
        "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
        "required": ["queries"],
    }
    try:
        result, usage = provider.generate_json(
            model=model,
            system_instruction=_QUERY_NORMALIZE_INSTRUCTION,
            contents=[prompt],
            response_schema=schema,
        )
    except Exception:
        return [], _empty_usage(model)
    queries = []
    for q in result.get("queries", []):
        if isinstance(q, str) and q.strip() and q.strip() not in queries:
            queries.append(q.strip())
    return queries[:4], usage


def find_and_download_midi(
    *,
    masterclass: MasterclassManifest,
    provider: LlmProvider,
    config: MidiFinderConfig | None = None,
) -> MidiFindResult:
    """Search public-domain catalogs, let the LLM pick, validate the download.

    The whole flow is bounded by ``overall_deadline_sec`` so a hung HTTP call
    never strands the UI. The catalog search is deterministic; the only LLM
    call is the small pick step against a structured candidate list.
    """

    config = config or MidiFinderConfig()
    audit: dict[str, Any] = {"sources": {}, "queries_tried": []}

    # Try the user's exact wording first; if Mutopia returns nothing, ask
    # Gemini Flash to normalize the query (fix typos, drop noise) and retry.
    queries_to_try: list[str] = [masterclass.piece_name.strip()]
    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    def _try_query(query: str) -> int:
        try:
            results = mutopia_search(query, timeout=config.catalog_timeout_sec)
        except Exception as exc:
            audit["queries_tried"].append({"query": query, "error": f"{type(exc).__name__}: {exc}"})
            return 0
        added = 0
        for c in results[: config.max_candidates_per_source]:
            if c["midi_url"] in seen_urls:
                continue
            seen_urls.add(c["midi_url"])
            candidates.append(c)
            added += 1
        audit["queries_tried"].append({"query": query, "found": len(results), "added": added})
        return added

    _try_query(queries_to_try[0])

    normalize_usage = _empty_usage(config.model)
    normalized: list[str] = []
    if not candidates or len(candidates) < 3:
        normalized, normalize_usage = normalize_search_queries(
            provider=provider,
            piece_name=masterclass.piece_name,
            movement=masterclass.movement,
            instrument_profile=masterclass.instrument_profile,
            model=config.model,
        )
        for q in normalized:
            queries_to_try.append(q)
            _try_query(q)
            if len(candidates) >= config.max_candidates_per_source * 2:
                break

    audit["sources"]["mutopia"] = {"count": len(candidates)}

    if not candidates:
        return MidiFindResult(
            found=False,
            reasoning=(
                "No Mutopia candidates after trying: "
                + ", ".join(repr(q) for q in queries_to_try)
            ),
            candidates=[],
            usage=normalize_usage,
        )

    # 2. Let Gemini pick the best candidate from the verified list.
    try:
        pick, usage = gemini_pick_best(
            provider=provider,
            piece_name=masterclass.piece_name,
            movement=masterclass.movement,
            instrument_profile=masterclass.instrument_profile,
            candidates=candidates,
            model=config.model,
        )
    except Exception as exc:
        return MidiFindResult(
            found=False,
            reasoning=f"LLM pick failed: {type(exc).__name__}: {exc}",
            candidates=candidates,
            usage=_empty_usage(config.model),
        )

    chosen_url = (pick.get("midi_url") or "").strip()
    pick_reasoning = (pick.get("reasoning") or "").strip()
    confidence = (pick.get("confidence") or "").strip() or None

    if not chosen_url:
        return MidiFindResult(
            found=False,
            reasoning=pick_reasoning or "LLM rejected all candidates",
            candidates=candidates,
            pick_reasoning=pick_reasoning,
            usage=usage,
        )

    # Sanity: chosen URL must be one of the candidates we surfaced.
    chosen_candidate = next(
        (c for c in candidates if c["midi_url"].rstrip("/") == chosen_url.rstrip("/")),
        None,
    )
    if chosen_candidate is None:
        return MidiFindResult(
            found=False,
            reasoning=f"LLM returned an off-list URL ({chosen_url}); refusing.",
            candidates=candidates,
            pick_reasoning=pick_reasoning,
            usage=usage,
        )

    # 3. Download + validate.
    try:
        data = download_midi(chosen_url, timeout=config.download_timeout_sec, max_bytes=config.max_midi_bytes)
    except Exception as exc:
        return MidiFindResult(
            found=False,
            midi_url=chosen_url,
            reasoning=f"download rejected: {type(exc).__name__}: {exc}",
            candidates=candidates,
            pick_reasoning=pick_reasoning,
            usage=usage,
        )

    return MidiFindResult(
        found=True,
        midi_url=chosen_url,
        midi_bytes=data,
        source=chosen_candidate.get("source"),
        title=chosen_candidate.get("title"),
        composer=chosen_candidate.get("composer"),
        attribution=f"{chosen_candidate.get('composer','')} — {chosen_candidate.get('title','')} (via {chosen_candidate.get('source','catalog')})".strip(" —"),
        confidence=confidence,
        reasoning=pick_reasoning,
        candidates=candidates,
        pick_reasoning=pick_reasoning,
        usage=usage,
    )


def auto_attach_midi_to_masterclass(
    *,
    storage: ObjectStorage,
    masterclass_store: MasterclassStore,
    manifest: MasterclassManifest,
    provider: LlmProvider,
    config: MidiFinderConfig | None = None,
) -> MasterclassManifest:
    """Run the find+validate flow and persist any discovered MIDI on the masterclass."""

    manifest.metadata["midi_find_state"] = "running"
    manifest.metadata["midi_find_substage"] = "searching mutopia catalog"
    manifest.metadata["midi_find_updated_at"] = datetime.now(UTC).isoformat()
    masterclass_store.save(manifest)

    result = find_and_download_midi(masterclass=manifest, provider=provider, config=config or MidiFinderConfig())

    audit = {
        "found": result.found,
        "midi_url": result.midi_url,
        "source": result.source,
        "title": result.title,
        "composer": result.composer,
        "attribution": result.attribution,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "pick_reasoning": result.pick_reasoning,
        "candidates": result.candidates,
        "input_tokens": result.usage.input_tokens if result.usage else None,
        "output_tokens": result.usage.output_tokens if result.usage else None,
        "estimated_cost_usd": result.usage.estimated_cost_usd if result.usage else None,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    audit_key = masterclass_store.artifact_key(manifest.masterclass, "reference/midi_find.json")
    storage.write_json(audit_key, audit)
    manifest.artifacts["reference/midi_find.json"] = audit_key

    if result.found and result.midi_bytes:
        midi_key = masterclass_store.artifact_key(manifest.masterclass, "reference/midi/auto.mid")
        storage.write_bytes(midi_key, result.midi_bytes, content_type="audio/midi")
        manifest.artifacts["reference/midi"] = midi_key
        manifest.metadata["reference_midi_filename"] = "auto.mid"
        manifest.metadata["reference_midi_size_bytes"] = len(result.midi_bytes)
        manifest.metadata["reference_midi_uploaded_at"] = datetime.now(UTC).isoformat()
        manifest.metadata["reference_midi_source"] = result.source
        manifest.metadata["reference_midi_url"] = result.midi_url
        manifest.metadata["reference_midi_attribution"] = result.attribution
        manifest.metadata["reference_midi_confidence"] = result.confidence
        manifest.metadata["midi_find_state"] = "ready"
        manifest.metadata["midi_find_error"] = None
    else:
        manifest.metadata["midi_find_state"] = "not_found"
        manifest.metadata["midi_find_error"] = result.reasoning or "no candidate accepted"

    manifest.metadata["midi_find_substage"] = None
    manifest.metadata["midi_find_updated_at"] = datetime.now(UTC).isoformat()
    masterclass_store.save(manifest)
    return manifest
