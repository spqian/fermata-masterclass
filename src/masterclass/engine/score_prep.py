from __future__ import annotations

import io
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from masterclass.agent.llm import LlmProvider, LlmUsage
from masterclass.core.masterclasses import MasterclassStore
from masterclass.core.models import MasterclassManifest
from masterclass.engine.staff_detection import StaffDetectionConfig, detect_barlines_in_system, detect_staff_systems_from_image
from masterclass.storage.base import ObjectStorage


SCORE_PREP_SYSTEM_INSTRUCTION = (
    "You are a careful music librarian preparing a piano/violin score for an interactive "
    "masterclass player. You will be shown rasterized pages of an IMSLP-style score PDF in "
    "order. For each page identify whether it is front matter (title, index, blank) or music. "
    "A SYSTEM is the full row of music a player reads at once. For solo melodic instruments "
    "(violin, cello, viola) a system is one staff line. For piano music a system is the "
    "GRAND STAFF — both the treble (right hand) and bass (left hand) staves bracketed "
    "together as one row of music. Never split a piano grand staff into two systems. The "
    "bbox for a piano system must enclose BOTH the treble and bass staves AND the brace "
    "that joins them.\n"
    "For music pages, return:\n"
    "  - movement boundaries with tempo, time, and key signatures,\n"
    "  - the first and last measure on the page,\n"
    "  - one entry per system (full row of music) on the page, with:\n"
    "      * 1-indexed system_index within the page,\n"
    "      * first_measure and last_measure of that system,\n"
    "      * bars: one object per printed bar in that system, with bar_number, "
    "x_frac_start, and x_frac_end. These x fractions are normalized 0-1 page "
    "coordinates. Count printed barlines carefully; the measure count implied by "
    "first_measure..last_measure must equal len(bars).\n"
    "      * a normalized bounding box bbox = {x, y, w, h} in [0, 1] coordinates relative to "
    "        the page image, where (0,0) is the top-left and (1,1) is the bottom-right.\n"
    "Bounding boxes should be tight but include any clef/key/time signature, dynamics, and the "
    "lower piano staff for grand-staff systems.\n"
    "Be conservative: prefer null when unsure."
)

SCORE_PREP_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "first_music_page": {"type": "integer", "description": "1-indexed page number where the first measure begins"},
        "page_count": {"type": "integer"},
        "instrument": {"type": "string", "description": "Best guess at the instrument for this score"},
        "movements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                    "tempo_marking": {"type": "string"},
                    "time_signature": {"type": "string"},
                    "key_signature": {"type": "string"},
                    "start_page": {"type": "integer"},
                    "end_page": {"type": "integer"},
                    "first_measure": {"type": "integer"},
                    "last_measure": {"type": "integer"},
                    "measure_count": {"type": "integer"},
                },
                "required": ["id", "start_page"],
            },
        },
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "kind": {"type": "string", "description": "title|index|blank|music|other"},
                    "movement_id": {"type": "integer"},
                    "first_measure": {"type": "integer"},
                    "last_measure": {"type": "integer"},
                    "system_count": {"type": "integer"},
                    "systems": {
                        "type": "array",
                        "description": "One entry per system (line of music) on the page, top to bottom.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "system_index": {"type": "integer"},
                                "first_measure": {"type": "integer"},
                                "last_measure": {"type": "integer"},
                                "bbox": {
                                    "type": "object",
                                    "description": "Normalized bbox in [0,1] page coordinates",
                                    "properties": {
                                        "x": {"type": "number"},
                                        "y": {"type": "number"},
                                        "w": {"type": "number"},
                                        "h": {"type": "number"},
                                    },
                                    "required": ["x", "y", "w", "h"],
                                },
                                "bars": {
                                    "type": "array",
                                    "description": "One object per bar; x coordinates are normalized page fractions.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "bar_number": {"type": "integer"},
                                            "x_frac_start": {"type": "number"},
                                            "x_frac_end": {"type": "number"},
                                        },
                                        "required": ["bar_number", "x_frac_start", "x_frac_end"],
                                    },
                                },
                            },
                            "required": ["system_index", "bbox"],
                        },
                    },
                },
                "required": ["page", "kind"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["first_music_page", "page_count", "movements", "pages"],
}


@dataclass(frozen=True)
class ScorePrepConfig:
    model: str = "gemini-2.5-pro"
    max_pages: int = 60
    page_dpi: int = 150
    use_staff_detection_fallback: bool = True


@dataclass
class ScorePrepResult:
    prep: dict[str, Any]
    usage: LlmUsage
    page_keys: list[str]


def rasterize_pdf_pages(pdf_bytes: bytes, *, dpi: int, max_pages: int) -> list[bytes]:
    """Rasterize each PDF page to PNG bytes using PyMuPDF.

    Caller controls page count to keep multimodal cost bounded.
    """

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("PyMuPDF (pymupdf) is required for score prep") from exc

    images: list[bytes] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for index, page in enumerate(document):
            if index >= max_pages:
                break
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(pixmap.tobytes("png"))
    return images


def prepare_score(
    *,
    storage: ObjectStorage,
    masterclass_store: MasterclassStore,
    manifest: MasterclassManifest,
    provider: LlmProvider,
    config: ScorePrepConfig | None = None,
) -> ScorePrepResult:
    """Run a full score-prep pass for a masterclass and persist results.

    Reads the attached PDF from storage, rasterizes its pages, asks the
    multimodal LLM to identify front matter, music pages, and movement
    boundaries, and writes the structured prep document plus rasterized page
    images back into masterclass storage.
    """

    config = config or ScorePrepConfig()
    pdf_key = manifest.artifacts.get("reference/score_pdf")
    if not pdf_key:
        raise ValueError("masterclass has no reference/score_pdf attached")

    _set_prep_state(masterclass_store, manifest, "running", error=None)

    def _mark_substage(label: str) -> None:
        manifest.metadata["score_prep_substage"] = label
        manifest.metadata["score_prep_updated_at"] = datetime.now(UTC).isoformat()
        masterclass_store.save(manifest)

    started = datetime.now(UTC)
    manifest.metadata["score_prep_started_at"] = started.isoformat()
    masterclass_store.save(manifest)

    try:
        _mark_substage("reading PDF")
        pdf_bytes = storage.read_bytes(pdf_key)

        _mark_substage("rasterizing pages")
        page_images = rasterize_pdf_pages(pdf_bytes, dpi=config.page_dpi, max_pages=config.max_pages)
        if not page_images:
            raise ValueError("PDF has no pages")

        _mark_substage(f"storing {len(page_images)} page raster(s)")
        page_keys: list[str] = []
        for index, png in enumerate(page_images, start=1):
            key = masterclass_store.artifact_key(manifest.masterclass, f"reference/score_pages/page-{index:03d}.png")
            storage.write_bytes(key, png, content_type="image/png")
            page_keys.append(key)
            manifest.artifacts[f"reference/score_pages/page-{index:03d}.png"] = key
        masterclass_store.save(manifest)

        prep: dict[str, Any] | None = None
        contents: list[Any] = []
        reprompted = False
        source = "audiveris"
        audiveris_meta: dict[str, Any] = {}
        try:
            from masterclass.engine.audiveris_omr import audiveris_pdf_to_musicxml
            from masterclass.engine.score_layout_from_musicxml import score_prep_from_musicxml

            _mark_substage("running Audiveris OMR on PDF")
            xml_bytes, audiveris_meta = audiveris_pdf_to_musicxml(pdf_bytes, page_dpi=config.page_dpi)
            # Persist the raw MusicXML so the audio-truth score-matcher (and
            # any future consumer that wants per-note information) can read it
            # without re-running Audiveris. This unblocks dropping the
            # Gemini-driven MIDI search: the matcher reads MusicXML directly,
            # which has strictly more information than MIDI (voice numbers,
            # beam groups, dynamics, articulations).
            try:
                xml_filename = audiveris_meta.get("output_filename") or "audiveris.musicxml"
                xml_ext = ".mxl" if str(xml_filename).lower().endswith(".mxl") else ".musicxml"
                xml_content_type = "application/vnd.recordare.musicxml+zip" if xml_ext == ".mxl" else "application/vnd.recordare.musicxml+xml"
                xml_key = masterclass_store.artifact_key(manifest.masterclass, f"reference/musicxml{xml_ext}")
                storage.write_bytes(xml_key, xml_bytes, content_type=xml_content_type)
                manifest.artifacts[f"reference/musicxml{xml_ext}"] = xml_key
                manifest.metadata["score_prep_musicxml_bytes"] = len(xml_bytes)
            except Exception as _xml_err:  # pragma: no cover - best-effort, layout below still works
                logging.warning("Failed to persist MusicXML artifact: %s", _xml_err)
            _mark_substage("converting MusicXML to score_prep layout")
            prep = score_prep_from_musicxml(xml_bytes, page_images=page_images, instrument=manifest.instrument)
            prep["_meta"] = {"source": "audiveris", **audiveris_meta, **(prep.get("_meta") or {})}
            manifest.metadata["score_prep_source"] = "audiveris"
            usage = LlmUsage(
                provider="audiveris",
                model=str(audiveris_meta.get("audiveris_version") or "audiveris"),
                input_tokens=0,
                output_tokens=0,
                estimated_cost_usd=0.0,
            )
        except Exception as exc:
            logging.warning("Audiveris failed, falling back to Gemini: %s", exc)
            manifest.metadata["score_prep_audiveris_error"] = str(exc)
            manifest.metadata["score_prep_source"] = "gemini"
            source = "gemini"
            prompt = (
                f"This score has {len(page_images)} rasterized page(s) below, in order. "
                "Identify front matter and music pages. Return JSON matching the supplied schema. "
                f"Piece name as known: {manifest.piece_name!r}. "
                f"Movement (if known): {manifest.movement!r}. "
                f"Instrument profile (if known): {manifest.instrument_profile!r}."
            )
            contents = [prompt]
            for index, png in enumerate(page_images, start=1):
                contents.append(f"--- page {index} ---")
                contents.append({"mime_type": "image/png", "data": png, "label": f"page-{index}"})

            _mark_substage(f"asking {config.model} to read the score ({len(page_images)} pages)")
            prep, usage = provider.generate_json(
                model=config.model,
                system_instruction=SCORE_PREP_SYSTEM_INSTRUCTION,
                contents=contents,
                response_schema=SCORE_PREP_RESPONSE_SCHEMA,
            )

        _mark_substage("cross-checking layout against MIDI")
        prep.setdefault("page_count", len(page_images))
        midi_measures_total = _masterclass_midi_measure_count(storage=storage, manifest=manifest)
        score_prep_measures_total = _score_prep_measure_count(prep)
        if source == "gemini" and midi_measures_total is not None and abs(score_prep_measures_total - midi_measures_total) > 1:
            reprompted = True
            corrective = (
                f"The MIDI for this piece has {midi_measures_total} measures but you reported "
                f"{score_prep_measures_total}. Re-examine each system. For grand-staff piano music, "
                "each system covers multiple bars (typically 3-5). Count barlines (vertical lines) "
                "carefully and update. Return the complete corrected JSON matching the schema."
            )
            prep, retry_usage = provider.generate_json(
                model=config.model,
                system_instruction=SCORE_PREP_SYSTEM_INSTRUCTION,
                contents=contents + [corrective],
                response_schema=SCORE_PREP_RESPONSE_SCHEMA,
            )
            usage = _merge_usage(usage, retry_usage)
            prep.setdefault("page_count", len(page_images))
            score_prep_measures_total = _score_prep_measure_count(prep)

        _mark_substage("parsing and validating layout")
        if config.use_staff_detection_fallback:
            _mark_substage("running staff-detection validation")
            _apply_staff_detection_fallback(
                prep=prep,
                page_images=page_images,
                storage=storage,
                masterclass_store=masterclass_store,
                manifest=manifest,
            )
        barline_corrections = 0 if source == "audiveris" else _apply_barline_cross_check(prep=prep, page_images=page_images)
        score_prep_measures_total = _score_prep_layout_measure_count(prep)
        redistributed_to_midi = False
        played_movement_id: int | None = None
        played_movement_measure_count: int | None = None
        if source == "audiveris" and midi_measures_total is not None:
            midi_movement_hint = _movement_number_from_midi_url(manifest.metadata.get("reference_midi_url"))
            played_movement_id = _find_movement_matching_measure_count(
                prep,
                midi_measures_total,
                tolerance=2,
                hinted_movement_id=midi_movement_hint,
            )
            if played_movement_id is not None:
                manifest.metadata["played_movement_id"] = played_movement_id
                played_movement_measure_count = _movement_measure_count(prep, played_movement_id)
                if (
                    played_movement_measure_count is not None
                    and redistribute_movement_bars_to_midi(
                        prep,
                        movement_id=played_movement_id,
                        midi_bar_count=midi_measures_total,
                    )
                ):
                    redistributed_to_midi = True
                    played_movement_measure_count = _movement_measure_count(prep, played_movement_id)
                    score_prep_measures_total = _score_prep_layout_measure_count(prep)
            else:
                manifest.metadata.pop("played_movement_id", None)
        if (
            midi_measures_total is not None
            and abs(score_prep_measures_total - midi_measures_total) > 1
            and source != "audiveris"
        ):
            redistributed_to_midi = _redistribute_system_measures_to_total(prep, total_measures=midi_measures_total)
            score_prep_measures_total = _score_prep_layout_measure_count(prep)
        if midi_measures_total is not None:
            manifest.metadata["score_prep_midi_measure_count"] = midi_measures_total
            manifest.metadata["score_prep_layout_measure_count"] = score_prep_measures_total
            if played_movement_measure_count is not None:
                manifest.metadata["score_prep_played_movement_measure_count"] = played_movement_measure_count
            else:
                manifest.metadata.pop("score_prep_played_movement_measure_count", None)
            if source == "gemini":
                manifest.metadata["score_prep_gemini_measure_count"] = score_prep_measures_total
            else:
                manifest.metadata.pop("score_prep_gemini_measure_count", None)
            manifest.metadata["score_prep_measure_count"] = score_prep_measures_total
            manifest.metadata["score_prep_reprompted"] = reprompted
            manifest.metadata["score_prep_redistributed_to_midi"] = redistributed_to_midi
            mismatch_basis = played_movement_measure_count if (source == "audiveris" and played_movement_id is not None) else score_prep_measures_total
            if abs((mismatch_basis or 0) - midi_measures_total) > 1:
                manifest.metadata["score_prep_measure_mismatch"] = (
                    f"{source}={mismatch_basis} midi={midi_measures_total}"
                )
            else:
                manifest.metadata.pop("score_prep_measure_mismatch", None)
        manifest.metadata["score_prep_barline_corrections"] = barline_corrections
        existing_meta = prep.get("_meta") if isinstance(prep.get("_meta"), dict) else {}
        prep["_meta"] = {
            **existing_meta,
            "source": source,
            "model": config.model,
            "page_dpi": config.page_dpi,
            "rasterized_page_count": len(page_images),
            "generated_at": datetime.now(UTC).isoformat(),
            "provider": usage.provider if source == "audiveris" else provider.provider_name,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "estimated_cost_usd": usage.estimated_cost_usd,
            "midi_measures_total": midi_measures_total,
            "gemini_measures_total": score_prep_measures_total if source == "gemini" else None,
            "layout_measures_total": score_prep_measures_total,
            "score_prep_measures_total": score_prep_measures_total,
            "played_movement_id": played_movement_id,
            "played_movement_measure_count": played_movement_measure_count,
            "midi_measure_reprompted": reprompted,
            "barline_corrections": barline_corrections,
            "redistributed_to_midi": redistributed_to_midi,
        }

        _mark_substage("writing prep document")
        prep_key = masterclass_store.artifact_key(manifest.masterclass, "reference/score_prep.json")
        storage.write_json(prep_key, prep)
        manifest.artifacts["reference/score_prep.json"] = prep_key
        manifest.metadata["score_prep_state"] = "ready"
        manifest.metadata["score_prep_substage"] = None
        manifest.metadata["score_prep_error"] = None
        manifest.metadata["score_prep_first_music_page"] = prep.get("first_music_page")
        manifest.metadata["score_prep_movement_count"] = len(prep.get("movements") or [])
        manifest.metadata["score_prep_generated_at"] = prep["_meta"]["generated_at"]
        manifest.metadata["score_prep_elapsed_sec"] = round((datetime.now(UTC) - started).total_seconds(), 1)

        # Auto-infer instrument profile from Gemini's score-prep when the user
        # didn't pick one explicitly. Map common instrument names to v2 profile ids.
        if not manifest.instrument_profile:
            inferred = _infer_profile_from_instrument(prep.get("instrument"))
            if inferred:
                manifest.instrument_profile = inferred
                manifest.metadata["instrument_profile_inferred"] = True
                manifest.metadata["instrument_profile_inferred_from"] = prep.get("instrument")

        masterclass_store.save(manifest)
        return ScorePrepResult(prep=prep, usage=usage, page_keys=page_keys)
    except Exception as exc:
        _set_prep_state(masterclass_store, manifest, "failed", error=f"{type(exc).__name__}: {exc}")
        raise


def _merge_usage(first: LlmUsage, second: LlmUsage) -> LlmUsage:
    def add(a: int | None, b: int | None) -> int | None:
        if a is None and b is None:
            return None
        return int(a or 0) + int(b or 0)

    cost = None
    if first.estimated_cost_usd is not None or second.estimated_cost_usd is not None:
        cost = float(first.estimated_cost_usd or 0.0) + float(second.estimated_cost_usd or 0.0)
    return LlmUsage(
        provider=first.provider,
        model=first.model,
        input_tokens=add(first.input_tokens, second.input_tokens),
        output_tokens=add(first.output_tokens, second.output_tokens),
        estimated_cost_usd=cost,
    )


def _masterclass_midi_key(storage: ObjectStorage, manifest: MasterclassManifest) -> str | None:
    candidates = [
        manifest.artifacts.get("reference/midi"),
        manifest.artifacts.get("reference/midi/auto.mid"),
    ]
    prefix = (
        f"tenant/{manifest.masterclass.tenant_id}/users/{manifest.masterclass.user_id}"
        f"/masterclasses/{manifest.masterclass.masterclass_id}/reference/midi"
    )
    candidates.extend(key for key in storage.list_keys(prefix) if key.lower().endswith((".mid", ".midi")))
    for key in candidates:
        if key and storage.exists(str(key)):
            return str(key)
    return None


def _masterclass_midi_measure_count(*, storage: ObjectStorage, manifest: MasterclassManifest) -> int | None:
    key = _masterclass_midi_key(storage, manifest)
    if not key:
        return None
    try:
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(io.BytesIO(storage.read_bytes(key)))
        return max(1, len(midi.get_downbeats()))
    except Exception as exc:
        manifest.metadata["score_prep_midi_measure_error"] = f"{type(exc).__name__}: {exc}"
        return None


def _score_prep_measure_count(prep: dict[str, Any]) -> int:
    total = 0
    for page in prep.get("pages") or []:
        if not isinstance(page, dict) or page.get("kind") not in (None, "music"):
            continue
        for system in page.get("systems") or []:
            if not isinstance(system, dict):
                continue
            first = _as_int(system.get("first_measure"), default=None)
            last = _as_int(system.get("last_measure"), default=None)
            if first is not None and last is not None and last >= first:
                total += last - first + 1
            elif isinstance(system.get("bars"), list):
                total += len(system["bars"])
    return total


def _score_prep_layout_measure_count(prep: dict[str, Any]) -> int:
    movement_counts = [
        _as_int(movement.get("measure_count"), default=None)
        for movement in prep.get("movements") or []
        if isinstance(movement, dict)
    ]
    movement_counts = [count for count in movement_counts if count is not None]
    return int(sum(movement_counts)) if movement_counts else _score_prep_measure_count(prep)


def _movement_number_from_midi_url(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    clean = value.split("?", 1)[0].split("#", 1)[0].rstrip("/\\")
    filename = re.split(r"[/\\]", clean)[-1].lower()
    match = re.search(r"_(\d+)\.midi?$", filename)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _movement_measure_count(prep: dict[str, Any], movement_id: int) -> int | None:
    for movement in prep.get("movements") or []:
        if not isinstance(movement, dict):
            continue
        if _as_int(movement.get("id"), default=None) == movement_id:
            return _as_int(movement.get("measure_count"), default=None)
    return None


def _find_movement_matching_measure_count(
    prep: dict[str, Any],
    target: int,
    *,
    tolerance: int,
    hinted_movement_id: int | None = None,
) -> int | None:
    candidates: list[tuple[int, int]] = []
    existing_movement_ids: set[int] = set()
    for movement in prep.get("movements") or []:
        if not isinstance(movement, dict):
            continue
        movement_id = _as_int(movement.get("id"), default=None)
        count = _as_int(movement.get("measure_count"), default=None)
        if movement_id is None or count is None:
            continue
        existing_movement_ids.add(movement_id)
        delta = abs(count - target)
        candidates.append((delta, movement_id))
    if hinted_movement_id in existing_movement_ids:
        return hinted_movement_id
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def redistribute_movement_bars_to_midi(prep: dict[str, Any], *, movement_id: int, midi_bar_count: int) -> bool:
    """Redistribute one movement's system bars to the MIDI-derived bar count."""

    if midi_bar_count <= 0:
        return False
    movement = next(
        (
            item
            for item in prep.get("movements") or []
            if isinstance(item, dict) and _as_int(item.get("id"), default=None) == movement_id
        ),
        None,
    )
    if movement is None:
        return False

    systems: list[dict[str, Any]] = []
    for page in prep.get("pages") or []:
        if not isinstance(page, dict) or page.get("kind") != "music":
            continue
        for system in page.get("systems") or []:
            if not isinstance(system, dict):
                continue
            system_movement = _as_int(system.get("movement_id"), page.get("movement_id"), default=None)
            if system_movement == movement_id:
                systems.append(system)
    if not systems:
        return False

    weights = _movement_system_distribution_weights(systems)

    counts = _distribute_counts(total=midi_bar_count, weights=weights)
    measure = 1
    for system, count in zip(systems, counts):
        bbox = system.get("bbox") if isinstance(system.get("bbox"), dict) else {}
        try:
            x0 = float(bbox.get("x") or 0.0)
            x1 = x0 + float(bbox.get("w") or 1.0)
        except (TypeError, ValueError):
            x0, x1 = 0.0, 1.0
        x0 = max(0.0, min(1.0, x0))
        x1 = max(x0, min(1.0, x1))
        bars = []
        if count > 0:
            boundaries = _fit_existing_bar_boundaries_to_count(
                system,
                x0=x0,
                x1=x1,
                count=count,
                opening_system=measure == 1,
            )
            for offset in range(count):
                bars.append(
                    {
                        "bar_number": measure + offset,
                        "x_frac_start": round(boundaries[offset], 6),
                        "x_frac_end": round(boundaries[offset + 1], 6),
                    }
                )
            system["first_measure"] = measure
            system["last_measure"] = measure + count - 1
            measure += count
        else:
            system.pop("first_measure", None)
            system.pop("last_measure", None)
        system["bars"] = bars

    movement["first_measure"] = 1
    movement["last_measure"] = midi_bar_count
    movement["measure_count"] = midi_bar_count
    _refresh_page_ranges_from_systems(prep)
    prep.setdefault("_meta", {})["redistributed_movement_to_midi"] = {
        "movement_id": movement_id,
        "midi_bar_count": midi_bar_count,
        "system_counts": counts,
    }
    return True


def _fit_existing_bar_boundaries_to_count(
    system: dict[str, Any],
    *,
    x0: float,
    x1: float,
    count: int,
    opening_system: bool = False,
) -> list[float]:
    bars = system.get("bars") if isinstance(system.get("bars"), list) else []
    starts: list[float] = []
    candidates: list[float] = [x1]
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        try:
            start_value = float(bar.get("x_frac_start"))
        except (TypeError, ValueError):
            start_value = None
        if start_value is not None and x0 - 0.03 <= start_value <= x1 + 0.03:
            starts.append(max(x0, min(x1, start_value)))
        for key in ("x_frac_start", "x_frac_end"):
            try:
                value = float(bar.get(key))
            except (TypeError, ValueError):
                continue
            if x0 - 0.03 <= value <= x1 + 0.03:
                candidates.append(max(x0, min(x1, value)))
    music_start = min(starts) if starts else x0
    if music_start <= x0 + max(0.01, (x1 - x0) * 0.04):
        music_start = x0
    candidates.append(music_start)
    candidates = sorted(set(round(value, 6) for value in candidates))
    if len(candidates) == count + 1:
        return candidates
    fitted = [music_start]
    for index in range(1, count):
        expected = music_start + (x1 - music_start) * index / count
        if opening_system and count == 2:
            expected += (x1 - music_start) * 0.070
        inner = [value for value in candidates[1:-1] if value not in fitted]
        if inner:
            nearest = min(inner, key=lambda value: abs(value - expected))
            if abs(nearest - expected) <= max(0.035, (x1 - music_start) / count * 0.45):
                fitted.append(nearest)
                continue
        fitted.append(expected)
    fitted.append(x1)
    return sorted(max(0.0, min(1.0, value)) for value in fitted)


def _distribute_counts(*, total: int, weights: list[float]) -> list[int]:
    if not weights:
        return []
    weight_sum = sum(max(0.0, weight) for weight in weights)
    if weight_sum <= 0:
        weights = [1.0 for _ in weights]
        weight_sum = float(len(weights))
    average = weight_sum / len(weights)
    if average > 0 and max(abs(weight - average) for weight in weights) <= average * 0.08:
        base = total // len(weights)
        counts = [base for _ in weights]
        remaining = total - sum(counts)
        if remaining > 0:
            counts[-1] += 1
            remaining -= 1
        index = 1
        while remaining > 0 and index < len(counts) - 1:
            counts[index] += 1
            remaining -= 1
            index += 1
        index = 0
        while remaining > 0:
            counts[index % len(counts)] += 1
            remaining -= 1
            index += 1
        return counts
    counts: list[int] = []
    previous = 0
    cumulative = 0.0
    for weight in weights:
        cumulative += max(0.0, weight)
        boundary = int(math.floor(total * cumulative / weight_sum))
        counts.append(boundary - previous)
        previous = boundary
    if counts:
        counts[-1] += total - sum(counts)
    if total >= len(counts):
        for index, count in enumerate(counts):
            if count > 0:
                continue
            donor = max(range(len(counts)), key=lambda idx: (counts[idx], -idx))
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[index] = 1
    return counts


def _movement_system_distribution_weights(systems: list[dict[str, Any]]) -> list[float]:
    """Weight per system used to distribute MIDI bars across systems.

    Combines two signals (each contributes 50% of the final weight):
      1. **Width** of the system bbox — wider systems usually hold more bars.
      2. **Audiveris's per-system bar count** — noisy but a real signal of
         how many barlines were detected in this strip. Capped at [1, 6] to
         dampen extreme over/under-detection.

    The blend keeps narrow-system / dense-system signals where they're useful
    while keeping the result close to uniform when Audiveris was unreliable.
    """
    widths: list[float] = []
    for system in systems:
        bbox = system.get("bbox") if isinstance(system.get("bbox"), dict) else {}
        try:
            width = float(bbox.get("w") or 0.0)
        except (TypeError, ValueError):
            width = 0.0
        widths.append(width if 0.05 <= width <= 1.05 else 0.0)

    audiveris_counts: list[float] = []
    for system in systems:
        bars = system.get("bars") if isinstance(system.get("bars"), list) else []
        # Cap at 6 so a system where Audiveris over-detected 20 barlines doesn't dominate.
        # Floor at 2 because even sparse single-bar systems are rare in standard repertoire;
        # a count of 1 usually means Audiveris missed a barline.
        count = max(2.0, min(6.0, float(len(bars))))
        audiveris_counts.append(count)

    width_sum = sum(widths)
    count_sum = sum(audiveris_counts)
    positive_widths = [width for width in widths if width > 0]
    if positive_widths and max(positive_widths) - min(positive_widths) <= (sum(positive_widths) / len(positive_widths)) * 0.08:
        return widths
    if width_sum <= 0 or count_sum <= 0:
        # Fall back to whichever signal has signal, or uniform.
        if width_sum > 0:
            return widths
        if count_sum > 0:
            return audiveris_counts
        return [1.0 for _ in systems]

    # Normalize both to the same scale (each sums to len(systems)) then average.
    n = float(len(systems))
    width_norm = [w * n / width_sum for w in widths]
    count_norm = [c * n / count_sum for c in audiveris_counts]
    return [(width_norm[i] + count_norm[i]) / 2.0 for i in range(len(systems))]


def _refresh_page_ranges_from_systems(prep: dict[str, Any]) -> None:
    for page in prep.get("pages") or []:
        if not isinstance(page, dict) or page.get("kind") != "music":
            continue
        systems = [system for system in page.get("systems") or [] if isinstance(system, dict)]
        page["system_count"] = len(systems)
        firsts = [_as_int(system.get("first_measure"), default=None) for system in systems]
        lasts = [_as_int(system.get("last_measure"), default=None) for system in systems]
        firsts = [value for value in firsts if value is not None]
        lasts = [value for value in lasts if value is not None]
        if firsts and lasts:
            page["first_measure"] = min(firsts)
            page["last_measure"] = max(lasts)


def _apply_barline_cross_check(*, prep: dict[str, Any], page_images: list[bytes]) -> int:
    corrections = 0
    pages = [page for page in prep.get("pages") or [] if isinstance(page, dict)]
    by_page = {int(page["page"]): page for page in pages if isinstance(page.get("page"), int)}
    config = StaffDetectionConfig()
    for page_number in _score_prep_candidate_pages(prep, page_count=len(page_images)):
        page = by_page.get(page_number)
        if page is None or page.get("kind") not in (None, "music"):
            continue
        detected_systems = detect_staff_systems_from_image(page_images[page_number - 1], config=config)
        if detected_systems and _should_replace_with_detected_systems(page, detected_systems):
            _replace_page_systems_from_detection(page, detected_systems)
            corrections += 1

        for system in page.get("systems") or []:
            if not isinstance(system, dict) or not _plausible_normalized_bbox(system.get("bbox")):
                continue
            barlines = detect_barlines_in_system(page_images[page_number - 1], system["bbox"])
            if len(barlines) < 2:
                continue
            detected_count = len(barlines) - 1
            existing_bars = system.get("bars") if isinstance(system.get("bars"), list) else []
            declared_count = _system_measure_count(system) or len(existing_bars)
            if detected_count > max(declared_count, len(existing_bars)) or (not existing_bars and detected_count == declared_count):
                first = _as_int(system.get("first_measure"), default=None) or _infer_first_measure_for_system(page, system)
                _set_system_bars_from_boundaries(system, first_measure=first, boundaries=barlines)
                corrections += 1
    _refresh_page_measure_ranges(prep)
    return corrections


def _should_replace_with_detected_systems(page: dict[str, Any], detected_systems: list[dict[str, Any]]) -> bool:
    existing = [system for system in page.get("systems") or [] if isinstance(system, dict)]
    if not existing:
        return True
    if len(detected_systems) >= len(existing):
        return False
    existing_count = sum(_system_measure_count(system) for system in existing)
    avg_height = sum(float((system.get("bbox") or {}).get("h") or 0.0) for system in existing) / max(1, len(existing))
    detected_avg_height = sum(float((record.get("bbox") or {}).get("h") or 0.0) for record in detected_systems) / max(
        1, len(detected_systems)
    )
    return len(existing) >= len(detected_systems) * 2 and detected_avg_height > avg_height * 1.35 and existing_count > 0


def _redistribute_system_measures_to_total(prep: dict[str, Any], *, total_measures: int) -> bool:
    systems: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for page in prep.get("pages") or []:
        if not isinstance(page, dict) or page.get("kind") not in (None, "music"):
            continue
        for system in page.get("systems") or []:
            if isinstance(system, dict) and _plausible_normalized_bbox(system.get("bbox")):
                systems.append((page, system))
    if not systems or total_measures < len(systems):
        return False

    base = total_measures // len(systems)
    remainder = total_measures % len(systems)
    counts = [base + (1 if index >= len(systems) - remainder else 0) for index in range(len(systems))]
    measure = 1
    for (_, system), count in zip(systems, counts):
        system["first_measure"] = measure
        system["last_measure"] = measure + count - 1
        if len(system.get("bars") or []) != count:
            _set_system_bars_from_boundaries(
                system,
                first_measure=measure,
                boundaries=_even_bar_boundaries(system.get("bbox"), count=count),
            )
        else:
            for offset, bar in enumerate(system["bars"]):
                bar["bar_number"] = measure + offset
        measure += count
    _refresh_page_measure_ranges(prep)
    prep.setdefault("_meta", {})["measure_redistribution"] = {
        "target_total": total_measures,
        "system_counts": counts,
    }
    return True


def _even_bar_boundaries(bbox: Any, *, count: int) -> list[float]:
    if not isinstance(bbox, dict) or count <= 0:
        return [0.0, 1.0]
    x0 = float(bbox.get("x") or 0.0)
    x1 = x0 + float(bbox.get("w") or 1.0)
    left = max(0.0, min(1.0, x0))
    right = max(left, min(1.0, x1))
    step = (right - left) / count
    return [round(left + step * index, 6) for index in range(count)] + [round(right, 6)]


def _replace_page_systems_from_detection(page: dict[str, Any], detected_systems: list[dict[str, Any]]) -> None:
    old_systems = [system for system in page.get("systems") or [] if isinstance(system, dict)]
    page_first = _as_int(page.get("first_measure"), default=1) or 1
    page_last = _as_int(page.get("last_measure"), default=None)
    if page_last is None:
        page_last = max(((_as_int(system.get("last_measure"), default=0) or 0) for system in old_systems), default=page_first)
    total_measures = max(1, page_last - page_first + 1)
    weights = [max(1, round(total_measures / max(1, len(detected_systems)))) for _ in detected_systems]
    while sum(weights) > total_measures and any(weight > 1 for weight in weights):
        weights[max(range(len(weights)), key=lambda idx: weights[idx])] -= 1
    while sum(weights) < total_measures:
        weights[-1] += 1

    systems: list[dict[str, Any]] = []
    measure = page_first
    for index, record in enumerate(detected_systems, start=1):
        count = weights[index - 1]
        first = measure
        last = first + count - 1
        systems.append({"system_index": index, "first_measure": first, "last_measure": last, "bbox": record["bbox"]})
        measure = last + 1
    page["systems"] = systems
    page["system_count"] = len(systems)
    page["_staff_detection_replaced_systems"] = True


def _system_measure_count(system: dict[str, Any]) -> int:
    first = _as_int(system.get("first_measure"), default=None)
    last = _as_int(system.get("last_measure"), default=None)
    if first is None or last is None or last < first:
        return 0
    return last - first + 1


def _infer_first_measure_for_system(page: dict[str, Any], target: dict[str, Any]) -> int:
    first = _as_int(page.get("first_measure"), default=1) or 1
    for system in page.get("systems") or []:
        if system is target:
            return first
        if isinstance(system, dict):
            first += max(1, _system_measure_count(system))
    return first


def _set_system_bars_from_boundaries(system: dict[str, Any], *, first_measure: int, boundaries: list[float]) -> None:
    bars = []
    for offset, (x0, x1) in enumerate(zip(boundaries, boundaries[1:])):
        bars.append(
            {
                "bar_number": int(first_measure) + offset,
                "x_frac_start": round(float(x0), 6),
                "x_frac_end": round(float(x1), 6),
            }
        )
    system["bars"] = bars
    system["first_measure"] = int(first_measure)
    system["last_measure"] = int(first_measure) + len(bars) - 1


def _refresh_page_measure_ranges(prep: dict[str, Any]) -> None:
    movement_ranges: dict[int, list[int]] = {}
    for page in prep.get("pages") or []:
        if not isinstance(page, dict) or page.get("kind") not in (None, "music"):
            continue
        systems = [system for system in page.get("systems") or [] if isinstance(system, dict)]
        firsts = [_as_int(system.get("first_measure"), default=None) for system in systems]
        lasts = [_as_int(system.get("last_measure"), default=None) for system in systems]
        firsts = [value for value in firsts if value is not None]
        lasts = [value for value in lasts if value is not None]
        if firsts and lasts:
            page["first_measure"] = min(firsts)
            page["last_measure"] = max(lasts)
            movement_id = _as_int(page.get("movement_id"), default=None)
            if movement_id is not None:
                movement_ranges.setdefault(movement_id, []).extend([page["first_measure"], page["last_measure"]])
    for movement in prep.get("movements") or []:
        if not isinstance(movement, dict):
            continue
        movement_id = _as_int(movement.get("id"), default=None)
        values = movement_ranges.get(movement_id) if movement_id is not None else None
        if values:
            movement["first_measure"] = min(values)
            movement["last_measure"] = max(values)
            movement["measure_count"] = movement["last_measure"] - movement["first_measure"] + 1


def _as_int(*values: Any, default: int | None = None) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _infer_profile_from_instrument(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    if not s:
        return None
    if "piano" in s or "klavier" in s or "keyboard" in s or "harpsichord" in s or "organ" in s:
        return "piano"
    if "violin" in s and ("solo" in s or "alone" in s or "unaccompanied" in s):
        return "violin_solo"
    if s == "violin" or "violino" in s:
        return "violin_solo"
    if "viola" in s:
        return "viola_solo"
    if "cello" in s or "violoncello" in s:
        return "cello_solo"
    return None


def _set_prep_state(store: MasterclassStore, manifest: MasterclassManifest, state: str, *, error: str | None) -> None:
    manifest.metadata["score_prep_state"] = state
    manifest.metadata["score_prep_error"] = error
    manifest.metadata["score_prep_updated_at"] = datetime.now(UTC).isoformat()
    store.save(manifest)


def _apply_staff_detection_fallback(
    *,
    prep: dict[str, Any],
    page_images: list[bytes],
    storage: ObjectStorage,
    masterclass_store: MasterclassStore,
    manifest: MasterclassManifest,
) -> None:
    pages = prep.setdefault("pages", [])
    if not isinstance(pages, list):
        pages = []
        prep["pages"] = pages

    by_page: dict[int, dict[str, Any]] = {}
    for entry in pages:
        if isinstance(entry, dict) and isinstance(entry.get("page"), int):
            by_page[entry["page"]] = entry

    fallback_pages: list[int] = []
    for page_number in _score_prep_candidate_pages(prep, page_count=len(page_images)):
        page = by_page.get(page_number)
        if page is None:
            page = {"page": page_number, "kind": "music"}
            pages.append(page)
            by_page[page_number] = page
        if _page_needs_staff_detection_fallback(page):
            fallback_pages.append(page_number)

    if not fallback_pages:
        return

    counts: dict[str, int] = {}
    config = StaffDetectionConfig()
    for page_number in fallback_pages:
        records = detect_staff_systems_from_image(page_images[page_number - 1], config=config)
        if not records:
            counts[str(page_number)] = 0
            continue

        systems: list[dict[str, Any]] = []
        for record in records:
            system_index = int(record["system_index"])
            rel = f"reference/score_systems/page-{page_number:03d}-system-{system_index:02d}.png"
            key = masterclass_store.artifact_key(manifest.masterclass, rel)
            storage.write_bytes(key, record["crop_png"], content_type="image/png")
            manifest.artifacts[rel] = key
            systems.append({"system_index": system_index, "bbox": record["bbox"]})

        page = by_page[page_number]
        page["systems"] = systems
        page["system_count"] = len(systems)
        page["_staff_detection_fallback"] = True
        counts[str(page_number)] = len(systems)

    if counts:
        prep.setdefault("_meta", {})
        prep["_staff_detection_fallback"] = {
            "pages": sorted(counts.keys(), key=int),
            "system_counts": counts,
        }
        manifest.metadata["staff_detection_fallback_pages"] = sorted(counts.keys(), key=int)
        manifest.metadata["staff_detection_fallback_system_counts"] = counts


def _score_prep_candidate_pages(prep: dict[str, Any], *, page_count: int) -> list[int]:
    candidates: list[int] = []
    pages = prep.get("pages") or []
    if isinstance(pages, list):
        for entry in pages:
            if not isinstance(entry, dict):
                continue
            page = entry.get("page")
            if not isinstance(page, int) or page < 1 or page > page_count:
                continue
            kind = entry.get("kind")
            if kind == "music" or entry.get("systems") or entry.get("system_count"):
                candidates.append(page)

    if not candidates:
        first_music = prep.get("first_music_page")
        if isinstance(first_music, int) and 1 <= first_music <= page_count:
            candidates.extend(range(first_music, page_count + 1))

    return sorted(set(candidates))


def _page_needs_staff_detection_fallback(page: dict[str, Any]) -> bool:
    systems = page.get("systems")
    declared_count = page.get("system_count")
    if not isinstance(systems, list) or not systems:
        return page.get("kind") == "music" or bool(declared_count)
    if isinstance(declared_count, int) and declared_count > 0 and len(systems) != declared_count:
        return True
    return any(not _plausible_normalized_bbox(system.get("bbox")) for system in systems if isinstance(system, dict))


def _plausible_normalized_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, dict):
        return False
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return False
    values = (x, y, w, h)
    if not all(math.isfinite(value) for value in values):
        return False
    if x < -0.001 or y < -0.001 or w <= 0.02 or h <= 0.02:
        return False
    if x > 1.001 or y > 1.001 or w > 1.001 or h > 0.7:
        return False
    if x + w > 1.02 or y + h > 1.02:
        return False
    return True


def select_score_pages_for_lesson(
    *,
    storage: ObjectStorage,
    masterclass: MasterclassManifest,
    first_measure: int | None,
    last_measure: int | None,
    max_pages: int = 8,
) -> tuple[list[bytes], list[dict[str, Any]]]:
    """Pick the most relevant rasterized score pages and per-page layout.

    Returns ``(page_pngs, layout)`` where ``layout[i]`` describes the page
    backing ``page_pngs[i]`` (page number, system list with bboxes/measure
    ranges, etc.). Falls back to all music pages (capped) if no measure
    metadata is available or if the prep document doesn't carry per-page
    measure ranges.
    """

    prep_key = masterclass.artifacts.get("reference/score_prep.json")
    if not prep_key:
        return [], []
    try:
        prep = storage.read_json(prep_key) or {}
    except (FileNotFoundError, ValueError):
        return [], []

    pages_meta = prep.get("pages") or []
    by_page: dict[int, dict[str, Any]] = {}
    for entry in pages_meta:
        if isinstance(entry, dict) and isinstance(entry.get("page"), int):
            by_page[entry["page"]] = entry

    music_page_numbers: list[int] = []
    for entry in pages_meta:
        if not isinstance(entry, dict):
            continue
        page = entry.get("page")
        kind = entry.get("kind") or ""
        if not isinstance(page, int):
            continue
        if kind and kind != "music":
            continue
        music_page_numbers.append(page)
    if not music_page_numbers:
        first_music = prep.get("first_music_page")
        page_count = int(prep.get("page_count") or 0)
        if isinstance(first_music, int) and page_count >= first_music:
            music_page_numbers = list(range(first_music, page_count + 1))

    if first_measure and last_measure:
        narrowed: list[int] = []
        for entry in pages_meta:
            if not isinstance(entry, dict):
                continue
            page = entry.get("page")
            kind = entry.get("kind") or ""
            if not isinstance(page, int) or (kind and kind != "music"):
                continue
            page_first = entry.get("first_measure")
            page_last = entry.get("last_measure")
            if isinstance(page_first, int) and isinstance(page_last, int):
                if page_first <= last_measure and page_last >= first_measure:
                    narrowed.append(page)
        if narrowed:
            music_page_numbers = sorted(set(narrowed))

    music_page_numbers = sorted(set(music_page_numbers))[:max_pages]

    pngs: list[bytes] = []
    layout: list[dict[str, Any]] = []
    for page in music_page_numbers:
        rel = f"reference/score_pages/page-{page:03d}.png"
        key = masterclass.artifacts.get(rel)
        if not key or not storage.exists(key):
            continue
        pngs.append(storage.read_bytes(key))
        meta = by_page.get(page) or {}
        layout.append({
            "page": page,
            "first_measure": meta.get("first_measure"),
            "last_measure": meta.get("last_measure"),
            "system_count": meta.get("system_count"),
            "systems": meta.get("systems") or [],
            "movement_id": meta.get("movement_id"),
        })
    return pngs, layout
