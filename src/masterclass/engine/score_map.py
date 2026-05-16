from __future__ import annotations

import bisect
import io
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from masterclass.core.artifact_catalog import ArtifactCatalog
from masterclass.core.masterclasses import MasterclassStore
from masterclass.core.models import MasterclassManifest, SessionManifest, TenantContext
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage


_KEY_FLATS = {
    "f_major": {"a#": "Bb"},
    "bb_major": {"a#": "Bb", "d#": "Eb"},
    "eb_major": {"a#": "Bb", "d#": "Eb", "g#": "Ab"},
    "ab_major": {"a#": "Bb", "d#": "Eb", "g#": "Ab", "c#": "Db"},
    "db_major": {"a#": "Bb", "d#": "Eb", "g#": "Ab", "c#": "Db", "f#": "Gb"},
    "gb_major": {"a#": "Bb", "d#": "Eb", "g#": "Ab", "c#": "Db", "f#": "Gb"},
    "d_minor": {"a#": "Bb"},
    "g_minor": {"a#": "Bb", "d#": "Eb"},
    "c_minor": {"a#": "Bb", "d#": "Eb", "g#": "Ab"},
    "f_minor": {"a#": "Bb", "d#": "Eb", "g#": "Ab", "c#": "Db"},
    "bb_minor": {"a#": "Bb", "d#": "Eb", "g#": "Ab", "c#": "Db", "f#": "Gb"},
    "eb_minor": {"a#": "Bb", "d#": "Eb", "g#": "Ab", "c#": "Db", "f#": "Gb"},
}


@dataclass(frozen=True)
class ScoreMapConfig:
    first_bar_x_pad_frac: float = 0.06
    trailing_x_pad_frac: float = 0.04
    beats_per_bar: float = 4.0
    chord_time_round_digits: int = 4
    align_score_time_round_digits: int = 3
    hmm_score_time_round_digits: int = 3  # deprecated alias for align_score_time_round_digits
    copy_score_images_to_session: bool = True


@dataclass
class ScoreMapResult:
    score_map: dict[str, Any]
    bars: list[dict[str, Any]]
    notes: list[dict[str, Any]]
    systems: list[dict[str, Any]]
    masterclass_id: str
    score_map_key: str | None = None


def build_score_map(
    *,
    storage: ObjectStorage,
    masterclass_store: MasterclassStore,
    store: SessionStore,
    manifest: SessionManifest,
    config: ScoreMapConfig | None = None,
) -> ScoreMapResult:
    """Build a storage-scoped score_map.json for a lesson session."""

    config = config or ScoreMapConfig()
    masterclass_id = str(manifest.metadata.get("masterclass_id") or "").strip()
    if not masterclass_id:
        raise ValueError("session manifest metadata is missing masterclass_id")

    ctx = TenantContext(manifest.session.tenant_id, manifest.session.user_id)
    masterclass = masterclass_store.load_by_id(ctx, masterclass_id)
    score_prep_key = _artifact(masterclass, "reference/score_prep.json") or _artifact(
        manifest, "masterclass/reference/score_prep.json"
    )
    if not score_prep_key or not storage.exists(score_prep_key):
        raise ValueError("masterclass is missing reference/score_prep.json")
    score_prep = storage.read_json(score_prep_key) or {}

    # Score-note source: parse MusicXML directly (audio-truth has the
    # canonical loader). MIDI was retired with the audio-truth migration -
    # masterclasses created from PDF+OMR never have a reference/midi
    # artifact and synthesizing one round-tripped pitch quantization
    # errors we can avoid by reading MusicXML once.
    from masterclass.engine.audio_truth import _load_score_notes_from_musicxml
    xml_key = _find_musicxml_key(storage, masterclass, manifest)
    if not xml_key:
        raise ValueError(
            "masterclass is missing reference MusicXML; score_prep must run first"
        )
    score_notes = _load_score_notes_from_musicxml(storage.read_bytes(xml_key))
    if not score_notes:
        raise ValueError(f"MusicXML at {xml_key} parsed to zero notes")

    hinted_movement_id = _as_int(
        manifest.metadata.get("played_movement_id"),
        masterclass.metadata.get("played_movement_id"),
        default=None,
    )
    played_lo, played_hi = _played_measure_range(manifest, score_prep, hinted_movement_id=hinted_movement_id)
    movement = _select_movement(score_prep, manifest, played_lo, played_hi, hinted_movement_id=hinted_movement_id)
    score_key = _movement_key(score_prep, movement)
    total_measures = _as_int(
        (movement or {}).get("measure_count"),
        (movement or {}).get("last_measure"),
        score_prep.get("total_measures"),
        default=None,
    )

    movement_id = _as_int((movement or {}).get("id"), hinted_movement_id, default=None)
    layout_systems = _layout_systems(
        score_prep,
        played_lo=played_lo,
        played_hi=played_hi,
        movement_id=movement_id,
        config=config,
    )
    systems = _stage_system_images(
        storage=storage,
        store=store,
        manifest=manifest,
        masterclass=masterclass,
        systems=layout_systems,
        played_lo=played_lo,
        played_hi=played_hi,
        config=config,
    )
    bars = _build_bars(layout_systems, systems, played_lo=played_lo, played_hi=played_hi, config=config)
    notes = _build_notes(
        score_notes=score_notes,
        bars=bars,
        played_lo=played_lo,
        played_hi=played_hi,
        score_key=score_key,
        aligned_notes=_load_aligned_notes(storage, manifest),
        config=config,
    )
    total_measures = total_measures or max((int(n.get("measure") or 0) for n in score_notes), default=0) or len(notes)

    score_map = {
        "schema_version": 1,
        "score_id": masterclass_id,
        "masterclass_id": masterclass_id,
        "movement": manifest.movement or masterclass.movement or (movement or {}).get("title"),
        "key": score_key,
        "instrument": manifest.instrument or masterclass.instrument or score_prep.get("instrument"),
        "first_measure": played_lo,
        "last_measure": played_hi,
        "played_lo": played_lo,
        "played_hi": played_hi,
        "total_measures": total_measures,
        "alignment_source": "audio_truth" if _has_aligned_perf_times(notes) else "midi_only",
        "systems": systems,
        "bars": bars,
        "notes": notes,
        "notes_help": [
            "Each note has stable note_id, MIDI measure/beat, score_time, perf_time, names, MIDI pitches, and score x fractions.",
            "perf_time is read from the canonical aligned-notes artifact (analysis/audio_truth_matched_notes.json, with legacy aliases analysis/aligned_notes.json and analysis/hmm_aligned_notes.json) when present; otherwise it is null.",
            "Bar image/bbox entries are derived from masterclass reference/score_prep.json systems.",
        ],
        "_meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "score_prep_key": score_prep_key,
            "score_source_key": xml_key,
            "config": asdict(config),
        },
    }
    return ScoreMapResult(
        score_map=score_map,
        bars=bars,
        notes=notes,
        systems=systems,
        masterclass_id=masterclass_id,
    )


def persist_score_map(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    result: ScoreMapResult,
) -> None:
    """Persist score/score_map.json and stamp the session manifest."""

    key = store.artifact_key(manifest.session, "score/score_map.json")
    storage.write_json(key, result.score_map)
    manifest.artifacts["score/score_map.json"] = key
    manifest.metadata["score_map_state"] = "ready"
    manifest.metadata["score_map_bar_count"] = len(result.bars)
    manifest.metadata["score_map_note_count"] = len(result.notes)
    manifest.metadata["score_map_generated_at"] = datetime.now(UTC).isoformat()
    store.save(manifest)
    result.score_map_key = key


def spell_pitch_name(name: str, key: str | None) -> str:
    if not name or not key or "#" not in name:
        return name
    key_norm = key.strip().lower().replace(" ", "_").replace("-", "_")
    flats = _KEY_FLATS.get(key_norm)
    if not flats:
        return name
    pc, octave = "", ""
    for ch in name:
        if ch.isdigit() or ch == "-":
            octave += ch
        else:
            pc += ch
    return flats.get(pc.lower(), pc) + octave


def spell_pitch_names(names: list[str], key: str | None) -> list[str]:
    return [spell_pitch_name(name, key) for name in names]


def make_note_id(measure: int, beat: float, names: list[str]) -> str:
    """PoC-compatible stable note id: m{measure}_b{beat:.2f}_{pitch(+pitch...)}."""

    return f"m{measure}_b{beat:.2f}_{'+'.join(names)}"


def _artifact(manifest: MasterclassManifest | SessionManifest, name: str) -> str | None:
    value = manifest.artifacts.get(name)
    return str(value) if value else None


def _find_midi_key(storage: ObjectStorage, masterclass: MasterclassManifest, manifest: SessionManifest) -> str | None:
    candidates = [
        _artifact(masterclass, "reference/midi"),
        _artifact(masterclass, "reference/midi/auto.mid"),
        _artifact(manifest, "masterclass/reference/midi"),
        _artifact(manifest, "masterclass/reference/midi/auto.mid"),
    ]
    prefix = f"tenant/{masterclass.masterclass.tenant_id}/users/{masterclass.masterclass.user_id}/masterclasses/{masterclass.masterclass.masterclass_id}/reference/midi"
    candidates.extend(key for key in storage.list_keys(prefix) if key.lower().endswith((".mid", ".midi")))
    for key in candidates:
        if key and storage.exists(key):
            return key
    return None


def _find_musicxml_key(storage: ObjectStorage, masterclass: MasterclassManifest, manifest: SessionManifest) -> str | None:
    """Locate the MusicXML reference. Delegates to ArtifactCatalog so the
    extension priority list (.musicxml -> .mxl -> bare) is defined in one
    place. Both manifests are consulted because some lessons only stamp
    the masterclass manifest with the reference key."""
    key = ArtifactCatalog(manifest, masterclass).musicxml()
    if key and storage.exists(key):
        return key
    return None


def _load_or_synthesize_midi(*args, **kwargs) -> bytes | None:
    """Deprecated. score_map now reads MusicXML directly via
    audio_truth._load_score_notes_from_musicxml. Retained as a stub in
    case external callers import the symbol."""
    return None


def _played_measure_range(
    manifest: SessionManifest, score_prep: dict[str, Any], *, hinted_movement_id: int | None = None
) -> tuple[int, int]:
    first = _as_int(manifest.metadata.get("first_measure"), default=None)
    last = _as_int(manifest.metadata.get("last_measure"), default=None)
    auto_first = _as_int(manifest.metadata.get("auto_detected_first_measure"), default=None)
    auto_last = _as_int(manifest.metadata.get("auto_detected_last_measure"), default=None)
    movement = _select_movement(
        score_prep,
        manifest,
        first or auto_first or 1,
        last or auto_last or first or auto_first or 1,
        hinted_movement_id=hinted_movement_id,
    )

    # Pull the score's natural extent from the movement (preferred) or aggregate
    # the per-page measure ranges so we never silently clamp to "1..1" when the
    # user submits a lesson without picking a measure window.
    score_first = _as_int((movement or {}).get("first_measure"), default=None)
    score_last = _as_int((movement or {}).get("last_measure"), default=None)
    if score_first is None or score_last is None:
        page_lasts = [
            _as_int(p.get("last_measure"), default=None)
            for p in (score_prep.get("pages") or [])
            if isinstance(p, dict) and (p.get("kind") == "music" or not p.get("kind"))
        ]
        page_lasts = [p for p in page_lasts if p]
        if page_lasts:
            score_first = score_first or 1
            score_last = score_last or max(page_lasts)

    first = first or auto_first or score_first or 1
    last = last or auto_last or score_last or first
    if last < first:
        raise ValueError(f"invalid played measure range {first}..{last}")
    return first, last


def _select_movement(
    score_prep: dict[str, Any],
    manifest: SessionManifest,
    played_lo: int,
    played_hi: int,
    *,
    hinted_movement_id: int | None = None,
) -> dict[str, Any] | None:
    movements = [m for m in score_prep.get("movements") or [] if isinstance(m, dict)]
    if not movements:
        return None
    target = (manifest.movement or "").strip().lower()
    if target:
        for movement in movements:
            title = str(movement.get("title") or "").strip().lower()
            if title and (title == target or target in title or title in target):
                return movement
    for movement in movements:
        movement_id = _as_int(movement.get("id"), default=None)
        if movement_id is not None and hinted_movement_id == movement_id:
            return movement
    for movement in movements:
        first = _as_int(movement.get("first_measure"), default=None)
        last = _as_int(movement.get("last_measure"), default=None)
        if first is not None and last is not None and first <= played_hi and last >= played_lo:
            return movement
    return movements[0]


def _movement_key(score_prep: dict[str, Any], movement: dict[str, Any] | None) -> str | None:
    key = (movement or {}).get("key_signature") or (movement or {}).get("key") or score_prep.get("key")
    if not key:
        return None
    return str(key).strip().lower().replace(" ", "_").replace("-", "_")


def _layout_systems(
    score_prep: dict[str, Any],
    *,
    played_lo: int,
    played_hi: int,
    movement_id: int | None = None,
    config: ScoreMapConfig,
) -> list[dict[str, Any]]:
    systems: list[dict[str, Any]] = []
    for page in score_prep.get("pages") or []:
        if not isinstance(page, dict) or page.get("kind") not in (None, "music"):
            continue
        page_no = _as_int(page.get("page"), default=None)
        if not page_no:
            continue
        for system in page.get("systems") or []:
            if not isinstance(system, dict):
                continue
            system_no = _as_int(system.get("system_index"), system.get("system"), default=None)
            if not system_no:
                continue
            system_movement_id = _as_int(system.get("movement_id"), page.get("movement_id"), default=None)
            if movement_id is not None and system_movement_id is not None and system_movement_id != movement_id:
                continue
            first = _as_int(system.get("first_measure"), default=None)
            last = _as_int(system.get("last_measure"), default=None)
            if first is None or last is None:
                first = _as_int(page.get("first_measure"), default=played_lo) or played_lo
                last = _as_int(page.get("last_measure"), default=played_hi) or played_hi
            if last < played_lo or first > played_hi:
                continue
            systems.append(
                {
                    "system": page_no * 100 + system_no,
                    "page": page_no,
                    "system_on_page": system_no,
                    "movement_id": system_movement_id,
                    "first_measure": first,
                    "last_measure": last,
                    "bbox": system.get("bbox"),
                    "bars": system.get("bars") if isinstance(system.get("bars"), list) else None,
                    "first_bar_x_pad_frac": float(system.get("first_bar_x_pad_frac", config.first_bar_x_pad_frac)),
                    "trailing_x_pad_frac": float(system.get("trailing_x_pad_frac", config.trailing_x_pad_frac)),
                }
            )
    systems.sort(key=lambda row: (int(row["page"]), int(row["system_on_page"])))
    return systems


def _stage_system_images(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    masterclass: MasterclassManifest,
    systems: list[dict[str, Any]],
    played_lo: int,
    played_hi: int,
    config: ScoreMapConfig,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    copied: set[str] = set()
    for system in systems:
        page = int(system["page"])
        system_on_page = int(system["system_on_page"])
        crop_rel = f"reference/score_systems/page-{page:03d}-system-{system_on_page:02d}.png"
        page_rel = f"reference/score_pages/page-{page:03d}.png"
        masterclass_prefix = (
            f"tenant/{masterclass.masterclass.tenant_id}/users/{masterclass.masterclass.user_id}"
            f"/masterclasses/{masterclass.masterclass.masterclass_id}"
        )
        src_key = masterclass.artifacts.get(crop_rel) or f"{masterclass_prefix}/{crop_rel}"
        image_name = f"system_p{page:03d}_s{system_on_page:02d}.png"
        image_kind = "system_crop"
        if not src_key or not storage.exists(src_key):
            src_key = masterclass.artifacts.get(page_rel) or f"{masterclass_prefix}/{page_rel}"
            image_name = f"page-{page:03d}.png"
            image_kind = "page"
        image = None
        if src_key and storage.exists(src_key):
            dst_rel = f"score/{image_name}"
            dst_key = store.artifact_key(manifest.session, dst_rel)
            if config.copy_score_images_to_session and dst_key not in copied:
                storage.write_bytes(dst_key, storage.read_bytes(src_key), content_type="image/png")
                manifest.artifacts[dst_rel] = dst_key
                copied.add(dst_key)
            image = dst_rel
        entry = dict(system)
        entry["image"] = image
        entry["image_kind"] = image_kind
        entry["bars"] = list(range(max(int(system["first_measure"]), played_lo), min(int(system["last_measure"]), played_hi) + 1))
        out.append(entry)
    return out


def _build_bars(
    layout_systems: list[dict[str, Any]],
    systems: list[dict[str, Any]],
    *,
    played_lo: int,
    played_hi: int,
    config: ScoreMapConfig,
) -> list[dict[str, Any]]:
    system_by_id = {int(system["system"]): system for system in systems}
    bars: list[dict[str, Any]] = []
    for layout in layout_systems:
        first = max(played_lo, int(layout["first_measure"]))
        last = min(played_hi, int(layout["last_measure"]))
        if last < first:
            continue
        sys_id = int(layout["system"])
        system = system_by_id.get(sys_id, layout)
        n_bars = max(1, last - first + 1)
        explicit_bars = _explicit_bars_for_range(layout.get("bars"), first=first, last=last, system_bbox=layout.get("bbox"))
        if explicit_bars:
            for bar in explicit_bars:
                x0 = float(bar["x_frac_start"])
                x1 = float(bar["x_frac_end"])
                bars.append(
                    {
                        "measure": int(bar["bar_number"]),
                        "midi_measure": int(bar["bar_number"]),
                        "page": int(layout["page"]),
                        "system": sys_id,
                        "system_on_page": int(layout["system_on_page"]),
                        "image": system.get("image"),
                        "bbox": _bar_bbox(layout.get("bbox"), x0, x1),
                        "system_bbox": layout.get("bbox"),
                        "highlight_x_frac": _relative_highlight(layout.get("bbox"), x0, x1),
                        "alignment_source": "score_prep_bars",
                    }
                )
            continue
        head_pad = float(layout.get("first_bar_x_pad_frac", config.first_bar_x_pad_frac))
        tail_pad = float(layout.get("trailing_x_pad_frac", config.trailing_x_pad_frac))
        usable = max(0.05, 1.0 - head_pad - tail_pad)
        per_bar = usable / n_bars
        for offset, measure in enumerate(range(first, last + 1)):
            x0 = head_pad + offset * per_bar
            x1 = x0 + per_bar
            bars.append(
                {
                    "measure": measure,
                    "midi_measure": measure,
                    "page": int(layout["page"]),
                    "system": sys_id,
                    "system_on_page": int(layout["system_on_page"]),
                    "image": system.get("image"),
                    "bbox": layout.get("bbox"),
                    "system_bbox": layout.get("bbox"),
                    "first_bar_x_pad_frac": head_pad,
                    "trailing_x_pad_frac": tail_pad,
                    "highlight_x_frac": [round(x0, 4), round(x1, 4)],
                    "alignment_source": "score_prep",
                }
            )
    return bars


def _explicit_bars_for_range(raw_bars: Any, *, first: int, last: int, system_bbox: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_bars, list):
        return []
    bars: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_bars):
        if not isinstance(raw, dict):
            continue
        number = _as_int(raw.get("bar_number"), raw.get("measure"), default=first + index)
        x0 = _as_float(raw.get("x_frac_start"), raw.get("x_start"), default=None)
        x1 = _as_float(raw.get("x_frac_end"), raw.get("x_end"), default=None)
        if number is None or x0 is None or x1 is None or number < first or number > last:
            continue
        if x1 <= x0:
            continue
        if _looks_relative_to_system(system_bbox, x0, x1):
            bbox = system_bbox if isinstance(system_bbox, dict) else {}
            sx = float(bbox.get("x") or 0.0)
            sw = float(bbox.get("w") or 1.0)
            x0 = sx + x0 * sw
            x1 = sx + x1 * sw
        bars.append({"bar_number": number, "x_frac_start": max(0.0, x0), "x_frac_end": min(1.0, x1)})
    bars.sort(key=lambda row: int(row["bar_number"]))
    return bars


def _looks_relative_to_system(system_bbox: Any, x0: float, x1: float) -> bool:
    if not isinstance(system_bbox, dict):
        return False
    sx = _as_float(system_bbox.get("x"), default=None)
    sw = _as_float(system_bbox.get("w"), default=None)
    if sx is None or sw is None:
        return False
    return 0.0 <= x0 <= 1.0 and 0.0 <= x1 <= 1.0 and (x0 < sx - 0.02 or x1 > sx + sw + 0.02)


def _bar_bbox(system_bbox: Any, x0: float, x1: float) -> dict[str, float] | Any:
    if not isinstance(system_bbox, dict):
        return system_bbox
    y = float(system_bbox.get("y") or 0.0)
    h = float(system_bbox.get("h") or 0.0)
    return {
        "x": round(max(0.0, min(1.0, x0)), 6),
        "y": round(max(0.0, min(1.0, y)), 6),
        "w": round(max(0.0, min(1.0, x1) - max(0.0, min(1.0, x0))), 6),
        "h": round(max(0.0, min(1.0, h)), 6),
    }


def _relative_highlight(system_bbox: Any, x0: float, x1: float) -> list[float]:
    if not isinstance(system_bbox, dict):
        return [round(x0, 4), round(x1, 4)]
    sx = float(system_bbox.get("x") or 0.0)
    sw = max(1e-6, float(system_bbox.get("w") or 1.0))
    return [round(max(0.0, min(1.0, (x0 - sx) / sw)), 4), round(max(0.0, min(1.0, (x1 - sx) / sw)), 4)]


def _build_notes(
    *,
    score_notes: list[dict[str, Any]],
    bars: list[dict[str, Any]],
    played_lo: int,
    played_hi: int,
    score_key: str | None,
    aligned_notes: list[dict[str, Any]],
    config: ScoreMapConfig,
) -> list[dict[str, Any]]:
    """Build per-note score-map rows from a list of normalized score notes.

    Each ``score_notes`` row has the shape produced by
    ``audio_truth._load_score_notes_from_musicxml``: at minimum
    ``score_time_sec``, ``midi_pitch``, ``measure``, plus optional
    ``duration_sec`` / ``track_name`` / ``staff_index``.

    Previously this function parsed pretty_midi and derived measure from
    ``midi.get_downbeats()``. We now trust the MusicXML ``measure`` field
    directly (already the source of truth in audio-truth's matcher) and
    derive downbeat times as the min score_time per measure.
    """
    import pretty_midi  # only for note_number_to_name; no MIDI parsing

    if not score_notes:
        raise RuntimeError("score has no notes (MusicXML parse returned empty)")

    notes_by_measure: dict[int, list[dict[str, Any]]] = {}
    for n in score_notes:
        m = int(n.get("measure") or 0)
        if m <= 0:
            continue
        notes_by_measure.setdefault(m, []).append(n)

    # Derive downbeats: first score_time per measure, in measure order.
    downbeats: list[float] = []
    measure_order: list[int] = sorted(notes_by_measure)
    for m in measure_order:
        downbeats.append(min(float(n["score_time_sec"]) for n in notes_by_measure[m]))
    # End-of-piece sentinel (next-measure boundary) for last-measure bar_dur.
    score_end = max(
        float(n.get("score_time_sec", 0.0)) + float(n.get("duration_sec", 0.0) or 0.0)
        for n in score_notes
    )

    events_by_score_time: dict[float, list[dict[str, Any]]] = {}
    for note in score_notes:
        key = round(float(note["score_time_sec"]), config.chord_time_round_digits)
        events_by_score_time.setdefault(key, []).append({
            "start": float(note["score_time_sec"]),
            "pitch": int(note["midi_pitch"]),
            "name": pretty_midi.note_number_to_name(int(note["midi_pitch"])),
            "measure": int(note.get("measure") or 0),
        })

    align_lookup, align_conf, align_dwell = _aligned_lookups(aligned_notes, config=config)
    bar_by_measure = {int(bar["midi_measure"]): bar for bar in bars}
    events_by_measure: dict[int, list[float]] = {}
    for score_time, evs in events_by_score_time.items():
        measure = int(evs[0]["measure"]) if evs and evs[0].get("measure") else 1
        if played_lo <= measure <= played_hi:
            events_by_measure.setdefault(measure, []).append(score_time)

    out: list[dict[str, Any]] = []
    for measure in sorted(events_by_measure):
        bar = bar_by_measure.get(measure)
        if not bar:
            continue
        event_times = sorted(events_by_measure[measure])
        n_events = max(1, len(event_times))
        x0, x1 = bar.get("highlight_x_frac", [0.0, 1.0])
        bar_width = float(x1) - float(x0)
        # downbeats list is indexed by position-in-played-order; map by
        # measure number via measure_order.
        try:
            mi = measure_order.index(measure)
            db_start = downbeats[mi]
            db_end = downbeats[mi + 1] if mi + 1 < len(downbeats) else score_end
        except ValueError:
            db_start = event_times[0]
            db_end = score_end
        bar_dur = max(1e-6, float(db_end) - float(db_start))
        for index, score_time in enumerate(event_times):
            notes_at = events_by_score_time[score_time]
            names = spell_pitch_names([str(note["name"]) for note in notes_at], score_key)
            pitches = [int(note["pitch"]) for note in notes_at]
            beat = 1.0 + float(config.beats_per_bar) * (float(score_time) - float(db_start)) / bar_dur
            nx0 = float(x0) + (index / n_events) * bar_width
            nx1 = float(x0) + ((index + 1) / n_events) * bar_width
            lookup_key = round(float(score_time), config.align_score_time_round_digits)
            perf_time = align_lookup.get(lookup_key)
            out.append(
                {
                    "note_id": make_note_id(measure, beat, names),
                    "system": bar["system"],
                    "measure": bar["measure"],
                    "midi_measure": measure,
                    "beat_in_bar": round(beat, 2),
                    "score_time": round(float(score_time), 3),
                    "perf_time": round(float(perf_time), 3) if perf_time is not None else None,
                    "x_frac": round(nx0, 4),
                    "x_frac_end": round(nx1, 4),
                    "names": names,
                    "pitch_midi": pitches[0] if len(pitches) == 1 else pitches,
                    "midi_pitches": pitches,
                    "is_chord": len(notes_at) > 1,
                    "align_confidence": align_conf.get(lookup_key, "none"),
                    "align_dwell_sec": align_dwell.get(lookup_key, 0.0),
                    # Legacy aliases; will be removed once consumers migrate
                    # to ``align_confidence`` / ``align_dwell_sec``.
                    "hmm_confidence": align_conf.get(lookup_key, "none"),
                    "hmm_dwell_sec": align_dwell.get(lookup_key, 0.0),
                    "interpolated": False,
                    "is_bar_anchor": index == 0,
                    "confidence": "high" if index == 0 else align_conf.get(lookup_key, "none"),
                }
            )
    out.sort(key=lambda row: (int(row["midi_measure"]), float(row["score_time"]), str(row["note_id"])))
    return out


def _load_aligned_notes(storage: ObjectStorage, manifest: SessionManifest) -> list[dict[str, Any]]:
    """Score_map's per-note source. Reads the unified aligned-notes
    accessor (preferring audio_truth_matched_notes.json) and serializes
    the typed dataclasses back to dicts for the legacy lookup helpers
    below."""
    from masterclass.engine.aligned_notes import load_aligned_notes
    return [n.to_dict() for n in load_aligned_notes(storage, manifest)]


def _aligned_lookups(
    aligned_notes: list[dict[str, Any]], *, config: ScoreMapConfig
) -> tuple[dict[float, float], dict[float, str], dict[float, float]]:
    perf: dict[float, float] = {}
    conf: dict[float, str] = {}
    dwell: dict[float, float] = {}
    for note in aligned_notes:
        score_time = _as_float(note.get("score_time_sec"), note.get("score_time_in_movement"), note.get("score_time"), default=None)
        perf_time = _as_float(note.get("performed_time_sec"), note.get("perf_time"), default=None)
        if score_time is None or perf_time is None:
            continue
        key = round(score_time, config.align_score_time_round_digits)
        perf[key] = perf_time
        conf[key] = str(note.get("confidence") or "low")
        dwell[key] = float(_as_float(note.get("dwell_sec"), default=0.0) or 0.0)
    return perf, conf, dwell


# Deprecated alias; remove once no external callers import the old name.
_hmm_lookups = _aligned_lookups


def _midi_measure_count(*args, **kwargs) -> int:
    """Deprecated: score_map derives total_measures from MusicXML notes."""
    return 0


def _has_aligned_perf_times(notes: list[dict[str, Any]]) -> bool:
    return any(note.get("perf_time") is not None for note in notes)


# Deprecated alias for callers still using the HMM-era name.
_has_hmm_times = _has_aligned_perf_times


def _as_int(*values: Any, default: int | None = None) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _as_float(*values: Any, default: float | None = None) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default
