from __future__ import annotations

import bisect
import io
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

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
    hmm_score_time_round_digits: int = 3
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

    midi_key = _find_midi_key(storage, masterclass, manifest)
    if not midi_key:
        raise ValueError("masterclass/session is missing reference MIDI")
    midi_bytes = storage.read_bytes(midi_key)

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
        midi_bytes=midi_bytes,
        bars=bars,
        played_lo=played_lo,
        played_hi=played_hi,
        score_key=score_key,
        hmm_notes=_load_hmm_notes(storage, manifest),
        config=config,
    )
    total_measures = total_measures or _midi_measure_count(midi_bytes)

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
        "alignment_source": "hmm_viterbi" if _has_hmm_times(notes) else "midi_only",
        "systems": systems,
        "bars": bars,
        "notes": notes,
        "notes_help": [
            "Each note has stable note_id, MIDI measure/beat, score_time, perf_time, names, MIDI pitches, and score x fractions.",
            "perf_time is read from analysis/hmm_aligned_notes.json when present; otherwise it is null.",
            "Bar image/bbox entries are derived from masterclass reference/score_prep.json systems.",
        ],
        "_meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "score_prep_key": score_prep_key,
            "midi_key": midi_key,
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
    midi_bytes: bytes,
    bars: list[dict[str, Any]],
    played_lo: int,
    played_hi: int,
    score_key: str | None,
    hmm_notes: list[dict[str, Any]],
    config: ScoreMapConfig,
) -> list[dict[str, Any]]:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    downbeats = list(map(float, midi.get_downbeats()))
    if not downbeats:
        raise RuntimeError("MIDI has no detectable measure structure")
    midi_notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            midi_notes.append(
                {
                    "start": float(note.start),
                    "end": float(note.end),
                    "pitch": int(note.pitch),
                    "name": pretty_midi.note_number_to_name(int(note.pitch)),
                    "velocity": int(note.velocity),
                }
            )
    midi_notes.sort(key=lambda row: (float(row["start"]), -int(row["pitch"])))

    events_by_score_time: dict[float, list[dict[str, Any]]] = {}
    for note in midi_notes:
        key = round(float(note["start"]), config.chord_time_round_digits)
        events_by_score_time.setdefault(key, []).append(note)

    hmm_lookup, hmm_conf, hmm_dwell = _hmm_lookups(hmm_notes, config=config)
    bar_by_measure = {int(bar["midi_measure"]): bar for bar in bars}
    events_by_measure: dict[int, list[float]] = {}
    for score_time in sorted(events_by_score_time):
        measure = max(1, bisect.bisect_right(downbeats, score_time))
        if played_lo <= measure <= played_hi:
            events_by_measure.setdefault(measure, []).append(score_time)

    out: list[dict[str, Any]] = []
    for measure in sorted(events_by_measure):
        bar = bar_by_measure.get(measure)
        if not bar:
            continue
        event_times = events_by_measure[measure]
        n_events = max(1, len(event_times))
        x0, x1 = bar.get("highlight_x_frac", [0.0, 1.0])
        bar_width = float(x1) - float(x0)
        db_start = downbeats[measure - 1] if measure - 1 < len(downbeats) else event_times[0]
        db_end = downbeats[measure] if measure < len(downbeats) else midi.get_end_time()
        bar_dur = max(1e-6, float(db_end) - float(db_start))
        for index, score_time in enumerate(event_times):
            notes_at = events_by_score_time[score_time]
            names = spell_pitch_names([str(note["name"]) for note in notes_at], score_key)
            pitches = [int(note["pitch"]) for note in notes_at]
            beat = 1.0 + float(config.beats_per_bar) * (float(score_time) - float(db_start)) / bar_dur
            nx0 = float(x0) + (index / n_events) * bar_width
            nx1 = float(x0) + ((index + 1) / n_events) * bar_width
            lookup_key = round(float(score_time), config.hmm_score_time_round_digits)
            perf_time = hmm_lookup.get(lookup_key)
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
                    "hmm_confidence": hmm_conf.get(lookup_key, "none"),
                    "hmm_dwell_sec": hmm_dwell.get(lookup_key, 0.0),
                    "interpolated": False,
                    "is_bar_anchor": index == 0,
                    "confidence": "high" if index == 0 else hmm_conf.get(lookup_key, "none"),
                }
            )
    out.sort(key=lambda row: (int(row["midi_measure"]), float(row["score_time"]), str(row["note_id"])))
    return out


def _load_hmm_notes(storage: ObjectStorage, manifest: SessionManifest) -> list[dict[str, Any]]:
    key = manifest.artifacts.get("analysis/hmm_aligned_notes.json")
    if key and storage.exists(key):
        data = storage.read_json(key)
        notes = data.get("notes") if isinstance(data, dict) else data
        return [note for note in notes or [] if isinstance(note, dict)]
    key = manifest.artifacts.get("analysis/hmm_alignment.json")
    if key and storage.exists(key):
        data = storage.read_json(key)
        return [note for note in data.get("note_alignments") or [] if isinstance(note, dict)]
    return []


def _hmm_lookups(
    hmm_notes: list[dict[str, Any]], *, config: ScoreMapConfig
) -> tuple[dict[float, float], dict[float, str], dict[float, float]]:
    perf: dict[float, float] = {}
    conf: dict[float, str] = {}
    dwell: dict[float, float] = {}
    for note in hmm_notes:
        score_time = _as_float(note.get("score_time_in_movement"), note.get("score_time"), default=None)
        perf_time = _as_float(note.get("performed_time_sec"), note.get("perf_time"), default=None)
        if score_time is None or perf_time is None:
            continue
        key = round(score_time, config.hmm_score_time_round_digits)
        perf[key] = perf_time
        conf[key] = str(note.get("confidence") or "low")
        dwell[key] = float(_as_float(note.get("dwell_sec"), default=0.0) or 0.0)
    return perf, conf, dwell


def _midi_measure_count(midi_bytes: bytes) -> int:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    downbeats = midi.get_downbeats()
    return max(1, len(downbeats))


def _has_hmm_times(notes: list[dict[str, Any]]) -> bool:
    return any(note.get("perf_time") is not None for note in notes)


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
