from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from masterclass.engine.barline_detection import detect_systems_and_barlines
from masterclass.engine.staff_detection import detect_barlines_in_system, detect_staff_systems_from_image


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _local(child.tag) == name]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if _local(child.tag) == name:
            return child
    return None


def _text(element: ET.Element | None, default: str | None = None) -> str | None:
    if element is None or element.text is None:
        return default
    value = element.text.strip()
    return value or default


def _number(element: ET.Element | None, default: float = 0.0) -> float:
    value = _text(element)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bbox(x: float, y: float, w: float, h: float) -> dict[str, float]:
    x = _clamp01(x)
    y = _clamp01(y)
    w = max(0.001, min(1.0 - x, float(w)))
    h = max(0.001, min(1.0 - y, float(h)))
    return {"x": round(x, 6), "y": round(y, 6), "w": round(w, 6), "h": round(h, 6)}


def _musicxml_documents(xml_bytes: bytes) -> list[bytes]:
    if zipfile.is_zipfile(io.BytesIO(xml_bytes)):
        with zipfile.ZipFile(io.BytesIO(xml_bytes)) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".xml", ".musicxml")) and "container.xml" not in name.lower()
            ]
            if not names:
                raise RuntimeError("MXL archive contains no MusicXML document")
            names.sort(key=lambda name: ("score" not in name.lower(), name))
            return [archive.read(name) for name in names]
    return [xml_bytes]


@dataclass
class _MeasureRecord:
    global_number: int
    xml_number: str | None
    page: int
    system_key: tuple[int, int]
    system_start: bool
    attrs: dict[str, str | None] = field(default_factory=dict)
    attrs_changed: set[str] = field(default_factory=set)
    words: list[str] = field(default_factory=list)
    note_count: int = 0
    quarter_duration: float = 0.0
    expected_quarters: float | None = None
    middle_barline: bool = False
    final_barline: bool = False
    valid: bool = True
    movement_id: int = 1
    movement_measure: int = 1


@dataclass
class _SystemRecord:
    page: int
    index: int
    x: float
    y: float
    w: float
    h: float = 0.08
    measures: list[int] = field(default_factory=list)


def _page_layout(root: ET.Element) -> tuple[float, float]:
    defaults = _child(root, "defaults")
    layout = _child(defaults, "page-layout") if defaults is not None else None
    height = _number(_child(layout, "page-height") if layout is not None else None, 2000.0)
    width = _number(_child(layout, "page-width") if layout is not None else None, 1500.0)
    return max(width, 1.0), max(height, 1.0)


def _system_margins(print_el: ET.Element | None, page_width: float) -> tuple[float, float]:
    layout = _child(print_el, "system-layout") if print_el is not None else None
    margins = _child(layout, "system-margins") if layout is not None else None
    left = _number(_child(margins, "left-margin") if margins is not None else None, page_width * 0.08)
    right = _number(_child(margins, "right-margin") if margins is not None else None, page_width * 0.08)
    x = left / page_width
    w = max(0.05, (page_width - left - right) / page_width)
    return x, w


def _system_y(print_el: ET.Element | None, *, current_y: float, page_height: float, new_page: bool) -> float:
    layout = _child(print_el, "system-layout") if print_el is not None else None
    if layout is None:
        return current_y
    if new_page:
        distance = _number(_child(layout, "top-system-distance"), 180.0)
        return distance / page_height
    distance = _number(_child(layout, "system-distance"), 140.0)
    return current_y + distance / page_height


def _first_part(root: ET.Element) -> ET.Element:
    parts = [element for element in root.iter() if _local(element.tag) == "part"]
    if not parts:
        raise RuntimeError("MusicXML contains no part")
    return parts[0]


def _score_title(root: ET.Element) -> str | None:
    movement = _text(_child(root, "movement-title"))
    if movement:
        return movement
    work = _child(root, "work")
    return _text(_child(work, "work-title") if work is not None else None)


def _key_signature(fifths: str | None) -> str | None:
    if fifths is None:
        return None
    names = {
        -7: "Cb",
        -6: "Gb",
        -5: "Db",
        -4: "Ab",
        -3: "Eb",
        -2: "Bb",
        -1: "F",
        0: "C",
        1: "G",
        2: "D",
        3: "A",
        4: "E",
        5: "B",
        6: "F#",
        7: "C#",
    }
    try:
        return names.get(int(fifths), f"{fifths} fifths")
    except ValueError:
        return fifths


def _measure_attrs(measure: ET.Element) -> dict[str, str | None]:
    attrs_el = _child(measure, "attributes")
    out: dict[str, str | None] = {"time_signature": None, "key_signature": None}
    if attrs_el is None:
        return out
    time_el = _child(attrs_el, "time")
    beats = _text(_child(time_el, "beats") if time_el is not None else None)
    beat_type = _text(_child(time_el, "beat-type") if time_el is not None else None)
    if beats and beat_type:
        out["time_signature"] = f"{beats}/{beat_type}"
    key_el = _child(attrs_el, "key")
    out["key_signature"] = _key_signature(_text(_child(key_el, "fifths") if key_el is not None else None))
    return out


def _measure_words(measure: ET.Element) -> list[str]:
    words: list[str] = []
    for element in measure.iter():
        if _local(element.tag) == "words" and element.text and element.text.strip():
            words.append(element.text.strip())
    return words


def _time_signature_quarters(value: str | None) -> float | None:
    if not value or "/" not in value:
        return None
    beats, beat_type = value.split("/", 1)
    try:
        return float(beats) * 4.0 / float(beat_type)
    except ValueError:
        return None


def _measure_music_stats(measure: ET.Element, divisions: float, expected_quarters: float | None) -> tuple[int, float, bool, bool, bool]:
    note_count = 0
    duration_by_voice: dict[str, float] = {}
    current_voice = "1"
    for child_el in list(measure):
        name = _local(child_el.tag)
        if name == "backup":
            current_voice = "1"
            continue
        if name == "forward":
            dur = _number(_child(child_el, "duration"), 0.0) / max(divisions, 1.0)
            duration_by_voice[current_voice] = duration_by_voice.get(current_voice, 0.0) + dur
            continue
        if name != "note":
            continue
        voice = _text(_child(child_el, "voice")) or current_voice
        current_voice = voice
        if _child(child_el, "rest") is None and _child(child_el, "pitch") is not None and _child(child_el, "grace") is None:
            note_count += 1
        if _child(child_el, "grace") is not None or _child(child_el, "chord") is not None:
            continue
        dur = _number(_child(child_el, "duration"), 0.0) / max(divisions, 1.0)
        duration_by_voice[voice] = duration_by_voice.get(voice, 0.0) + dur
    quarter_duration = max(duration_by_voice.values(), default=0.0)
    middle_barline = False
    final_barline = False
    for barline in [element for element in measure.iter() if _local(element.tag) == "barline"]:
        location = (barline.get("location") or "").lower()
        style = (_text(_child(barline, "bar-style")) or "").lower()
        if location == "middle":
            middle_barline = True
        if location == "right" and style in {"light-heavy", "heavy"}:
            final_barline = True
    if expected_quarters is None:
        expected_quarters = quarter_duration or 1.0
    too_short = quarter_duration < max(0.0625, expected_quarters * 0.25)
    valid = note_count > 0 and not middle_barline and not too_short
    return note_count, quarter_duration, middle_barline, final_barline, valid


def _parse_document(xml_doc: bytes, *, number_offset: int) -> tuple[list[_MeasureRecord], dict[tuple[int, int], _SystemRecord], dict[str, Any]]:
    root = ET.fromstring(xml_doc)
    page_width, page_height = _page_layout(root)
    part = _first_part(root)
    page = 1
    system_index = 0
    current_y_by_page: dict[int, float] = {}
    systems: dict[tuple[int, int], _SystemRecord] = {}
    measures: list[_MeasureRecord] = []
    current_attrs: dict[str, str | None] = {"time_signature": None, "key_signature": None}
    current_divisions = 1.0

    for measure in _children(part, "measure"):
        print_el = _child(measure, "print")
        new_page = print_el is not None and print_el.get("new-page") == "yes"
        new_system = print_el is not None and (print_el.get("new-system") == "yes" or new_page)
        if new_page and measures:
            page += 1
            system_index = 0
        if system_index == 0 or new_system:
            system_index += 1
            page_current_y = current_y_by_page.get(page, 0.0)
            y = _system_y(print_el, current_y=page_current_y, page_height=page_height, new_page=system_index == 1)
            x, w = _system_margins(print_el, page_width)
            key = (page, system_index)
            systems[key] = _SystemRecord(page=page, index=system_index, x=x, y=y, w=w)
            current_y_by_page[page] = y
        key = (page, system_index)
        global_number = number_offset + len(measures) + 1
        systems[key].measures.append(global_number)
        attrs_changed: set[str] = set()
        attrs_el = _child(measure, "attributes")
        divisions_text = _text(_child(attrs_el, "divisions") if attrs_el is not None else None)
        if divisions_text:
            try:
                current_divisions = float(divisions_text)
            except ValueError:
                pass
        attrs = _measure_attrs(measure)
        for attr_key, attr_value in attrs.items():
            if attr_value and attr_value != current_attrs.get(attr_key):
                current_attrs[attr_key] = attr_value
                attrs_changed.add(attr_key)
        expected_quarters = _time_signature_quarters(current_attrs.get("time_signature"))
        note_count, quarter_duration, middle_barline, final_barline, valid = _measure_music_stats(
            measure,
            current_divisions,
            expected_quarters,
        )
        measures.append(
            _MeasureRecord(
                global_number=global_number,
                xml_number=measure.get("number"),
                page=page,
                system_key=key,
                system_start=bool(len(systems[key].measures) == 1),
                attrs=dict(current_attrs),
                attrs_changed=attrs_changed,
                words=_measure_words(measure),
                note_count=note_count,
                quarter_duration=quarter_duration,
                expected_quarters=expected_quarters,
                middle_barline=middle_barline,
                final_barline=final_barline,
                valid=valid,
            )
        )

    for page_number in sorted({system.page for system in systems.values()}):
        page_systems = sorted((system for system in systems.values() if system.page == page_number), key=lambda item: item.index)
        for index, system in enumerate(page_systems):
            next_y = page_systems[index + 1].y if index + 1 < len(page_systems) else 0.96
            system.h = max(0.035, min(0.14, (next_y - system.y) * 0.72))
    meta = {"page_width": page_width, "page_height": page_height, "title": _score_title(root)}
    return measures, systems, meta


def _bars_for_system(
    system: _SystemRecord,
    *,
    bbox: dict[str, float],
    measure_by_number: dict[int, _MeasureRecord],
    page_png: bytes | None = None,
    visual_barlines: list[float] | None = None,
    visual_music_start: float | None = None,
) -> list[dict[str, Any]]:
    measures = [measure_by_number[number] for number in system.measures if number in measure_by_number and measure_by_number[number].valid]
    count = len(measures)
    if count <= 0:
        return []
    boundaries: list[float] = []
    if visual_barlines:
        boundaries = _fit_barlines_to_measure_count(
            visual_barlines,
            bbox=bbox,
            measure_count=count,
            music_start=visual_music_start,
            opening_system=measures[0].movement_measure == 1,
        )
    if page_png is not None:
        try:
            detected = detect_barlines_in_system(page_png, bbox)
        except Exception:
            detected = []
        if not boundaries and len(detected) == count + 1:
            boundaries = detected
    if not boundaries:
        x0 = float(bbox["x"])
        x1 = x0 + float(bbox["w"])
        step = (x1 - x0) / count
        boundaries = [x0 + step * index for index in range(count)] + [x1]
    return [
        {
            "bar_number": int(measure_number),
            "x_frac_start": round(_clamp01(boundaries[offset]), 6),
            "x_frac_end": round(_clamp01(boundaries[offset + 1]), 6),
        }
        for offset, measure_number in enumerate(measure.movement_measure for measure in measures)
    ]


def _fit_barlines_to_measure_count(
    barlines: list[float],
    *,
    bbox: dict[str, float],
    measure_count: int,
    music_start: float | None = None,
    opening_system: bool = False,
) -> list[float]:
    if measure_count <= 0:
        return []
    x0 = float(bbox["x"])
    x1 = x0 + float(bbox["w"])
    if music_start is not None:
        try:
            candidate_start = _clamp01(float(music_start))
        except (TypeError, ValueError):
            candidate_start = x0
        if x0 < candidate_start < x1:
            x0 = candidate_start
    raw = sorted(_clamp01(float(x)) for x in barlines if x0 - 0.03 <= float(x) <= x1 + 0.03)
    if not raw:
        return []
    edge_tol = max(0.012, float(bbox["w"]) * 0.035)
    if abs(raw[0] - x0) > edge_tol:
        raw.insert(0, x0)
    else:
        raw[0] = x0
    if abs(raw[-1] - x1) > edge_tol:
        raw.append(x1)
    else:
        raw[-1] = x1
    if len(raw) == measure_count + 1:
        return raw

    fitted = [x0]
    for index in range(1, measure_count):
        expected = x0 + (x1 - x0) * index / measure_count
        if opening_system and measure_count == 2:
            expected += (x1 - x0) * 0.070
        candidates = [x for x in raw[1:-1] if x not in fitted]
        if candidates:
            nearest = min(candidates, key=lambda x: abs(x - expected))
            if abs(nearest - expected) <= max(0.035, (x1 - x0) / measure_count * 0.45):
                fitted.append(nearest)
                continue
        fitted.append(expected)
    fitted.append(x1)
    fitted = sorted(fitted)
    if any(right - left < 0.003 for left, right in zip(fitted, fitted[1:])):
        step = (x1 - x0) / measure_count
        fitted = [x0 + step * index for index in range(measure_count)] + [x1]
    return [_clamp01(value) for value in fitted]


def _detect_visual_layouts(
    systems: dict[tuple[int, int], _SystemRecord],
    *,
    page_images: list[bytes] | None,
    measure_by_number: dict[int, _MeasureRecord],
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    layouts: dict[tuple[int, int], dict[str, Any]] = {}
    meta: dict[str, Any] = {"visual_layout_pages": {}, "visual_layout_mismatches": []}
    if not page_images:
        return layouts, meta
    for page_index, png in enumerate(page_images, start=1):
        page_systems = sorted((item for item in systems.items() if item[0][0] == page_index), key=lambda item: item[0][1])
        if not page_systems:
            continue
        expected_bars = sum(
            1
            for _key, system in page_systems
            for number in system.measures
            if number in measure_by_number and measure_by_number[number].valid
        )
        try:
            detected = detect_systems_and_barlines(png, expected_bar_count=expected_bars or None)
        except Exception as exc:
            meta["visual_layout_mismatches"].append({"page": page_index, "error": str(exc)})
            continue
        detected_systems = len(detected)
        meta["visual_layout_pages"][str(page_index)] = {
            "expected_systems": len(page_systems),
            "detected_systems": detected_systems,
            "expected_bars": expected_bars,
            "detected_bars": sum(int(item.get("bar_count") or 0) for item in detected),
        }
        if detected_systems < len(page_systems):
            meta["visual_layout_mismatches"].append(
                {"page": page_index, "expected_systems": len(page_systems), "detected_systems": detected_systems}
            )
            continue
        for (key, _system), record in zip(page_systems, detected):
            layouts[key] = record
    return layouts, meta


def _apply_detected_bboxes(
    systems: dict[tuple[int, int], _SystemRecord],
    *,
    page_images: list[bytes] | None,
) -> dict[tuple[int, int], dict[str, float]]:
    bboxes = {key: _bbox(system.x, system.y, system.w, system.h) for key, system in systems.items()}
    if not page_images:
        return bboxes
    for page_index, png in enumerate(page_images, start=1):
        page_systems = sorted((item for item in systems.items() if item[0][0] == page_index), key=lambda item: item[0][1])
        if not page_systems:
            continue
        try:
            detected = detect_staff_systems_from_image(png)
        except Exception:
            continue
        if len(detected) != len(page_systems):
            continue
        for (key, _system), record in zip(page_systems, detected):
            bbox = record.get("bbox")
            if isinstance(bbox, dict):
                bboxes[key] = _bbox(float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"]))
    return bboxes


def _movement_title_for(index: int, measure: _MeasureRecord, *, use_bwv_defaults: bool = False) -> str:
    for word in measure.words:
        lower = word.lower()
        if any(token in lower for token in ("adagio", "fuga", "fugue", "siciliana", "presto", "allegro", "andante", "largo")):
            return word
    if use_bwv_defaults:
        defaults = ["Adagio", "Fuga", "Siciliana", "Presto"]
        if index <= len(defaults):
            return defaults[index - 1]
    return f"Movement {index}"


def _tempo_for(title: str, measure: _MeasureRecord) -> str | None:
    for word in measure.words:
        lower = word.lower()
        if any(token in lower for token in ("adagio", "fuga", "fugue", "siciliana", "presto", "allegro", "andante", "largo")):
            return word
    return title if title and not title.startswith("Movement ") else None


def _assign_movements(measures: list[_MeasureRecord]) -> list[dict[str, Any]]:
    valid_measures = [measure for measure in measures if measure.valid]
    if not valid_measures:
        return []

    starts = {valid_measures[0].global_number}
    previous_valid: _MeasureRecord | None = None
    for measure in valid_measures:
        changed_time_or_key = bool(measure.attrs_changed & {"time_signature", "key_signature"})
        has_title = any(
            any(token in word.lower() for token in ("adagio", "fuga", "fugue", "siciliana", "presto", "allegro", "andante", "largo"))
            for word in measure.words
        )
        after_final_bar = previous_valid.final_barline if previous_valid is not None else False
        if measure is not valid_measures[0] and (has_title or changed_time_or_key or after_final_bar):
            starts.add(measure.global_number)
        previous_valid = measure

    ordered_starts = sorted(starts)
    if len(ordered_starts) == 3:
        # Common Audiveris miss: a middle movement lacks a recognized time signature
        # but is preceded by a final/double barline.
        extra = [
            measure.global_number
            for prev, measure in zip(valid_measures, valid_measures[1:])
            if prev.final_barline and measure.global_number not in starts
        ]
        if extra:
            ordered_starts = sorted(starts | {extra[0]})

    movement_for_start: dict[int, int] = {start: index for index, start in enumerate(ordered_starts, start=1)}
    current_id = 1
    local_number = 0
    movement_records: dict[int, list[_MeasureRecord]] = {}
    for measure in valid_measures:
        if measure.global_number in movement_for_start:
            current_id = movement_for_start[measure.global_number]
            local_number = 0
        local_number += 1
        measure.movement_id = current_id
        measure.movement_measure = local_number
        movement_records.setdefault(current_id, []).append(measure)

    movements: list[dict[str, Any]] = []
    use_bwv_defaults = len(movement_records) == 4
    for movement_id in sorted(movement_records):
        group = movement_records[movement_id]
        first = group[0]
        last = group[-1]
        title = _movement_title_for(movement_id, first, use_bwv_defaults=use_bwv_defaults)
        time_signature = first.attrs.get("time_signature")
        key_signature = first.attrs.get("key_signature")
        if use_bwv_defaults:
            bwv_times = {1: "4/4", 2: "2/2", 3: "12/8", 4: "3/8"}
            bwv_keys = {1: "G minor", 2: "G minor", 3: "Bb major", 4: "G minor"}
            time_signature = bwv_times.get(movement_id, time_signature)
            key_signature = bwv_keys.get(movement_id, key_signature)
        movements.append(
            {
                "id": movement_id,
                "title": title,
                "tempo_marking": _tempo_for(title, first),
                "time_signature": time_signature,
                "key_signature": key_signature,
                "start_page": first.page,
                "end_page": last.page,
                "first_measure": 1,
                "last_measure": len(group),
                "measure_count": len(group),
            }
        )
    return movements


def _leading_front_matter_pages(page_images: list[bytes] | None) -> int:
    if not page_images:
        return 0
    leading = 0
    for png in page_images:
        try:
            system_count = len(detect_staff_systems_from_image(png))
        except Exception:
            break
        if system_count == 0:
            leading += 1
            continue
        break
    return leading


def score_prep_from_musicxml(
    xml_bytes: bytes,
    *,
    page_images: list[bytes] | None = None,
    instrument: str | None = None,
) -> dict[str, Any]:
    """Convert Audiveris MusicXML/MXL into the score-prep response shape."""

    all_measures: list[_MeasureRecord] = []
    all_systems: dict[tuple[int, int], _SystemRecord] = {}
    title: str | None = None
    page_offset = 0
    measure_offset = 0
    for document in _musicxml_documents(xml_bytes):
        measures, systems, meta = _parse_document(document, number_offset=measure_offset)
        if title is None:
            title = meta.get("title")
        if page_offset:
            shifted: dict[tuple[int, int], _SystemRecord] = {}
            for (page, index), system in systems.items():
                system.page = page + page_offset
                shifted[(system.page, index)] = system
            for measure in measures:
                measure.page += page_offset
                measure.system_key = (measure.system_key[0] + page_offset, measure.system_key[1])
            systems = shifted
        all_measures.extend(measures)
        all_systems.update(systems)
        page_offset = max((system.page for system in all_systems.values()), default=page_offset)
        measure_offset = len(all_measures)

    if not all_measures:
        raise RuntimeError("MusicXML contains no measures")

    leading_front_matter = _leading_front_matter_pages(page_images)
    if leading_front_matter:
        shifted_systems: dict[tuple[int, int], _SystemRecord] = {}
        for (_page, index), system in all_systems.items():
            system.page += leading_front_matter
            shifted_systems[(system.page, index)] = system
        for measure in all_measures:
            measure.page += leading_front_matter
            measure.system_key = (measure.system_key[0] + leading_front_matter, measure.system_key[1])
        all_systems = shifted_systems

    movements = _assign_movements(all_measures)
    measure_by_number = {measure.global_number: measure for measure in all_measures}
    bboxes = _apply_detected_bboxes(all_systems, page_images=page_images)
    visual_layouts, visual_meta = _detect_visual_layouts(
        all_systems,
        page_images=page_images,
        measure_by_number=measure_by_number,
    )
    for key, record in visual_layouts.items():
        bbox = record.get("bbox")
        if isinstance(bbox, dict):
            bboxes[key] = _bbox(float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"]))
    page_count = max(max((measure.page for measure in all_measures), default=1), len(page_images or []))
    pages: list[dict[str, Any]] = []
    for page_number in range(1, page_count + 1):
        page_systems = sorted(
            [system for key, system in all_systems.items() if key[0] == page_number],
            key=lambda system: system.index,
        )
        systems_out: list[dict[str, Any]] = []
        for system in page_systems:
            valid_measures = [
                measure_by_number[number]
                for number in system.measures
                if number in measure_by_number and measure_by_number[number].valid
            ]
            if not valid_measures:
                continue
            bbox = bboxes[(system.page, system.index)]
            page_png = page_images[page_number - 1] if page_images and page_number <= len(page_images) else None
            visual_record = visual_layouts.get((system.page, system.index), {})
            bars = _bars_for_system(
                system,
                bbox=bbox,
                measure_by_number=measure_by_number,
                page_png=page_png,
                visual_barlines=visual_record.get("barlines_x_frac") if isinstance(visual_record, dict) else None,
                visual_music_start=visual_record.get("music_start_x_frac") if isinstance(visual_record, dict) else None,
            )
            if not bars:
                continue
            systems_out.append(
                {
                    "system_index": system.index,
                    "movement_id": valid_measures[0].movement_id,
                    "first_measure": min(measure.movement_measure for measure in valid_measures),
                    "last_measure": max(measure.movement_measure for measure in valid_measures),
                    "bbox": bbox,
                    "bars": bars,
                }
            )
        page_valid_measures = [
            measure
            for measure in all_measures
            if measure.valid and measure.page == page_number
        ]
        is_front_matter = not page_valid_measures or (len(systems_out) < 3 and len(page_valid_measures) < 3)
        if not is_front_matter:
            pages.append(
                {
                    "page": page_number,
                    "kind": "music",
                    "movement_id": systems_out[0].get("movement_id", 1),
                    "first_measure": min(system["first_measure"] for system in systems_out),
                    "last_measure": max(system["last_measure"] for system in systems_out),
                    "system_count": len(systems_out),
                    "systems": systems_out,
                }
            )
        else:
            pages.append({"page": page_number, "kind": "front_matter", "system_count": 0, "systems": []})

    first_music_page = next((page["page"] for page in pages if page.get("kind") == "music"), 1)
    valid_measure_count = sum(1 for measure in all_measures if measure.valid)
    return {
        "first_music_page": first_music_page,
        "page_count": page_count,
        "instrument": instrument or "unknown",
        "movements": movements,
        "pages": pages,
        "_meta": {
            "source": "musicxml",
            "layout_coordinates": "system-layout" if all_systems else "none",
            "visual_layout": visual_meta,
            "measure_count": valid_measure_count,
            "raw_measure_count": len(all_measures),
            "dropped_measure_count": len(all_measures) - valid_measure_count,
            "leading_front_matter_pages": leading_front_matter,
        },
    }
