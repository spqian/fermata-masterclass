from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from masterclass.core.masterclasses import MasterclassStore
from masterclass.core.models import MasterclassManifest
from masterclass.storage.base import ObjectStorage


@dataclass(frozen=True)
class StaffDetectionConfig:
    dark_threshold: int = 170
    staff_row_width_ratio: float = 0.33
    row_group_gap_px: int = 3
    staff_gap_min_px: int = 5
    staff_gap_max_px: int = 18
    staff_span_max_px: int = 70
    fallback_row_width_ratio: float = 0.025
    fallback_row_min_dark_px: int = 20
    fallback_row_group_gap_px: int = 8
    fallback_system_height_min_px: int = 20
    fallback_system_height_max_px: int = 120
    crop_padding_y_px: int = 55
    system_band_width_ratio: float = 0.08
    system_band_gap_px: int = 15
    system_band_merge_gap_px: int = 30
    system_band_min_height_px: int = 70
    system_band_max_height_px: int = 240
    validation_staff_row_width_ratio: float = 0.25


def detect_staff_systems_from_image(
    page_png: bytes,
    *,
    config: StaffDetectionConfig | None = None,
) -> list[dict[str, Any]]:
    """Detect score systems by horizontal dark-pixel projection.

    The primary thresholds mirror the original PoC. A deterministic band pass is
    used to group adjacent staves into score systems for grand-staff pages.
    """

    config = config or StaffDetectionConfig()
    image = Image.open(io.BytesIO(page_png)).convert("RGB")
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    projection = (gray < config.dark_threshold).sum(axis=1)

    systems = _detect_system_bands(projection, width=width, height=height, config=config)
    if not systems:
        staves = _detect_staves(
            projection,
            width=width,
            config=config,
            row_width_ratio=config.staff_row_width_ratio,
        )
        systems = _group_staves_into_systems(staves, height=height)
    if not systems:
        systems = _detect_poc_fallback_systems(projection, width=width, config=config)

    records: list[dict[str, Any]] = []
    for index, (top, bottom) in enumerate(systems, start=1):
        y0 = max(0, int(top) - config.crop_padding_y_px)
        y1 = min(height, int(bottom) + config.crop_padding_y_px)
        if y1 <= y0:
            continue
        crop = image.crop((0, y0, width, y1))
        output = io.BytesIO()
        crop.save(output, format="PNG")
        records.append(
            {
                "system_index": index,
                "bbox": _normalize_bbox(0, y0, width, y1, width=width, height=height),
                "crop_png": output.getvalue(),
            }
        )
    return records


def detect_staff_systems_for_masterclass(
    *,
    storage: ObjectStorage,
    masterclass_store: MasterclassStore,
    manifest: MasterclassManifest,
    config: StaffDetectionConfig | None = None,
) -> list[dict[str, Any]]:
    """Run staff detection for attached rasterized score pages and persist crops."""

    config = config or StaffDetectionConfig()
    page_items = _score_page_artifacts(manifest)
    all_records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for page_number, page_key in page_items:
        records = detect_staff_systems_from_image(storage.read_bytes(page_key), config=config)
        counts[str(page_number)] = len(records)
        for record in records:
            system_index = int(record["system_index"])
            rel = f"reference/score_systems/page-{page_number:03d}-system-{system_index:02d}.png"
            key = masterclass_store.artifact_key(manifest.masterclass, rel)
            storage.write_bytes(key, record["crop_png"], content_type="image/png")
            manifest.artifacts[rel] = key
            all_records.append(
                {
                    "page": page_number,
                    "system_index": system_index,
                    "bbox": record["bbox"],
                    "artifact": rel,
                }
            )

    manifest.metadata["staff_detection_system_counts"] = counts
    manifest.metadata["staff_detection_system_count"] = sum(counts.values())
    masterclass_store.save(manifest)
    return all_records


def detect_barlines_in_system(page_png: bytes, system_bbox: dict) -> list[float]:
    """Return normalized page x-fractions of vertical barlines in a system bbox."""

    from PIL import Image as _Image
    import numpy as _np

    try:
        x = float(system_bbox["x"])
        y = float(system_bbox["y"])
        w = float(system_bbox["w"])
        h = float(system_bbox["h"])
    except (KeyError, TypeError, ValueError):
        return []

    image = _Image.open(io.BytesIO(page_png)).convert("L")
    gray = _np.asarray(image)
    height, width = gray.shape
    x0 = max(0, min(width - 1, int(round(x * width))))
    x1 = max(x0 + 1, min(width, int(round((x + w) * width))))
    y0 = max(0, min(height - 1, int(round(y * height))))
    y1 = max(y0 + 1, min(height, int(round((y + h) * height))))

    strip = gray[y0:y1, x0:x1]
    if strip.size == 0:
        return []

    dark = strip < 150
    # Staff-detected bboxes intentionally include vertical padding; trim that
    # padding for the 80%-of-strip barline projection while still returning page
    # coordinates.
    row_projection = dark.sum(axis=1)
    content_rows = _np.where(row_projection > max(8, int((x1 - x0) * 0.015)))[0]
    if content_rows.size:
        top = max(0, int(content_rows[0]) - 2)
        bottom = min(dark.shape[0], int(content_rows[-1]) + 3)
        dark_for_projection = dark[top:bottom, :]
    else:
        dark_for_projection = dark

    span_height = dark_for_projection.shape[0]
    if span_height <= 0:
        return []
    projection = dark_for_projection.sum(axis=0)
    threshold = max(2, int(span_height * 0.8))
    columns = _np.where(projection >= threshold)[0]
    if columns.size == 0:
        # Engraved piano scores often draw separate vertical strokes through the
        # treble and bass staves rather than a single uninterrupted black line
        # through the whitespace between them. Keep the documented 80% rule as
        # the primary detector, but use a conservative lower fallback for those
        # common barlines.
        threshold = max(12, int(span_height * 0.42))
        columns = _np.where(projection >= threshold)[0]
    if columns.size == 0:
        return []

    groups = _consecutive_groups(columns, max_gap=2)
    positions: list[float] = []
    min_gap_px = max(6, int(width * 0.006))
    for left, right in groups:
        if right - left + 1 > max(20, int(width * 0.025)):
            continue
        center = int(round((left + right) / 2))
        page_x = x0 + center
        if positions and page_x / width - positions[-1] < min_gap_px / width:
            continue
        positions.append(_clamp01(page_x / width))
    return positions


def _detect_system_bands(
    projection: np.ndarray,
    *,
    width: int,
    height: int,
    config: StaffDetectionConfig,
) -> list[tuple[int, int]]:
    threshold = max(config.fallback_row_min_dark_px, int(width * config.system_band_width_ratio))
    rows = np.where(projection > threshold)[0]
    bands = _consecutive_groups(rows, max_gap=config.system_band_gap_px)
    bands = [
        band
        for band in bands
        if config.system_band_min_height_px <= band[1] - band[0] + 1 <= config.system_band_max_height_px
    ]

    merged: list[tuple[int, int]] = []
    for top, bottom in bands:
        if (
            merged
            and top - merged[-1][1] <= config.system_band_merge_gap_px
            and bottom - merged[-1][0] + 1 <= config.system_band_max_height_px
        ):
            merged[-1] = (merged[-1][0], bottom)
        else:
            merged.append((top, bottom))

    validated: list[tuple[int, int]] = []
    min_staff_height = config.staff_gap_min_px * 4
    for top, bottom in merged:
        if bottom - top < min_staff_height:
            continue
        staves = _detect_staves(
            projection,
            width=width,
            config=config,
            row_width_ratio=config.validation_staff_row_width_ratio,
            y_min=top,
            y_max=bottom,
        )
        if staves:
            validated.append((max(0, top), min(height - 1, bottom)))
    return validated


def _detect_staves(
    projection: np.ndarray,
    *,
    width: int,
    config: StaffDetectionConfig,
    row_width_ratio: float,
    y_min: int | None = None,
    y_max: int | None = None,
) -> list[tuple[int, int]]:
    rows = np.where(projection > int(width * row_width_ratio))[0]
    if y_min is not None:
        rows = rows[rows >= y_min]
    if y_max is not None:
        rows = rows[rows <= y_max]
    row_groups = _consecutive_groups(rows, max_gap=config.row_group_gap_px)
    centers = [int(round((top + bottom) / 2)) for top, bottom in row_groups]

    staves: list[tuple[int, int]] = []
    i = 0
    while i <= len(centers) - 5:
        group = centers[i : i + 5]
        gaps = np.diff(group)
        if (
            np.all((gaps >= config.staff_gap_min_px) & (gaps <= config.staff_gap_max_px))
            and group[-1] - group[0] <= config.staff_span_max_px
        ):
            staves.append((group[0], group[-1]))
            i += 5
        else:
            i += 1
    return staves


def _group_staves_into_systems(staves: list[tuple[int, int]], *, height: int) -> list[tuple[int, int]]:
    if not staves:
        return []
    if len(staves) == 1:
        return staves

    gaps = [staves[i + 1][0] - staves[i][1] for i in range(len(staves) - 1)]
    if not gaps:
        return staves
    median_gap = float(np.median(gaps))
    pair_gap_limit = max(45.0, min(height * 0.08, median_gap * 1.35))

    systems: list[tuple[int, int]] = []
    i = 0
    while i < len(staves):
        top, bottom = staves[i]
        if i + 1 < len(staves) and staves[i + 1][0] - bottom <= pair_gap_limit:
            bottom = staves[i + 1][1]
            i += 2
        else:
            i += 1
        systems.append((top, bottom))
    return systems


def _detect_poc_fallback_systems(
    projection: np.ndarray,
    *,
    width: int,
    config: StaffDetectionConfig,
) -> list[tuple[int, int]]:
    threshold = max(config.fallback_row_min_dark_px, int(width * config.fallback_row_width_ratio))
    rows = np.where(projection > threshold)[0]
    systems: list[tuple[int, int]] = []
    for top, bottom in _consecutive_groups(rows, max_gap=config.fallback_row_group_gap_px):
        if not (config.fallback_system_height_min_px <= bottom - top <= config.fallback_system_height_max_px):
            continue
        staves = _detect_staves(
            projection,
            width=width,
            config=config,
            row_width_ratio=config.validation_staff_row_width_ratio,
            y_min=top,
            y_max=bottom,
        )
        if staves:
            systems.append((top, bottom))
    return systems


def _consecutive_groups(rows: np.ndarray, *, max_gap: int) -> list[tuple[int, int]]:
    if rows.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = int(rows[0])
    prev = int(rows[0])
    for row_value in rows[1:]:
        row = int(row_value)
        if row - prev <= max_gap:
            prev = row
            continue
        groups.append((start, prev))
        start = row
        prev = row
    groups.append((start, prev))
    return groups


def _normalize_bbox(x0: int, y0: int, x1: int, y1: int, *, width: int, height: int) -> dict[str, float]:
    return {
        "x": _clamp01(x0 / width),
        "y": _clamp01(y0 / height),
        "w": _clamp01((x1 - x0) / width),
        "h": _clamp01((y1 - y0) / height),
    }


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(max(0.0, min(1.0, value)), 6)


def _score_page_artifacts(manifest: MasterclassManifest) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    pattern = re.compile(r"^reference/score_pages/page-(\d{3})\.png$")
    for rel, key in manifest.artifacts.items():
        match = pattern.match(rel)
        if match:
            pages.append((int(match.group(1)), key))
    pages.sort(key=lambda item: item[0])
    return pages
