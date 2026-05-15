from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class _SystemBand:
    top: int
    bottom: int
    staff_top: int
    staff_bottom: int
    x0: int
    x1: int
    staff_lines: list[int]


def detect_systems_and_barlines(
    page_image_bytes: bytes,
    *,
    expected_bar_count: int | None = None,
) -> list[dict[str, Any]]:
    """Detect score systems and barline boundaries from a raster page.

    Returns [{"system_index": 1, "bbox": {"x", "y", "w", "h"},
    "barlines_x_frac": [...], "bar_count": N}, ...]. Coordinates are normalized
    to page fractions.
    """

    gray = _load_gray(page_image_bytes)
    systems = _detect_system_bands(gray)
    if not systems:
        return []
    # The projection detector is the production path: it is simple, fast, and
    # produced the best cross-page soft barline accuracy in the prototype.
    results = _detect_with_vertical_projection(gray, systems)
    for index, item in enumerate(results, start=1):
        item["system_index"] = index
    if expected_bar_count and expected_bar_count > 0:
        _mark_count_mismatch(results, expected_bar_count)
    return results


def detect_with_algorithm(page_image_bytes: bytes, algorithm: str) -> list[dict[str, Any]]:
    """Prototype hook for algorithm comparison: A, B, C, or D."""

    gray = _load_gray(page_image_bytes)
    systems = _detect_system_bands(gray)
    if not systems:
        return []
    algo = algorithm.upper()
    if algo == "A":
        results = _detect_with_vertical_projection(gray, systems)
    elif algo == "B":
        results = _detect_with_hough(gray, systems)
    elif algo == "C":
        results = _detect_with_components(gray, systems)
    elif algo == "D":
        results = _detect_with_lsd(gray, systems)
    else:
        raise ValueError(f"unknown barline algorithm: {algorithm}")
    for index, item in enumerate(results, start=1):
        item["system_index"] = index
    return results


def _load_gray(page_image_bytes: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(page_image_bytes)).convert("L"))


def _detect_system_bands(gray: np.ndarray, *, percentile: float = 90.0) -> list[_SystemBand]:
    height, width = gray.shape
    dark = gray < 185
    projection = dark.sum(axis=1).astype(np.float32)
    smooth = gaussian_filter1d(projection, sigma=1.2)
    threshold = max(width * 0.22, float(np.percentile(smooth, percentile)))
    rows = np.where(smooth >= threshold)[0]
    line_groups = _groups(rows, max_gap=3)
    staff_lines = [int(round((a + b) / 2)) for a, b in line_groups if b - a <= 8]
    staves: list[tuple[int, int, list[int]]] = []
    i = 0
    while i <= len(staff_lines) - 5:
        lines = staff_lines[i : i + 5]
        gaps = np.diff(lines)
        if np.all((gaps >= 4) & (gaps <= 26)) and lines[-1] - lines[0] <= 95:
            staves.append((lines[0], lines[-1], lines))
            i += 5
        else:
            i += 1

    if not staves:
        # Fallback to broad horizontal content bands when staff lines are broken.
        threshold = max(18, int(width * 0.035))
        bands = _groups(np.where(projection > threshold)[0], max_gap=14)
        staves = [(a, b, []) for a, b in bands if 18 <= b - a <= 230]

    systems_raw: list[tuple[int, int, list[int]]] = []
    idx = 0
    while idx < len(staves):
        top, bottom, lines = staves[idx]
        merged_lines = list(lines)
        if idx + 1 < len(staves):
            next_top, next_bottom, next_lines = staves[idx + 1]
            gap = next_top - bottom
            # Grand staff: two staves are close relative to inter-system gaps.
            if gap <= min(85, max(55, int(height * 0.045))):
                bottom = next_bottom
                merged_lines.extend(next_lines)
                idx += 2
            else:
                idx += 1
        else:
            idx += 1
        systems_raw.append((top, bottom, merged_lines))

    if not systems_raw:
        return []

    systems: list[_SystemBand] = []
    for n, (staff_top, staff_bottom, lines) in enumerate(systems_raw):
        prev_bottom = systems_raw[n - 1][1] if n else 0
        next_top = systems_raw[n + 1][0] if n + 1 < len(systems_raw) else height
        prev_limit = int(round((prev_bottom + staff_top) / 2)) if n else 0
        next_limit = int(round((staff_bottom + next_top) / 2)) if n + 1 < len(systems_raw) else height
        staff_span = staff_bottom - staff_top + 1
        desired_top = staff_top - max(28, int(staff_span * 0.55))
        desired_bottom = staff_bottom + max(32, int(staff_span * 0.75))
        top = max(0, prev_limit, desired_top)
        bottom = min(height, next_limit, desired_bottom)
        if bottom <= top:
            top = max(0, prev_limit)
            bottom = min(height, next_limit)
        x0, x1 = _estimate_system_x(gray, max(0, staff_top - 3), min(height, staff_bottom + 4))
        systems.append(_SystemBand(top, bottom, staff_top, staff_bottom, x0, x1, lines))

    systems = _dedupe_systems(systems)
    if percentile > 80 and len(systems) < 5 and height > 1200:
        retry = _detect_system_bands(gray, percentile=80.0)
        if len(retry) > len(systems):
            return retry
    return systems


def _estimate_system_x(gray: np.ndarray, y0: int, y1: int) -> tuple[int, int]:
    height, width = gray.shape
    crop = gray[y0:y1, :]
    if crop.size == 0:
        return int(width * 0.04), int(width * 0.96)
    projection = (crop < 190).sum(axis=0)
    columns = np.where(projection > max(1, int((y1 - y0) * 0.10)))[0]
    if columns.size == 0:
        return int(width * 0.04), int(width * 0.96)
    x0 = max(0, int(columns[0]) - 8)
    x1 = min(width, int(columns[-1]) + 9)
    if x1 - x0 < width * 0.45:
        return int(width * 0.04), int(width * 0.96)
    return x0, x1


def _detect_with_vertical_projection(gray: np.ndarray, systems: list[_SystemBand]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    dark = gray < 150
    for system in systems:
        y0, y1 = max(0, system.staff_top - 3), min(gray.shape[0], system.staff_bottom + 4)
        x0, x1 = system.x0, system.x1
        crop = dark[y0:y1, x0:x1]
        projection = gaussian_filter1d(crop.sum(axis=0).astype(np.float32), sigma=1.0)
        thresh = max(crop.shape[0] * 0.45, float(np.percentile(projection, 97.5)))
        xs = np.where(projection >= thresh)[0]
        candidates = [(x0 + (a + b) // 2, float(projection[a : b + 1].max())) for a, b in _groups(xs, max_gap=3)]
        candidates = _filter_staff_crossing_candidates(gray, system, candidates)
        out.append(_system_record(gray, system, _clean_candidates(candidates, x0, x1, gray.shape[1])))
    return out


def _detect_with_hough(gray: np.ndarray, systems: list[_SystemBand]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for system in systems:
        y0, y1 = max(0, system.staff_top - 8), min(gray.shape[0], system.staff_bottom + 9)
        x0, x1 = system.x0, system.x1
        strip = gray[y0:y1, x0:x1]
        binary = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        min_len = max(18, int((system.staff_bottom - system.staff_top) * 0.55))
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=18, minLineLength=min_len, maxLineGap=6)
        candidates: list[tuple[int, float]] = []
        if lines is not None:
            for line in lines[:, 0, :]:
                x_a, y_a, x_b, y_b = [int(v) for v in line]
                if abs(x_a - x_b) > 3:
                    continue
                length = abs(y_b - y_a)
                if length < min_len:
                    continue
                candidates.append((x0 + int(round((x_a + x_b) / 2)), float(length)))
        candidates = _filter_staff_crossing_candidates(gray, system, candidates)
        if len(candidates) < 2:
            # Hough can miss very thin print; projection gives a deterministic fallback.
            rec = _detect_with_vertical_projection(gray, [system])[0]
            out.append(rec)
        else:
            out.append(_system_record(gray, system, _clean_candidates(candidates, x0, x1, gray.shape[1])))
    return out


def _detect_with_components(gray: np.ndarray, systems: list[_SystemBand]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for system in systems:
        y0, y1 = max(0, system.staff_top - 8), min(gray.shape[0], system.staff_bottom + 9)
        x0, x1 = system.x0, system.x1
        strip = gray[y0:y1, x0:x1]
        binary = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, max(9, int(strip.shape[0] * 0.35))))
        merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _centroids = cv2.connectedComponentsWithStats(merged)
        candidates: list[tuple[int, float]] = []
        for label in range(1, n):
            x, _y, w, h, area = [int(v) for v in stats[label]]
            if h >= strip.shape[0] * 0.42 and w <= max(14, strip.shape[1] * 0.025) and area > h:
                candidates.append((x0 + x + w // 2, float(h)))
        candidates = _filter_staff_crossing_candidates(gray, system, candidates)
        out.append(_system_record(gray, system, _clean_candidates(candidates, x0, x1, gray.shape[1])))
    return out


def _detect_with_lsd(gray: np.ndarray, systems: list[_SystemBand]) -> list[dict[str, Any]]:
    detector = cv2.createLineSegmentDetector(0)
    out: list[dict[str, Any]] = []
    for system in systems:
        y0, y1 = max(0, system.staff_top - 8), min(gray.shape[0], system.staff_bottom + 9)
        x0, x1 = system.x0, system.x1
        strip = gray[y0:y1, x0:x1]
        lines = detector.detect(strip)[0]
        candidates: list[tuple[int, float]] = []
        min_len = max(18, int((system.staff_bottom - system.staff_top) * 0.50))
        if lines is not None:
            for line in lines[:, 0, :]:
                xa, ya, xb, yb = [float(v) for v in line]
                if abs(xa - xb) > 3.5:
                    continue
                length = abs(yb - ya)
                if length >= min_len:
                    candidates.append((x0 + int(round((xa + xb) / 2)), length))
        candidates = _filter_staff_crossing_candidates(gray, system, candidates)
        out.append(_system_record(gray, system, _clean_candidates(candidates, x0, x1, gray.shape[1])))
    return out


def _system_record(gray: np.ndarray, system: _SystemBand, barlines_px: list[int]) -> dict[str, Any]:
    height, width = gray.shape
    x0, x1 = system.x0, system.x1
    music_start_x = _music_start_x(gray, system)
    boundaries = [_refine_barline_right_edge(gray, system, x) for x in barlines_px if music_start_x <= x <= x1 + 4]
    boundaries = _ensure_edge_boundaries(boundaries, music_start_x, x1, width)
    bbox = {
        "x": _clamp(system.x0 / width),
        "y": _clamp(system.top / height),
        "w": _clamp((system.x1 - system.x0) / width),
        "h": _clamp((system.bottom - system.top) / height),
    }
    return {
        "system_index": 0,
        "bbox": bbox,
        "music_start_x_frac": _clamp(music_start_x / width),
        "barlines_x_frac": [_clamp(x / width) for x in boundaries],
        "bar_count": max(0, len(boundaries) - 1),
        "staff_lines_y_frac": [_clamp(y / height) for y in system.staff_lines],
    }


def _music_start_x(gray: np.ndarray, system: _SystemBand) -> int:
    """Estimate where musical content begins after clef/key/time prefix."""

    height, width = gray.shape
    system_width = system.x1 - system.x0
    fallback = int(round(system.x0 + system_width * 0.08))
    y0 = max(0, system.staff_top - 2)
    y1 = min(height, system.staff_bottom + 3)
    if system_width <= 0 or y1 <= y0:
        return fallback

    density = (gray[y0:y1, system.x0 : system.x1] < 170).sum(axis=0).astype(np.float32)
    if density.size == 0:
        return fallback
    smooth = gaussian_filter1d(density, sigma=2.0)
    baseline = float(np.percentile(smooth, 20))
    excess = np.maximum(0.0, smooth - baseline)
    scan_limit = max(1, min(len(excess), int(system_width * 0.28)))
    scan = excess[:scan_limit]
    if scan.size == 0:
        return fallback

    prefix_peak = float(np.percentile(scan, 95))
    if prefix_peak < 4.0:
        return system.x0
    spacing = int(round(float(np.median(np.diff(system.staff_lines))))) if len(system.staff_lines) > 1 else 10
    valley_width = max(8, spacing * 2)
    low_threshold = max(2.0, prefix_peak * 0.18)
    high_threshold = max(4.0, prefix_peak * 0.35)
    seen_dense = False
    run_start: int | None = None
    for offset, value in enumerate(scan):
        if value > high_threshold:
            seen_dense = True
        if (seen_dense or offset < valley_width) and value <= low_threshold:
            if run_start is None:
                run_start = offset
            if offset - run_start + 1 >= valley_width:
                detected = system.x0 + run_start
                # Bach-like pages often have x0 already just after the clef; a
                # later low-density run is then the first inter-note gap, not the
                # prefix boundary. Keep the actual left edge in that case.
                if system.x0 / width > 0.085 and detected - system.x0 > system_width * 0.04:
                    return system.x0
                return max(system.x0, min(fallback, detected))
        else:
            run_start = None
    return fallback


def _refine_barline_right_edge(gray: np.ndarray, system: _SystemBand, x: int) -> int:
    """Return the right edge of the actual ink extent around a barline peak."""

    height, width = gray.shape
    xi = max(0, min(width - 1, int(round(x))))
    x0 = max(0, xi - 8)
    x1 = min(width, xi + 9)
    y0 = max(0, system.staff_top - 2)
    y1 = min(height, system.staff_bottom + 3)
    if x1 <= x0 or y1 <= y0:
        return xi
    projection = (gray[y0:y1, x0:x1] < 180).sum(axis=0)
    peak = int(projection.max()) if projection.size else 0
    threshold = max(2, int(peak * 0.30))
    cols = np.where(projection >= threshold)[0]
    if cols.size == 0:
        return xi
    groups = _groups(cols, max_gap=2)
    local_x = xi - x0
    containing = [group for group in groups if group[0] - 1 <= local_x <= group[1] + 1]
    left, right = containing[0] if containing else min(groups, key=lambda group: min(abs(local_x - group[0]), abs(local_x - group[1])))
    return min(width - 1, x0 + int(right) + 1)


def _clean_candidates(candidates: list[tuple[int, float]], x0: int, x1: int, width: int) -> list[int]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda item: item[0])
    clusters: list[list[tuple[int, float]]] = []
    merge_gap = max(4, int(width * 0.004))
    for x, score in candidates:
        if x < x0 - 10 or x > x1 + 10:
            continue
        if clusters and x - clusters[-1][-1][0] <= merge_gap:
            clusters[-1].append((x, score))
        else:
            clusters.append([(x, score)])
    out: list[int] = []
    min_gap = max(18, int(width * 0.018))
    for cluster in clusters:
        total = sum(max(score, 1.0) for _x, score in cluster)
        center = int(round(sum(x * max(score, 1.0) for x, score in cluster) / total))
        if out and center - out[-1] < min_gap:
            # Keep the stronger of two implausibly close candidates.
            if sum(s for _x, s in cluster) > 1.1:
                out[-1] = center
            continue
        out.append(center)
    return out


def _filter_staff_crossing_candidates(
    gray: np.ndarray,
    system: _SystemBand,
    candidates: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    if not candidates or not system.staff_lines:
        return candidates
    height, width = gray.shape
    needed = max(5 if len(system.staff_lines) <= 5 else 8, int(math.ceil(len(system.staff_lines) * 0.80)))
    filtered: list[tuple[int, float]] = []
    for x, score in candidates:
        xi = max(0, min(width - 1, int(round(x))))
        hits = 0
        coverage = 0
        coverage_needed = 0
        for y in system.staff_lines:
            y0 = max(0, int(y) - 1)
            y1 = min(height, int(y) + 2)
            x0 = max(0, xi - 2)
            x1 = min(width, xi + 3)
            if np.any(gray[y0:y1, x0:x1] < 170):
                hits += 1
        for staff_start in range(0, len(system.staff_lines), 5):
            staff = system.staff_lines[staff_start : staff_start + 5]
            if len(staff) < 5:
                continue
            y0 = max(0, int(staff[0]) - 1)
            y1 = min(height, int(staff[-1]) + 2)
            x0 = max(0, xi - 1)
            x1 = min(width, xi + 2)
            col_dark = np.any(gray[y0:y1, x0:x1] < 170, axis=1)
            coverage += int(col_dark.sum())
            coverage_needed += int((y1 - y0) * 0.90)
        if hits >= needed and coverage >= coverage_needed:
            filtered.append((x, score + hits * 10.0 + coverage))
    return filtered


def _ensure_edge_boundaries(xs: list[int], x0: int, x1: int, width: int) -> list[int]:
    tolerance = max(18, int(width * 0.018))
    values = sorted(xs)
    if not values or abs(values[0] - x0) > tolerance:
        values.insert(0, x0)
    else:
        values[0] = min(values[0], x0)
    if abs(values[-1] - x1) > tolerance:
        values.append(x1)
    else:
        values[-1] = max(values[-1], x1)
    deduped: list[int] = []
    for x in values:
        if not deduped or x - deduped[-1] > 4:
            deduped.append(x)
    return deduped


def _dedupe_systems(systems: list[_SystemBand]) -> list[_SystemBand]:
    out: list[_SystemBand] = []
    for system in sorted(systems, key=lambda s: s.staff_top):
        if out and system.staff_top - out[-1].staff_top < 20:
            continue
        out.append(system)
    for index, system in enumerate(out, start=1):
        rec = _system_record.__name__  # keep linters from mistaking the loop as unused in older checks
        del rec
    return out


def _mark_count_mismatch(results: list[dict[str, Any]], expected_bar_count: int) -> None:
    detected = sum(int(item.get("bar_count") or 0) for item in results)
    if abs(detected - expected_bar_count) > max(2, int(expected_bar_count * 0.12)):
        for item in results:
            item.setdefault("metadata", {})["bar_count_mismatch"] = {
                "detected_page_bars": detected,
                "expected_page_bars": expected_bar_count,
            }
    for index, item in enumerate(results, start=1):
        item["system_index"] = index


def _groups(values: np.ndarray, *, max_gap: int) -> list[tuple[int, int]]:
    if values.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(values[0])
    for raw in values[1:]:
        value = int(raw)
        if value - prev <= max_gap:
            prev = value
        else:
            groups.append((start, prev))
            start = prev = value
    groups.append((start, prev))
    return groups


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(max(0.0, min(1.0, float(value))), 6)

