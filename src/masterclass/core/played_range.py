"""Canonical model for "the range of measures the player actually played".

Every stage and view that needs to filter content to the played range should
derive it here, exactly once, instead of re-implementing the priority order
each time. The priority is:

1. ``manifest.metadata.first_measure`` + ``last_measure`` -- the explicit
   window the user typed when uploading the lesson. Source label:
   ``"user_specified"``.
2. ``manifest.metadata.auto_detected_first_measure`` +
   ``auto_detected_last_measure`` -- populated by audio-truth's auto-range
   detector when the user did not supply an explicit window. Source label:
   ``"auto_detected"``.
3. The score's natural extent from the bound masterclass (the movement's
   ``first_measure`` / ``last_measure``, or aggregated page extents). Source
   label: ``"score_extent"``.

The output is a frozen :class:`PlayedRange` carrying both the bounds and a
``source`` tag so callers can render "scoped to m.X-Y (auto-detected)" hints
in their UI without re-deriving the priority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from masterclass.core.models import MasterclassManifest, SessionManifest


@dataclass(frozen=True)
class PlayedRange:
    first_measure: int
    last_measure: int
    source: str  # "user_specified" | "auto_detected" | "score_extent" | "score_map"

    def contains(self, measure: Any) -> bool:
        """True iff ``measure`` (coerced to int) falls within ``[first, last]``."""
        if measure is None:
            return False
        try:
            m = int(measure)
        except (TypeError, ValueError):
            return False
        return self.first_measure <= m <= self.last_measure

    def label(self) -> str:
        return f"m.{self.first_measure}-{self.last_measure}"

    def measures(self) -> range:
        """Iterate measure numbers in the played range (inclusive)."""
        return range(self.first_measure, self.last_measure + 1)

    def filter_by_measure(self, items: Any, key: str = "measure") -> list[Any]:
        """Return only items whose ``key`` attribute or dict-field is in range.

        Works on both dicts (``item[key]``) and objects (``item.key``).
        Items with a missing or non-numeric measure are dropped — call sites
        that want to KEEP unmatched-measure rows should filter them in.

        This is THE pinch-point: every consumer that needs to scope a
        notes/comments/regions list to the lesson sandbox should use this
        helper, not roll their own filter. Doing so guarantees the
        played-range invariant cannot be silently violated by future code.
        """
        out: list[Any] = []
        for item in (items or []):
            measure: Any
            if isinstance(item, dict):
                measure = item.get(key)
            else:
                measure = getattr(item, key, None)
            if self.contains(measure):
                out.append(item)
        return out

    def filter_time_window(
        self,
        items: Any,
        perf_start_sec: float | None,
        perf_end_sec: float | None,
        start_key: str = "start",
        end_key: str = "end",
    ) -> list[Any]:
        """Return only items overlapping ``[perf_start_sec, perf_end_sec]``.

        Drops items entirely outside the lesson's perf-time envelope. Used
        for ranked-regions / dynamics-summary style rows that carry seconds
        instead of measure numbers but still belong inside the sandbox.
        """
        if perf_start_sec is None or perf_end_sec is None:
            return list(items or [])
        out: list[Any] = []
        for item in (items or []):
            s = item.get(start_key) if isinstance(item, dict) else getattr(item, start_key, None)
            e = item.get(end_key) if isinstance(item, dict) else getattr(item, end_key, None)
            if s is None and e is None:
                continue
            try:
                s_f = float(s) if s is not None else None
                e_f = float(e) if e is not None else s_f
            except (TypeError, ValueError):
                continue
            if s_f is None:
                s_f = e_f
            if e_f is None:
                e_f = s_f
            if s_f is None or e_f is None:
                continue
            if e_f < perf_start_sec or s_f > perf_end_sec:
                continue
            out.append(item)
        return out


def _as_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def derive_played_range(
    manifest: SessionManifest,
    masterclass: MasterclassManifest | None = None,
) -> PlayedRange:
    """Resolve the played range for a lesson.

    See module docstring for the priority order. ``masterclass`` may be
    ``None`` -- in that case the score-extent fallback degrades to whatever
    partial bounds the manifest carries (or ``m.1-1`` as a last resort).
    """
    meta = manifest.metadata or {}
    first = _as_int(meta.get("first_measure"))
    last = _as_int(meta.get("last_measure"))
    if first is not None and last is not None and last >= first:
        return PlayedRange(first, last, "user_specified")

    auto_first = _as_int(meta.get("auto_detected_first_measure"))
    auto_last = _as_int(meta.get("auto_detected_last_measure"))
    if auto_first is not None and auto_last is not None and auto_last >= auto_first:
        return PlayedRange(auto_first, auto_last, "auto_detected")

    extent_first: int | None = None
    extent_last: int | None = None
    if masterclass is not None:
        mc_meta = masterclass.metadata or {}
        extent_first = _as_int(
            mc_meta.get("first_measure"),
            mc_meta.get("score_first_measure"),
        )
        extent_last = _as_int(
            mc_meta.get("last_measure"),
            mc_meta.get("score_last_measure"),
        )
    extent_first = extent_first or first or auto_first or 1
    extent_last = extent_last or last or auto_last or extent_first
    if extent_last < extent_first:
        extent_last = extent_first
    return PlayedRange(int(extent_first), int(extent_last), "score_extent")


def derive_played_range_from_score_map(score_map: Any) -> PlayedRange | None:
    """Resolve the played range from an already-built ``score_map.json`` dict.

    Useful for code paths that have neither the SessionManifest nor the
    MasterclassManifest available (e.g. agent tools that only read
    score artifacts). Returns ``None`` if the dict does not carry usable
    extent fields.
    """
    if not isinstance(score_map, dict):
        return None
    first = _as_int(score_map.get("first_measure"), score_map.get("played_lo"))
    last = _as_int(score_map.get("last_measure"), score_map.get("played_hi"))
    if first is None or last is None or last < first:
        return None
    return PlayedRange(first, last, "score_map")
