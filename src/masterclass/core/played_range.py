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
