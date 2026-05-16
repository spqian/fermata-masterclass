"""Typed catalog for resolving session/masterclass artifact storage keys.

Centralizes the lookup of canonical artifact paths so callers no longer
embed magic strings like ``manifest.artifacts.get("masterclass/reference/musicxml.musicxml")``.
The path list in :data:`ArtifactCatalog._PATHS` is the single source of
truth; the public methods are thin wrappers that walk the candidate list
and return the first hit (or raise via the ``*_required`` variants).
"""

from __future__ import annotations

from typing import Optional

from masterclass.core.models import MasterclassManifest, SessionManifest


_MASTERCLASS_PREFIX = "masterclass/"


class ArtifactMissingError(KeyError):
    """Raised by ``*_required`` getters when no candidate artifact exists."""


class ArtifactCatalog:
    """Resolve logical artifact names to concrete storage keys.

    The catalog consults the session manifest first. For artifacts that
    can live on the parent masterclass (currently only MusicXML), the
    masterclass manifest is checked next, so consumers that hold the
    masterclass directly (e.g. score_map staging) can use the same API.
    """

    # Logical name -> ordered list of candidate keys.
    # Keys are session-manifest-relative. Entries that start with
    # ``masterclass/`` are also looked up against the masterclass manifest
    # (with the prefix stripped) when one is supplied.
    _PATHS: dict[str, tuple[str, ...]] = {
        "audio_wav": ("artifacts/audio.wav",),
        "aligned_notes": (
            "analysis/audio_truth_matched_notes.json",
            "analysis/audio_truth_notes.json",
            "analysis/hmm_aligned_notes.json",
        ),
        "musicxml": (
            "masterclass/reference/musicxml.musicxml",
            "masterclass/reference/musicxml.mxl",
            "masterclass/reference/musicxml",
        ),
        "score_prep": ("reference/score_prep.json",),
        "score_map": ("score/score_map.json",),
        "evidence_packet": (
            "analysis/evidence_packet.md",
            "evidence_packet.md",
        ),
        "analysis_json": ("analysis/analysis.json",),
        "analysis_md": ("analysis/analysis.md",),
        "audio_truth_matched": ("analysis/audio_truth_matched_notes.json",),
        "audio_truth_raw": ("analysis/audio_truth_notes.json",),
        "polyphonic_intonation": ("analysis/polyphonic_intonation.json",),
        "polyphonic_rhythm": ("analysis/polyphonic_rhythm.json",),
        "piano_voicing": ("analysis/piano_voicing.json",),
        "rich_onsets": ("analysis/rich_onsets.json",),
    }

    def __init__(
        self,
        session: SessionManifest,
        masterclass: Optional[MasterclassManifest] = None,
    ) -> None:
        self._session = session
        self._masterclass = masterclass

    # ---- generic helpers --------------------------------------------------

    def _candidates(self, name: str) -> tuple[str, ...]:
        try:
            return self._PATHS[name]
        except KeyError as exc:  # pragma: no cover - guard against typos
            raise KeyError(f"unknown logical artifact name: {name!r}") from exc

    def _lookup(self, name: str) -> Optional[str]:
        for candidate in self._candidates(name):
            key = self._session.artifacts.get(candidate)
            if key:
                return key
            if self._masterclass is not None and candidate.startswith(_MASTERCLASS_PREFIX):
                relative = candidate[len(_MASTERCLASS_PREFIX):]
                key = self._masterclass.artifacts.get(relative)
                if key:
                    return key
        return None

    def _required(self, name: str) -> str:
        value = self._lookup(name)
        if value:
            return value
        sid = self._session.session.session_id
        raise ArtifactMissingError(
            f"required artifact {name!r} missing for session {sid}; "
            f"candidates checked: {list(self._candidates(name))}"
        )

    # ---- typed getters ----------------------------------------------------

    def audio_wav(self) -> Optional[str]:
        return self._lookup("audio_wav")

    def audio_wav_required(self) -> str:
        return self._required("audio_wav")

    def aligned_notes(self) -> Optional[str]:
        return self._lookup("aligned_notes")

    def aligned_notes_required(self) -> str:
        return self._required("aligned_notes")

    def musicxml(self) -> Optional[str]:
        return self._lookup("musicxml")

    def musicxml_required(self) -> str:
        return self._required("musicxml")

    def score_prep(self) -> Optional[str]:
        return self._lookup("score_prep")

    def score_prep_required(self) -> str:
        return self._required("score_prep")

    def score_map(self) -> Optional[str]:
        return self._lookup("score_map")

    def score_map_required(self) -> str:
        return self._required("score_map")

    def evidence_packet(self) -> Optional[str]:
        return self._lookup("evidence_packet")

    def evidence_packet_required(self) -> str:
        return self._required("evidence_packet")

    def analysis_json(self) -> Optional[str]:
        return self._lookup("analysis_json")

    def analysis_md(self) -> Optional[str]:
        return self._lookup("analysis_md")

    def audio_truth_matched(self) -> Optional[str]:
        return self._lookup("audio_truth_matched")

    def audio_truth_raw(self) -> Optional[str]:
        return self._lookup("audio_truth_raw")

    def polyphonic_intonation(self) -> Optional[str]:
        return self._lookup("polyphonic_intonation")

    def polyphonic_rhythm(self) -> Optional[str]:
        return self._lookup("polyphonic_rhythm")

    def piano_voicing(self) -> Optional[str]:
        return self._lookup("piano_voicing")

    def rich_onsets(self) -> Optional[str]:
        return self._lookup("rich_onsets")


__all__ = ["ArtifactCatalog", "ArtifactMissingError"]
