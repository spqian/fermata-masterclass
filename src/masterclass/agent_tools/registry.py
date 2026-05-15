from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from masterclass.core.models import SessionRef
from masterclass.engine.instruments import InstrumentProfile, intonation_enabled_for_profile, load_instrument_profile
from masterclass.storage.base import ObjectStorage

from masterclass.agent_tools.get_frames import DESCRIPTION as GET_FRAMES_DESCRIPTION, GET_FRAMES_SCHEMA, get_frames
from masterclass.agent_tools.inspect_bar import DESCRIPTION as INSPECT_BAR_DESCRIPTION, INSPECT_BAR_SCHEMA, inspect_bar
from masterclass.agent_tools.inspect_chord import DESCRIPTION as INSPECT_CHORD_DESCRIPTION, INSPECT_CHORD_SCHEMA, inspect_chord
from masterclass.agent_tools.inspect_note import DESCRIPTION as INSPECT_NOTE_DESCRIPTION, INSPECT_NOTE_SCHEMA, inspect_note
from masterclass.agent_tools.inspect_voicing import DESCRIPTION as INSPECT_VOICING_DESCRIPTION, INSPECT_VOICING_SCHEMA, inspect_voicing
from masterclass.agent_tools.list_frames import DESCRIPTION as LIST_FRAMES_DESCRIPTION, LIST_FRAMES_SCHEMA, list_frames
from masterclass.agent_tools.listen import DESCRIPTION as LISTEN_DESCRIPTION, LISTEN_SCHEMA, listen
from masterclass.agent_tools.measure_dynamics import DESCRIPTION as MEASURE_DYNAMICS_DESCRIPTION, MEASURE_DYNAMICS_SCHEMA, measure_dynamics
from masterclass.agent_tools.measure_tempo import DESCRIPTION as MEASURE_TEMPO_DESCRIPTION, MEASURE_TEMPO_SCHEMA, measure_tempo
from masterclass.agent_tools.measure_trill import DESCRIPTION as MEASURE_TRILL_DESCRIPTION, MEASURE_TRILL_SCHEMA, measure_trill
from masterclass.agent_tools.measure_vibrato import DESCRIPTION as MEASURE_VIBRATO_DESCRIPTION, MEASURE_VIBRATO_SCHEMA, measure_vibrato
from masterclass.agent_tools.watch import DESCRIPTION as WATCH_DESCRIPTION, WATCH_SCHEMA, watch


ToolHandler = Callable[[ObjectStorage, SessionRef, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = spec

    def declarations(self) -> list[dict[str, Any]]:
        return [
            {"name": spec.name, "description": spec.description, "parameters": spec.schema}
            for spec in self._tools.values()
        ]

    def catalog(self) -> list[dict[str, str]]:
        return [{"name": spec.name, "description": spec.description} for spec in self._tools.values()]

    def call(self, storage: ObjectStorage, session: SessionRef, name: str, args: dict[str, Any]) -> dict[str, Any]:
        spec = self._tools.get(name)
        if spec is None:
            return {"error": f"unknown tool: {name}"}
        return spec.handler(storage, session, args)


ALL_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec("list_frames", LIST_FRAMES_DESCRIPTION, LIST_FRAMES_SCHEMA, list_frames),
    ToolSpec("get_frames", GET_FRAMES_DESCRIPTION, GET_FRAMES_SCHEMA, get_frames),
    ToolSpec("inspect_chord", INSPECT_CHORD_DESCRIPTION, INSPECT_CHORD_SCHEMA, inspect_chord),
    ToolSpec("measure_vibrato", MEASURE_VIBRATO_DESCRIPTION, MEASURE_VIBRATO_SCHEMA, measure_vibrato),
    ToolSpec("inspect_note", INSPECT_NOTE_DESCRIPTION, INSPECT_NOTE_SCHEMA, inspect_note),
    ToolSpec("inspect_bar", INSPECT_BAR_DESCRIPTION, INSPECT_BAR_SCHEMA, inspect_bar),
    ToolSpec("measure_tempo", MEASURE_TEMPO_DESCRIPTION, MEASURE_TEMPO_SCHEMA, measure_tempo),
    ToolSpec("measure_trill", MEASURE_TRILL_DESCRIPTION, MEASURE_TRILL_SCHEMA, measure_trill),
    ToolSpec("measure_dynamics", MEASURE_DYNAMICS_DESCRIPTION, MEASURE_DYNAMICS_SCHEMA, measure_dynamics),
    ToolSpec("listen", LISTEN_DESCRIPTION, LISTEN_SCHEMA, listen),
    ToolSpec("watch", WATCH_DESCRIPTION, WATCH_SCHEMA, watch),
    ToolSpec("inspect_voicing", INSPECT_VOICING_DESCRIPTION, INSPECT_VOICING_SCHEMA, inspect_voicing),
)

INTONATION_RELATED_TOOLS = {"measure_vibrato"}
PIANO_ONLY_TOOLS = {"inspect_voicing"}


def _coerce_profile(profile: InstrumentProfile | str | None) -> InstrumentProfile | None:
    if profile is None:
        return None
    if isinstance(profile, InstrumentProfile):
        return profile
    return load_instrument_profile(profile)


def _enabled_for_profile(spec: ToolSpec, profile: InstrumentProfile | None) -> bool:
    if profile is None:
        return True
    if spec.name in profile.disabled_tools:
        return False
    intonation_enabled = intonation_enabled_for_profile(profile) and profile.pitch_class != "fixed"
    if not intonation_enabled and spec.name in INTONATION_RELATED_TOOLS:
        return False
    if spec.name in PIANO_ONLY_TOOLS and profile.id != "piano":
        return False
    return True


def default_tool_registry(profile: InstrumentProfile | str | None = None) -> ToolRegistry:
    instrument_profile = _coerce_profile(profile)
    registry = ToolRegistry()
    for spec in ALL_TOOL_SPECS:
        if _enabled_for_profile(spec, instrument_profile):
            registry.register(spec)
    return registry
