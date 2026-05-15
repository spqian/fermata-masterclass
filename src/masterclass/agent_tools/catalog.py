from __future__ import annotations

from masterclass.engine.instruments import InstrumentProfile
from masterclass.agent_tools.registry import default_tool_registry


def tool_catalog_text(profile: InstrumentProfile | str | None = None) -> str:
    """Return a human-readable tool catalog block for prompt injection."""

    rows = default_tool_registry(profile).catalog()
    lines = ["Tool catalog:"]
    lines.extend(f"- {row['name']}: {row['description']}" for row in rows)
    return "\n".join(lines)
