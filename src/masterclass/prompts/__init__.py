"""Prompt templates for the masterclass engine."""


def load_teacher_prompt() -> str:
    """Load the teacher system prompt template."""
    from pathlib import Path

    return (Path(__file__).parent / "teacher_system.md").read_text(encoding="utf-8")
