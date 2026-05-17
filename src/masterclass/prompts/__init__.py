"""Prompt templates for the masterclass engine."""


def load_teacher_prompt() -> str:
    """Load the teacher system prompt template."""
    from pathlib import Path

    return (Path(__file__).parent / "teacher_system.md").read_text(encoding="utf-8")


def load_drill_evaluator_prompt() -> str:
    """Load the drill evaluator (practice-clip feedback) prompt."""
    from pathlib import Path

    return (Path(__file__).parent / "drill_evaluator.md").read_text(encoding="utf-8")
