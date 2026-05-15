from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

from masterclass.core.models import JobState, SessionManifest
from masterclass.core.sessions import SessionStore


class PipelineStep(StrEnum):
    INGEST = "ingest"
    ANALYZE = "analyze"
    ALIGN = "align"
    GENERATE_EVIDENCE = "generate_evidence"


StepHandler = Callable[[SessionStore, SessionManifest], None]


@dataclass
class Pipeline:
    handlers: dict[PipelineStep, StepHandler]

    def run(self, store: SessionStore, manifest: SessionManifest, steps: list[PipelineStep]) -> SessionManifest:
        for step in steps:
            manifest.state = _state_for_step(step)
            store.save(manifest)
            handler = self.handlers.get(step)
            if handler is None:
                raise ValueError(f"no handler registered for pipeline step: {step}")
            handler(store, manifest)
        manifest.state = JobState.AWAITING_LLM
        store.save(manifest)
        return manifest


def _state_for_step(step: PipelineStep) -> JobState:
    return {
        PipelineStep.INGEST: JobState.INGESTING,
        PipelineStep.ANALYZE: JobState.ANALYZING,
        PipelineStep.ALIGN: JobState.ALIGNING,
        PipelineStep.GENERATE_EVIDENCE: JobState.GENERATING_EVIDENCE,
    }[step]

