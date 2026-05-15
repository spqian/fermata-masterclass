from __future__ import annotations

from dataclasses import dataclass

from masterclass.agent.llm import LlmProvider
from masterclass.agent_tools.registry import ToolRegistry
from masterclass.core.models import JobState, SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage


@dataclass
class TeacherAgent:
    provider: LlmProvider
    tools: ToolRegistry
    storage: ObjectStorage

    def teach(self, store: SessionStore, manifest: SessionManifest, *, model: str, max_tool_calls: int) -> SessionManifest:
        """Run the storage-safe teacher loop.

        This is the production seam for the PoC teach.py behavior: tools are
        tenant-scoped through ToolRegistry, LLM usage is recorded on the
        manifest, and the raw response is persisted as an artifact. The richer
        multimodal prompt/media assembly will be ported on top of this seam.
        """

        manifest.state = JobState.TEACHING
        store.save(manifest)
        evidence_key = manifest.artifacts.get("analysis/evidence_packet.md") or manifest.artifacts.get("evidence_packet.md")
        if evidence_key and self.storage.exists(evidence_key):
            evidence = self.storage.read_bytes(evidence_key).decode("utf-8")
        else:
            evidence = "No evidence packet is available yet. Summarize only the available manifest metadata."
        prior_context_key = manifest.artifacts.get("context/prior_lessons.json")
        if prior_context_key and self.storage.exists(prior_context_key):
            prior_context = self.storage.read_bytes(prior_context_key).decode("utf-8")
        else:
            prior_context = "No prior lesson context is available for this session."

        system_instruction = (
            "You are a music masterclass teacher. Use the supplied evidence and callable tools only for "
            "measurable claims. Distinguish perception, measurement, and pedagogical hypothesis."
        )
        contents = [
            "Session manifest:\n" + str(manifest.to_json()),
            "\nPrior lesson context:\n" + prior_context,
            "\nEvidence:\n" + evidence,
        ]

        def run_tool(name: str, args: dict) -> dict:
            return self.tools.call(self.storage, manifest.session, name, args)

        text, usage, tool_calls = self.provider.generate_with_tools(
            model=model,
            system_instruction=system_instruction,
            contents=contents,
            tools=self.tools.declarations(),
            max_tool_calls=max_tool_calls,
            tool_executor=run_tool,
        )
        output_key = store.artifact_key(manifest.session, "llm/raw_teacher_response.txt")
        self.storage.write_bytes(output_key, text.encode("utf-8"), content_type="text/plain")
        manifest.artifacts["llm/raw_teacher_response.txt"] = output_key
        manifest.llm_usage.append({
            "provider": usage.provider,
            "model": usage.model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "estimated_cost_usd": usage.estimated_cost_usd,
            "tool_calls": tool_calls,
        })
        manifest.state = JobState.READY
        store.save(manifest)
        return manifest
