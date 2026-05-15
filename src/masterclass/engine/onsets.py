from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from masterclass.core.models import SessionManifest
from masterclass.core.sessions import SessionStore
from masterclass.storage.base import ObjectStorage


PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass(frozen=True)
class RichOnsetsConfig:
    sample_rate: int = 22050
    bins_per_octave: int = 60
    n_octaves: int = 7
    hop_length: int = 512
    min_spacing_ms: float = 80.0
    peak_delta: float = 0.07
    pc_window_sec: float = 0.12
    fmin_note: str = "C2"


@dataclass
class RichOnsetsResult:
    events: list[dict[str, Any]]
    summary: dict[str, Any]
    markdown: str
    config: dict[str, Any]


def detect_rich_onsets(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    config: RichOnsetsConfig | None = None,
) -> RichOnsetsResult:
    """Detect spectral-flux onsets for a lesson and persist the analysis artifacts."""

    config = config or RichOnsetsConfig()
    audio_key = manifest.artifacts.get("artifacts/audio.wav")
    if not audio_key:
        raise ValueError("manifest is missing artifacts/audio.wav; run ingestion first")

    with tempfile.TemporaryDirectory(prefix="mc-rich-onsets-") as tmp_raw:
        audio_path = Path(tmp_raw) / "audio.wav"
        storage.read_to_file(audio_key, audio_path)
        result = rich_onsets_from_wav_file(audio_path, config, manifest=manifest, audio_key=audio_key)

    persist_rich_onsets(storage=storage, store=store, manifest=manifest, result=result)
    return result


def persist_rich_onsets(
    *,
    storage: ObjectStorage,
    store: SessionStore,
    manifest: SessionManifest,
    result: RichOnsetsResult,
) -> None:
    """Write rich-onset artifacts and stamp the session manifest."""

    json_key = store.artifact_key(manifest.session, "analysis/rich_onsets.json")
    md_key = store.artifact_key(manifest.session, "analysis/rich_onsets.md")
    storage.write_json(json_key, {
        "schema_version": 1,
        "summary": result.summary,
        "onsets": result.events,
        "config": result.config,
    })
    storage.write_bytes(md_key, result.markdown.encode("utf-8"), content_type="text/markdown")
    manifest.artifacts["analysis/rich_onsets.json"] = json_key
    manifest.artifacts["analysis/rich_onsets.md"] = md_key
    manifest.metadata["rich_onsets_summary"] = result.summary
    store.save(manifest)


def events_from_wav_file(path: Path | str, config: RichOnsetsConfig | None = None) -> list[dict[str, Any]]:
    """Return rich onset events for a local WAV file without storage/manifest access."""

    return rich_onsets_from_wav_file(Path(path), config or RichOnsetsConfig()).events


def rich_onsets_from_wav_file(
    path: Path,
    config: RichOnsetsConfig,
    *,
    manifest: SessionManifest | None = None,
    audio_key: str | None = None,
) -> RichOnsetsResult:
    import warnings

    import librosa
    import numpy as np

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    y, sr = librosa.load(str(path), sr=config.sample_rate, mono=True)
    duration = float(len(y) / sr) if sr else 0.0
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=config.hop_length)
    wait = max(0, int(config.min_spacing_ms / 1000.0 * sr / config.hop_length))
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=config.hop_length,
        delta=config.peak_delta,
        wait=wait,
        backtrack=False,
        units="frames",
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=config.hop_length)

    fmin = librosa.note_to_hz(config.fmin_note)
    n_bins = config.bins_per_octave * config.n_octaves
    cqt = np.abs(librosa.cqt(
        y,
        sr=sr,
        hop_length=config.hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=config.bins_per_octave,
    ))
    rms = librosa.feature.rms(y=y, hop_length=config.hop_length)[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-6))

    base_midi = float(librosa.note_to_midi(config.fmin_note))
    pc_window_frames = max(1, int(config.pc_window_sec * sr / config.hop_length))
    events: list[dict[str, Any]] = []
    for t in onset_times:
        frame = int(round(float(t) * sr / config.hop_length))
        f_lo = max(0, frame)
        f_hi = min(cqt.shape[1], frame + pc_window_frames)
        if f_hi <= f_lo:
            continue

        cqt_slice = cqt[:, f_lo:f_hi].mean(axis=1)
        pc_profile = np.zeros(12)
        for bin_index in range(n_bins):
            midi_pitch = base_midi + bin_index * 12.0 / config.bins_per_octave
            pc = int(round(midi_pitch)) % 12
            pc_profile[pc] += cqt_slice[bin_index]
        pc_profile /= pc_profile.sum() + 1e-9
        pc_order = np.argsort(pc_profile)[::-1]

        peak_bin = int(np.argmax(cqt_slice))
        peak_midi = base_midi + peak_bin * 12.0 / config.bins_per_octave
        rms_index = min(len(rms_db) - 1, frame)
        onset_index = min(len(onset_env) - 1, frame)
        events.append({
            "time": round(float(t), 3),
            "loudness_db": round(float(rms_db[rms_index]), 1) if len(rms_db) else None,
            "onset_strength": round(float(onset_env[onset_index]), 3) if len(onset_env) else None,
            "pc_top1": PC_NAMES[int(pc_order[0])],
            "pc_top1_strength": round(float(pc_profile[pc_order[0]]), 3),
            "pc_top2": PC_NAMES[int(pc_order[1])],
            "pc_top2_strength": round(float(pc_profile[pc_order[1]]), 3),
            "note_estimate": librosa.midi_to_note(int(round(peak_midi))),
            "peak_midi": round(float(peak_midi), 2),
        })

    if events:
        strengths = sorted(float(event["onset_strength"] or 0.0) for event in events)
        threshold = strengths[len(strengths) * 3 // 4]
        for event in events:
            event["is_strong"] = bool(float(event["onset_strength"] or 0.0) >= threshold)

    loudness_values = [float(event["loudness_db"]) for event in events if event.get("loudness_db") is not None]
    summary = {
        "session_id": manifest.session.session_id if manifest else None,
        "repertoire": manifest.repertoire if manifest else None,
        "movement": manifest.movement if manifest else None,
        "instrument": manifest.instrument if manifest else None,
        "audio": audio_key or str(path),
        "recording_duration_sec": round(duration, 3),
        "n_onsets": len(events),
        "n_strong": sum(1 for event in events if event.get("is_strong")),
        "median_loudness_db": round(float(np.median(loudness_values)), 1) if loudness_values else None,
        "method_notes": [
            "Onsets from librosa.onset.onset_detect on the spectral-flux envelope.",
            f"min_spacing_ms={config.min_spacing_ms}, peak_delta={config.peak_delta}",
            f"Per-onset pitch class profile from a {int(config.pc_window_sec * 1000)}ms CQT window after the attack.",
            "is_strong = top quartile by onset_strength.",
        ],
    }
    return RichOnsetsResult(
        events=events,
        summary=summary,
        markdown=rich_onsets_markdown(summary, events),
        config=asdict(config),
    )


def rich_onsets_markdown(summary: dict[str, Any], events: list[dict[str, Any]]) -> str:
    lines = [
        f"# Rich Onsets - {summary.get('repertoire') or 'Untitled'}",
        "",
        f"- Session: `{summary.get('session_id')}`",
        f"- Movement: `{summary.get('movement')}`",
        f"- Instrument: `{summary.get('instrument')}`",
        f"- Duration: `{summary.get('recording_duration_sec')}` sec",
        f"- Onsets: `{summary.get('n_onsets')}` (`{summary.get('n_strong')}` strong)",
        f"- Median loudness: `{summary.get('median_loudness_db')}` dB",
        "",
        "## Method",
    ]
    for note in summary.get("method_notes", []):
        lines.append(f"- {note}")
    lines.extend(["", "## First onsets"])
    for event in events[:40]:
        strong = " strong" if event.get("is_strong") else ""
        lines.append(
            f"- `{event.get('time'):.3f}s` {event.get('note_estimate')} "
            f"pc={event.get('pc_top1')} loud={event.get('loudness_db')}dB "
            f"strength={event.get('onset_strength')}{strong}"
        )
    return "\n".join(lines)
