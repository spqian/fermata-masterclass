from __future__ import annotations

import base64
import io
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from masterclass.core.models import SessionRef, session_prefix
from masterclass.storage.base import ObjectStorage


def session_key(session: SessionRef, rel: str) -> str:
    return f"{session_prefix(session)}/{rel}"


def exists(storage: ObjectStorage, session: SessionRef, rel: str) -> bool:
    return storage.exists(session_key(session, rel))


def read_json(storage: ObjectStorage, session: SessionRef, rel: str, default: Any = None) -> Any:
    key = session_key(session, rel)
    if not storage.exists(key):
        return default
    return storage.read_json(key)


def read_first_json(storage: ObjectStorage, session: SessionRef, rels: list[str], default: Any = None) -> Any:
    for rel in rels:
        data = read_json(storage, session, rel, None)
        if data is not None:
            return data
    return default


def list_session_keys(storage: ObjectStorage, session: SessionRef, rel: str) -> list[str]:
    prefix = session_key(session, rel).rstrip("/")
    return sorted(storage.list_keys(prefix))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ffmpeg_path() -> str:
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    bundled = _project_root() / "tools" / "ffmpeg" / "bin" / exe
    return str(bundled) if bundled.exists() else (shutil.which("ffmpeg") or "ffmpeg")


def load_audio(storage: ObjectStorage, session: SessionRef, sr: int = 22050):
    import librosa

    key = session_key(session, "artifacts/audio.wav")
    if not storage.exists(key):
        raise FileNotFoundError("artifacts/audio.wav missing")
    return librosa.load(io.BytesIO(storage.read_bytes(key)), sr=sr, mono=True)


def midi_pitch_to_name(pitch: int) -> str:
    import librosa

    return str(librosa.midi_to_note(int(pitch)))


def _resolve_input_key(storage: ObjectStorage, session: SessionRef, input_rel: str) -> str | None:
    direct = session_key(session, input_rel)
    if storage.exists(direct):
        return direct
    if input_rel == "input/source_video":
        prefix = session_key(session, "input").rstrip("/") + "/"
        for k in storage.list_keys(prefix):
            low = k.lower()
            if low.endswith((".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi")):
                return k
    return None


def run_ffmpeg_from_storage(storage: ObjectStorage, session: SessionRef, input_rel: str, output_name: str, args: list[str]) -> tuple[bytes | None, str | None]:
    in_key = _resolve_input_key(storage, session, input_rel)
    if in_key is None:
        return None, f"{input_rel} missing"
    with TemporaryDirectory(prefix="mc-agent-tools-") as tmp_raw:
        tmp = Path(tmp_raw)
        source = tmp / Path(in_key).name
        out = tmp / output_name
        storage.read_to_file(in_key, source)
        if "--" in args:
            marker = args.index("--")
            pre_args = args[:marker]
            post_args = args[marker + 1:]
        else:
            pre_args = args
            post_args = []
        cmd = [ffmpeg_path(), "-y", "-loglevel", "error", *pre_args, "-i", str(source), *post_args, str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return None, f"ffmpeg failed: {res.stderr[:500]}"
        if not out.exists():
            return None, "ffmpeg produced no output"
        return out.read_bytes(), None


def extract_video_frames(storage: ObjectStorage, session: SessionRef, start: float, end: float, fps: float) -> tuple[list[dict[str, Any]], str | None]:
    video_key = _resolve_input_key(storage, session, "input/source_video")
    if video_key is None:
        return [], "input/source_video missing"
    with TemporaryDirectory(prefix="mc-agent-frames-") as tmp_raw:
        tmp = Path(tmp_raw)
        source = tmp / Path(video_key).name
        out_dir = tmp / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        storage.read_to_file(video_key, source)
        pattern = out_dir / "x_%03d.jpg"
        cmd = [
            ffmpeg_path(), "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-i", str(source), "-vf", f"fps={fps}", "-q:v", "3", str(pattern),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return [], f"ffmpeg failed: {res.stderr[:500]}"
        frames = sorted(out_dir.glob("x_*.jpg"))
        out: list[dict[str, Any]] = []
        count = len(frames)
        for i, path in enumerate(frames):
            t = start + (i * (end - start) / max(1, count - 1) if count > 1 else (end - start) / 2)
            rel = f"artifacts/frames/extra/x_{int(start*1000):07d}_{i+1:03d}.jpg"
            storage.write_bytes(session_key(session, rel), path.read_bytes(), content_type="image/jpeg")
            out.append({
                "path": rel,
                "estimated_time_sec": round(float(t), 3),
                "mime_type": "image/jpeg",
                "base64_jpeg": base64.b64encode(path.read_bytes()).decode("ascii"),
            })
        return out, None


def find_score_note(score_map: dict[str, Any], midi_measure: int, beat: float | None = None, pitch: str | None = None) -> dict[str, Any] | None:
    candidates = [n for n in score_map.get("notes", []) if int(n.get("midi_measure", n.get("measure", -999))) == midi_measure]
    if pitch:
        candidates = [n for n in candidates if pitch in (n.get("names") or n.get("note_names") or [])]
    if beat is not None:
        candidates.sort(key=lambda n: abs(float(n.get("beat_in_bar", n.get("beat", 0.0))) - beat))
    return candidates[0] if candidates else None


def note_score_time(note: dict[str, Any]) -> float | None:
    for key in ("score_time", "score_time_in_movement", "score_time_sec", "score_time_local"):
        if note.get(key) is not None:
            return float(note[key])
    return None


def note_perf_time(note: dict[str, Any]) -> float | None:
    for key in ("perf_time", "performed_time_sec"):
        if note.get(key) is not None:
            return float(note[key])
    return None

