from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from masterclass.toolchain.process import run_process


@dataclass(frozen=True)
class FfmpegToolchain:
    ffmpeg: Path
    ffprobe: Path

    @staticmethod
    def discover(ffmpeg: str | None = None, ffprobe: str | None = None) -> "FfmpegToolchain":
        bundled = Path(__file__).resolve().parents[3] / "tools" / "ffmpeg" / "bin"
        bundled_ffmpeg = bundled / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        bundled_ffprobe = bundled / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
        ffmpeg_path = (
            ffmpeg
            or os.environ.get("MASTERCLASS_FFMPEG")
            or (str(bundled_ffmpeg) if bundled_ffmpeg.exists() else None)
            or shutil.which("ffmpeg")
        )
        ffprobe_path = (
            ffprobe
            or os.environ.get("MASTERCLASS_FFPROBE")
            or (str(bundled_ffprobe) if bundled_ffprobe.exists() else None)
            or shutil.which("ffprobe")
        )
        if not ffmpeg_path or not Path(ffmpeg_path).exists():
            raise FileNotFoundError("ffmpeg not found; pass --ffmpeg or set MASTERCLASS_FFMPEG")
        if not ffprobe_path or not Path(ffprobe_path).exists():
            raise FileNotFoundError("ffprobe not found; pass --ffprobe or set MASTERCLASS_FFPROBE")
        return FfmpegToolchain(Path(ffmpeg_path).resolve(), Path(ffprobe_path).resolve())

    def probe(self, media: Path) -> dict:
        result = run_process(
            [
                str(self.ffprobe),
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(media),
            ],
            timeout_sec=120,
        )
        return json.loads(result.stdout)

    def extract_audio(self, media: Path, wav: Path) -> None:
        wav.parent.mkdir(parents=True, exist_ok=True)
        run_process(
            [
                str(self.ffmpeg),
                "-y",
                "-i",
                str(media),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "44100",
                "-sample_fmt",
                "s16",
                str(wav),
            ],
            timeout_sec=1800,
        )

    def extract_frames(self, media: Path, frame_dir: Path, *, every_seconds: float = 10.0) -> list[Path]:
        if every_seconds <= 0:
            raise ValueError("frame interval must be positive")
        frame_dir.mkdir(parents=True, exist_ok=True)
        run_process(
            [
                str(self.ffmpeg),
                "-y",
                "-i",
                str(media),
                "-vf",
                f"fps=1/{every_seconds}",
                "-q:v",
                "3",
                str(frame_dir / "frame_%04d.jpg"),
            ],
            timeout_sec=1800,
        )
        return sorted(frame_dir.glob("frame_*.jpg"))

