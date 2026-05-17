from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_tools_downloads() -> Path:
    path = _project_root() / "tools" / "downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_java(java_home: str | None) -> Path:
    """Locate the Java runtime that will execute Audiveris.

    Resolution order:
    1. ``java_home`` argument when explicitly supplied by the caller.
    2. ``JAVA_HOME`` environment variable (standard Linux/Docker convention).
    3. Bundled ``tools/jre`` under the project root (Windows dev box).
    """
    if java_home:
        home = Path(java_home)
    elif os.environ.get("JAVA_HOME"):
        home = Path(os.environ["JAVA_HOME"])
    else:
        home = _project_root() / "tools" / "jre"
    exe = home / "bin" / ("java.exe" if os.name == "nt" else "java")
    if not exe.exists():
        raise RuntimeError(
            f"Audiveris Java runtime is not installed at {exe}. "
            "Set JAVA_HOME, or install the bundled JRE under tools/jre."
        )
    return exe


def _resolve_audiveris_app(audiveris_home: str | None) -> Path:
    """Locate the directory containing ``audiveris.jar``.

    Resolution order:
    1. ``audiveris_home`` argument when explicitly supplied by the caller.
    2. ``AUDIVERIS_HOME`` environment variable.
    3. Bundled ``tools/audiveris`` under the project root.

    Also searches the standard Debian/Ubuntu install paths
    (``/opt/Audiveris/lib`` and ``/usr/share/audiveris/lib``) so the .deb
    package layout works without any env var.
    """
    if audiveris_home:
        home = Path(audiveris_home)
    elif os.environ.get("AUDIVERIS_HOME"):
        home = Path(os.environ["AUDIVERIS_HOME"])
    else:
        home = _project_root() / "tools" / "audiveris"
    candidates = [
        home / "lib",
        home / "app",
        home / "Audiveris" / "app",
        Path("/opt/Audiveris/lib"),
        Path("/opt/Audiveris/app"),
        Path("/usr/share/audiveris/lib"),
    ]
    candidates.extend(path.parent for path in home.rglob("audiveris.jar") if path.is_file())
    for app_dir in candidates:
        jar = app_dir / "audiveris.jar"
        if jar.exists():
            return app_dir
    raise RuntimeError(
        f"Audiveris is not installed under {home}. Expected an audiveris.jar "
        "inside tools/audiveris (Windows) or /opt/Audiveris/lib (Linux .deb)."
    )


def _audiveris_command(java_exe: Path, app_dir: Path) -> list[str]:
    return [str(java_exe), "-cp", str(app_dir / "*"), "Audiveris"]


def _count_pdf_pages(pdf_path: Path) -> int | None:
    try:
        import fitz

        with fitz.open(str(pdf_path)) as document:
            return int(document.page_count)
    except Exception:
        return None


def _music_sheet_ranges(pdf_path: Path) -> list[str]:
    try:
        import fitz
        from masterclass.engine.staff_detection import detect_staff_systems_from_image

        pages: list[int] = []
        with fitz.open(str(pdf_path)) as document:
            matrix = fitz.Matrix(100 / 72.0, 100 / 72.0)
            for index, page in enumerate(document, start=1):
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                system_count = len(detect_staff_systems_from_image(pixmap.tobytes("png")))
                if system_count > 0:
                    pages.append(index)
        if not pages:
            return []
        ranges: list[str] = []
        start = prev = pages[0]
        for page in pages[1:]:
            if page == prev + 1:
                prev = page
                continue
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = page
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        return ranges
    except Exception:
        return []


def _extract_version(output: str) -> str | None:
    for pattern in (r"Audiveris:\s*([^\s]+)", r"Audiveris Version:\s*([\w.\-:+]+)"):
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _pick_musicxml_output(out_dir: Path) -> Path:
    candidates = [
        path
        for path in out_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mxl", ".xml", ".musicxml"}
    ]
    candidates = [path for path in candidates if "container" not in path.name.lower()]
    if not candidates:
        raise RuntimeError(f"Audiveris completed but produced no MusicXML output in {out_dir}")
    candidates.sort(key=lambda path: (path.suffix.lower() != ".mxl", -path.stat().st_size, str(path)))
    return candidates[0]


def audiveris_pdf_to_musicxml(
    pdf_bytes: bytes,
    *,
    page_dpi: int = 300,
    timeout_sec: int = 600,
    java_home: str | None = None,
    audiveris_home: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Run bundled Audiveris on a PDF and return MusicXML/MXL bytes plus metadata."""

    java_exe = _resolve_java(java_home)
    app_dir = _resolve_audiveris_app(audiveris_home)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="audiveris-run-", dir=str(_default_tools_downloads())) as tmp_raw:
        tmp = Path(tmp_raw)
        pdf_path = tmp / "input.pdf"
        out_dir = tmp / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(pdf_bytes)
        pages = _count_pdf_pages(pdf_path)

        base_cmd = [
            *_audiveris_command(java_exe, app_dir),
            "-batch",
        ]
        cmd = [
            *base_cmd,
            "-export",
            "-output",
            str(out_dir),
            str(pdf_path),
        ]
        env = os.environ.copy()
        env["JAVA_HOME"] = str(java_exe.parents[1])
        env["PATH"] = str(java_exe.parent) + os.pathsep + env.get("PATH", "")
        def run(command: list[str]) -> subprocess.CompletedProcess[str]:
            try:
                return subprocess.run(
                    command,
                    cwd=str(_project_root()),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Audiveris timed out after {timeout_sec}s") from exc

        result = run(cmd)

        elapsed = time.perf_counter() - started
        combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.returncode != 0:
            tail = combined_output[-4000:].strip()
            raise RuntimeError(f"Audiveris failed with exit code {result.returncode}: {tail}")

        try:
            output = _pick_musicxml_output(out_dir)
        except RuntimeError as first_error:
            sheet_ranges = _music_sheet_ranges(pdf_path)
            if not sheet_ranges:
                raise first_error
            retry_out = tmp / "out_music_sheets"
            retry_out.mkdir(parents=True, exist_ok=True)
            retry_cmd = [
                *base_cmd,
                "-sheets",
                *sheet_ranges,
                "-export",
                "-output",
                str(retry_out),
                str(pdf_path),
            ]
            retry = run(retry_cmd)
            combined_output += "\n" + (retry.stdout or "") + "\n" + (retry.stderr or "")
            if retry.returncode != 0:
                tail = combined_output[-4000:].strip()
                raise RuntimeError(f"Audiveris failed on detected music sheets with exit code {retry.returncode}: {tail}") from first_error
            output = _pick_musicxml_output(retry_out)
            metadata_retry = {"selected_sheets": " ".join(sheet_ranges)}
        else:
            metadata_retry = {}
        data = output.read_bytes()
        xml_format = "mxl" if output.suffix.lower() == ".mxl" else "xml"
        metadata: dict[str, Any] = {
            "pages_processed": pages,
            "time_taken_sec": round(elapsed, 3),
            "audiveris_version": _extract_version(combined_output),
            "xml_format": xml_format,
            "page_dpi": page_dpi,
            "output_filename": output.name,
            **metadata_retry,
        }
        return data, metadata
