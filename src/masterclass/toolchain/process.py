from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


def run_process(argv: list[str], *, timeout_sec: int = 900) -> ProcessResult:
    if not argv:
        raise ValueError("argv must not be empty")
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    result = ProcessResult(argv=argv, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"process failed ({completed.returncode}): {' '.join(argv)}\n{completed.stderr[-2000:]}")
    return result

