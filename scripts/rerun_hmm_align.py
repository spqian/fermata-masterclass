from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from masterclass.core.models import TenantContext
from masterclass.core.sessions import SessionStore
from masterclass.engine.hmm_align import HmmAlignConfig, align_lesson_with_midi_hmm, persist_hmm_alignment
from masterclass.storage.local import LocalObjectStorage


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run only HMM alignment for a local lesson session.")
    parser.add_argument("session_id")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--user", default="default")
    parser.add_argument("--storage-root", default=str(ROOT / "local_adls"))
    parser.add_argument("--no-refine", action="store_true", help="Disable onset refinement.")
    args = parser.parse_args()

    storage = LocalObjectStorage(Path(args.storage_root))
    sessions = SessionStore(storage)
    ctx = TenantContext(args.tenant, args.user)
    manifest = sessions.load_by_id(ctx, args.session_id)

    midi_key = manifest.artifacts.get("masterclass/reference/midi")
    if not midi_key or not storage.exists(midi_key):
        raise RuntimeError("session manifest is missing masterclass/reference/midi")
    midi_bytes = storage.read_bytes(midi_key)

    started = time.time()
    result = align_lesson_with_midi_hmm(
        storage=storage,
        store=sessions,
        manifest=manifest,
        midi_bytes=midi_bytes,
        config=HmmAlignConfig(refine_with_onsets=False) if args.no_refine else None,
    )
    persist_hmm_alignment(storage=storage, store=sessions, manifest=manifest, result=result)

    print(f"re-ran HMM alignment in {time.time() - started:.1f}s")
    print(f"refinement_applied={result.refinement_applied}")
    print(f"notes_corrected={result.notes_with_onset_correction}/{len(result.notes)}")
    print(f"mean_onset_correction_ms={result.mean_onset_correction_ms}")
    print(f"bars_anchored={result.bars_anchored_to_onsets}/{len(result.measure_timestamps)}")
    print(f"bars_no_onset={result.bars_no_onset_match}")
    print("first 10 bar timestamps:")
    for row in result.measure_timestamps[:10]:
        print(f"  m{int(row['measure'])}: {float(row['start']):.3f}s")


if __name__ == "__main__":
    main()
