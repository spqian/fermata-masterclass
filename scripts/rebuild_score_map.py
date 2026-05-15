from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from masterclass.core.masterclasses import MasterclassStore
from masterclass.core.models import TenantContext
from masterclass.core.sessions import SessionStore
from masterclass.engine.score_map import build_score_map, persist_score_map
from masterclass.storage.local import LocalObjectStorage


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild score/score_map.json for a local lesson session.")
    parser.add_argument("session_id")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--user", default="default")
    parser.add_argument("--storage-root", default=str(ROOT / "local_adls"))
    args = parser.parse_args()

    storage = LocalObjectStorage(Path(args.storage_root))
    sessions = SessionStore(storage)
    masterclasses = MasterclassStore(storage)
    ctx = TenantContext(args.tenant, args.user)
    manifest = sessions.load_by_id(ctx, args.session_id)
    result = build_score_map(storage=storage, masterclass_store=masterclasses, store=sessions, manifest=manifest)
    persist_score_map(storage=storage, store=sessions, manifest=manifest, result=result)
    print(
        f"rebuilt {result.score_map_key}: "
        f"{len(result.systems)} systems, {len(result.bars)} bars, {len(result.notes)} notes"
    )


if __name__ == "__main__":
    main()
