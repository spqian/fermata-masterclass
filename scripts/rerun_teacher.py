"""Re-run the post-HMM lesson stages: intonation, rhythm, voicing, mechanical
comments, score_map rebuild, and the agentic teacher.

Use this after fixing alignment scope (e.g. auto-detected last_measure) without
re-extracting media or re-running HMM. The pipeline reads existing artifacts and
overwrites the downstream outputs in place.

Usage:
    tools\\python\\python.exe scripts\\rerun_teacher.py SESSION_ID [--tenant default] [--user default]

Set MASTERCLASS_LOCAL_ADLS_ROOT or rely on the project default (./local_adls).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    _load_env(project_root / ".env")
    sys.path.insert(0, str(project_root / "src"))
    os.environ.setdefault("MASTERCLASS_LOCAL_ADLS_ROOT", str(project_root / "local_adls"))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_id")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--user", default="default")
    ap.add_argument("--skip-teacher", action="store_true", help="Skip the LLM teach call (just rebuild mechanical/score_map).")
    args = ap.parse_args()

    from masterclass.core.models import TenantContext
    from masterclass.core.sessions import SessionStore
    from masterclass.core.masterclasses import MasterclassStore
    from masterclass.storage.local import LocalObjectStorage
    from masterclass.engine.score_map import build_score_map, persist_score_map
    from masterclass.engine.mechanical_comments import generate_mechanical_comments, persist_mechanical_comments
    from masterclass.engine.intonation import analyze_intonation
    from masterclass.engine.rhythm import analyze_rhythm, persist_rhythm
    from masterclass.engine.voicing import analyze_voicing, persist_voicing
    from masterclass.engine.teach_lesson import TeachConfig, teach_lesson
    from masterclass.agent.gemini import SharedKeyGeminiProvider
    from masterclass.agent.llm import SharedKeyGeminiConfig

    storage = LocalObjectStorage(Path(os.environ["MASTERCLASS_LOCAL_ADLS_ROOT"]))
    ctx = TenantContext(tenant_id=args.tenant, user_id=args.user)
    store = SessionStore(storage=storage)
    masterclasses = MasterclassStore(storage=storage)
    manifest = store.load_by_id(ctx, args.session_id)

    masterclass_id = manifest.metadata.get("masterclass_id")
    class_manifest = masterclasses.load_by_id(ctx, masterclass_id) if masterclass_id else None

    print(f"[1/5] re-running intonation analysis...")
    try:
        analyze_intonation(storage=storage, store=store, manifest=manifest)
        print("       ok")
    except Exception as exc:
        print(f"       skipped: {type(exc).__name__}: {exc}")

    print(f"[2/5] re-running rhythm analysis...")
    try:
        result = analyze_rhythm(storage=storage, store=store, manifest=manifest)
        persist_rhythm(storage=storage, store=store, manifest=manifest, result=result)
        print("       ok")
    except Exception as exc:
        print(f"       skipped: {type(exc).__name__}: {exc}")

    print(f"[3/5] re-running voicing analysis...")
    try:
        result = analyze_voicing(storage=storage, store=store, manifest=manifest)
        persist_voicing(storage=storage, store=store, manifest=manifest, result=result)
        print("       ok")
    except Exception as exc:
        print(f"       skipped: {type(exc).__name__}: {exc}")

    print(f"[4/5] rebuilding score_map and mechanical comments...")
    score_result = build_score_map(storage=storage, masterclass_store=masterclasses, store=store, manifest=manifest)
    persist_score_map(storage=storage, store=store, manifest=manifest, result=score_result)
    mech = generate_mechanical_comments(storage=storage, store=store, manifest=manifest)
    persist_mechanical_comments(storage=storage, store=store, manifest=manifest, result=mech)
    print(f"       ok ({len(mech.comments) if mech.comments else 0} mechanical comments)")

    if args.skip_teacher:
        print("[5/5] teacher skipped (--skip-teacher)")
        return 0

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[5/5] GEMINI_API_KEY not set; teacher skipped")
        return 0

    # Rebuild prior_lessons.json from current state so the teacher sees up-to-date
    # comments + lesson takeaways from sibling sessions in the same masterclass.
    if class_manifest is not None:
        print(f"[4b/5] rebuilding prior-lessons context from sibling sessions...")
        prior = _build_prior_lesson_context_inline(storage, store, masterclasses, class_manifest, manifest.session.session_id)
        prior_key = store.artifact_key(manifest.session, "context/prior_lessons.json")
        storage.write_json(prior_key, prior)
        manifest.artifacts["context/prior_lessons.json"] = prior_key
        store.save(manifest)
        n = len(prior.get("lessons") or [])
        comments_total = sum(len(l.get("teacher_comments") or []) for l in (prior.get("lessons") or []))
        print(f"       ok ({n} prior lesson(s), {comments_total} teacher comment(s) included)")

    print(f"[5/5] re-running agentic teacher (Gemini 2.5 Pro)...")
    provider = SharedKeyGeminiProvider(config=SharedKeyGeminiConfig())
    score_pages: list[bytes] = []
    score_layout = []
    if class_manifest is not None:
        from masterclass.engine.score_prep import select_score_pages_for_lesson
        try:
            score_pages, score_layout = select_score_pages_for_lesson(
                storage=storage,
                masterclass=class_manifest,
                first_measure=manifest.metadata.get("first_measure"),
                last_measure=manifest.metadata.get("last_measure"),
            )
        except Exception as exc:
            print(f"       (no score images: {exc})")

    teach_lesson(
        storage=storage,
        store=store,
        manifest=manifest,
        provider=provider,
        score_pages=score_pages,
        score_layout=score_layout,
        config=TeachConfig(),
    )
    print("       ok")
    return 0


def _read_past_lesson_full_inline(storage, store, lesson_manifest):
    for artifact_name in ("lesson/comments_enriched.json", "lesson/comments.json", "player/comments_enriched.json", "player/comments.json"):
        key = lesson_manifest.artifacts.get(artifact_name) or store.artifact_key(lesson_manifest.session, artifact_name)
        if not key or not storage.exists(key):
            continue
        try:
            payload = storage.read_json(key)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        comments_raw = payload.get("comments") or []
        comments = []
        if isinstance(comments_raw, list):
            for c in comments_raw:
                if not isinstance(c, dict):
                    continue
                comments.append({
                    "id": c.get("id"),
                    "start": c.get("start"),
                    "end": c.get("end"),
                    "measure": c.get("measure"),
                    "category": c.get("category"),
                    "severity": c.get("severity"),
                    "summary": c.get("summary") or c.get("title") or c.get("text"),
                    "text": c.get("text") or c.get("comment"),
                })
        return {"summary": payload.get("summary"), "progress_notes": payload.get("progress_notes"), "lesson": payload.get("lesson") or {}, "comments": comments}
    return {"summary": None, "progress_notes": None, "lesson": {}, "comments": []}


def _build_prior_lesson_context_inline(storage, store, mcs, class_manifest, current_session_id):
    from datetime import datetime, timezone
    from masterclass.core.models import TenantContext
    lessons = []
    for sid in (class_manifest.lessons or []):
        if sid == current_session_id:
            continue
        try:
            lesson = store.load_by_id(TenantContext(class_manifest.masterclass.tenant_id, class_manifest.masterclass.user_id), sid)
        except FileNotFoundError:
            continue
        full = _read_past_lesson_full_inline(storage, store, lesson)
        lessons.append({
            "session_id": sid,
            "created_at": lesson.created_at,
            "updated_at": lesson.updated_at,
            "state": lesson.state.value,
            "movement": lesson.movement,
            "notes": lesson.notes,
            "first_measure": lesson.metadata.get("first_measure") or lesson.metadata.get("auto_detected_first_measure"),
            "last_measure": lesson.metadata.get("last_measure") or lesson.metadata.get("auto_detected_last_measure"),
            "summary": full["summary"],
            "progress_notes": full["progress_notes"],
            "lesson": full["lesson"],
            "teacher_comments": full["comments"],
        })
    lessons.sort(key=lambda l: (l.get("created_at") or ""))
    return {
        "masterclass_id": class_manifest.masterclass.masterclass_id,
        "piece_name": class_manifest.piece_name,
        "movement": class_manifest.movement,
        "work_id": class_manifest.work_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lesson_count": len(lessons),
        "lessons": lessons,
    }


if __name__ == "__main__":
    raise SystemExit(main())
