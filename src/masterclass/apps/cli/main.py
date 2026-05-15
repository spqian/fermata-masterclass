from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import socketserver
from pathlib import Path

from masterclass.agent.dry_run import DryRunLlmProvider
from masterclass.agent.gemini import SharedKeyGeminiProvider
from masterclass.agent.teacher import TeacherAgent
from masterclass.agent_tools.registry import default_tool_registry
from masterclass.core.models import TenantContext
from masterclass.core.sessions import SessionStore
from masterclass.engine.analysis import analyze_session, build_evidence_packet
from masterclass.engine.ingest import IngestRequest, extract_media_artifacts, ingest_video
from masterclass.storage.local import LocalObjectStorage
from masterclass.toolchain.ffmpeg import FfmpegToolchain


class ThreadingHttpServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler with byte-range support for seekable local videos."""

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        if not os.path.exists(path):
            self.send_error(404, "File not found")
            return None
        ctype = self.guess_type(path)
        try:
            file_obj = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        file_size = os.fstat(file_obj.fileno()).st_size
        range_header = self.headers.get("Range")
        if not range_header:
            self.send_response(200)
            self.send_header("Content-type", ctype)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", self.date_time_string(os.path.getmtime(path)))
            self.end_headers()
            return file_obj

        byte_range = self._parse_range(range_header, file_size)
        if byte_range is None:
            file_obj.close()
            self.send_error(416, "Requested Range Not Satisfiable")
            return None
        start, end = byte_range
        length = end - start + 1
        file_obj.seek(start)
        self.send_response(206)
        self.send_header("Content-type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", self.date_time_string(os.path.getmtime(path)))
        self.end_headers()
        self.range = (start, end)
        return file_obj

    def copyfile(self, source, outputfile):
        byte_range = getattr(self, "range", None)
        if byte_range is None:
            return super().copyfile(source, outputfile)
        start, end = byte_range
        remaining = end - start + 1
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        path_lower = self.path.lower()
        if path_lower == "/" or any(path_lower.split("?", 1)[0].endswith(ext) for ext in (".html", ".json", ".js", ".css")):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    @staticmethod
    def _parse_range(header: str, file_size: int) -> tuple[int, int] | None:
        if not header.startswith("bytes="):
            return None
        spec = header[6:].split(",", 1)[0].strip()
        if "-" not in spec:
            return None
        left, right = spec.split("-", 1)
        try:
            if left == "":
                suffix = int(right)
                if suffix <= 0:
                    return None
                start = max(0, file_size - suffix)
                end = file_size - 1
            else:
                start = int(left)
                end = int(right) if right else file_size - 1
        except ValueError:
            return None
        if start < 0 or end < start or start >= file_size:
            return None
        return start, min(end, file_size - 1)


def _store(args: argparse.Namespace) -> SessionStore:
    return SessionStore(LocalObjectStorage(Path(args.storage_root)))


def command_create_session(args: argparse.Namespace) -> None:
    ctx = TenantContext(tenant_id=args.user_id, user_id=args.user_id)
    manifest = _store(args).create(
        ctx,
        source_filename=args.source_filename,
        repertoire=args.repertoire,
        movement=args.movement,
        instrument=args.instrument,
        instrument_profile=args.instrument_profile,
        notes=args.notes,
    )
    print(json.dumps(manifest.to_json(), indent=2))


def command_ingest(args: argparse.Namespace) -> None:
    storage = LocalObjectStorage(Path(args.storage_root))
    store = SessionStore(storage)
    ffmpeg = FfmpegToolchain.discover(args.ffmpeg, args.ffprobe)
    ctx = TenantContext(tenant_id=args.user_id, user_id=args.user_id)
    manifest = ingest_video(
        store=store,
        storage=storage,
        ffmpeg=ffmpeg,
        request=IngestRequest(
            tenant=ctx,
            video_path=Path(args.video),
            repertoire=args.repertoire,
            movement=args.movement,
            instrument=args.instrument,
            instrument_profile=args.instrument_profile,
            notes=args.notes,
            frame_interval_sec=args.frame_interval,
        ),
    )
    print(json.dumps(manifest.to_json(), indent=2))


def command_list_sessions(args: argparse.Namespace) -> None:
    ctx = TenantContext(tenant_id=args.user_id, user_id=args.user_id)
    manifests = _store(args).list_for_user(ctx)
    print(json.dumps([m.to_json() for m in manifests], indent=2))


def command_show_session(args: argparse.Namespace) -> None:
    ctx = TenantContext(tenant_id=args.user_id, user_id=args.user_id)
    manifest = _store(args).load_by_id(ctx, args.session_id)
    print(json.dumps(manifest.to_json(), indent=2))


def command_extract_media(args: argparse.Namespace) -> None:
    storage = LocalObjectStorage(Path(args.storage_root))
    store = SessionStore(storage)
    ffmpeg = FfmpegToolchain.discover(args.ffmpeg, args.ffprobe)
    ctx = TenantContext(tenant_id=args.user_id, user_id=args.user_id)
    manifest = store.load_by_id(ctx, args.session_id)
    manifest = extract_media_artifacts(
        store=store,
        storage=storage,
        ffmpeg=ffmpeg,
        manifest=manifest,
        frame_interval_sec=args.frame_interval,
    )
    print(json.dumps(manifest.to_json(), indent=2))


def command_analyze(args: argparse.Namespace) -> None:
    storage = LocalObjectStorage(Path(args.storage_root))
    store = SessionStore(storage)
    ctx = TenantContext(tenant_id=args.user_id, user_id=args.user_id)
    manifest = store.load_by_id(ctx, args.session_id)
    manifest = analyze_session(store=store, storage=storage, manifest=manifest)
    print(json.dumps(manifest.to_json(), indent=2))


def command_evidence_packet(args: argparse.Namespace) -> None:
    storage = LocalObjectStorage(Path(args.storage_root))
    store = SessionStore(storage)
    ctx = TenantContext(tenant_id=args.user_id, user_id=args.user_id)
    manifest = store.load_by_id(ctx, args.session_id)
    manifest = build_evidence_packet(store=store, storage=storage, manifest=manifest)
    print(json.dumps(manifest.to_json(), indent=2))


def command_teach(args: argparse.Namespace) -> None:
    storage = LocalObjectStorage(Path(args.storage_root))
    store = SessionStore(storage)
    ctx = TenantContext(tenant_id=args.user_id, user_id=args.user_id)
    manifest = store.load_by_id(ctx, args.session_id)
    provider = DryRunLlmProvider() if args.dry_run else SharedKeyGeminiProvider()
    agent = TeacherAgent(provider=provider, tools=default_tool_registry(manifest.instrument_profile), storage=storage)
    manifest = agent.teach(store, manifest, model=args.model, max_tool_calls=args.max_tool_calls)
    print(json.dumps(manifest.to_json(), indent=2))


def _poc_project(args: argparse.Namespace) -> None:
    raise SystemExit("poc bridge has been removed; use the native v2 pipeline instead")


def command_serve_player(args: argparse.Namespace) -> None:
    storage = LocalObjectStorage(Path(args.storage_root))
    store = SessionStore(storage)
    ctx = TenantContext(tenant_id=args.user_id, user_id=args.user_id)
    manifest = store.load_by_id(ctx, args.session_id)
    index_key = manifest.artifacts.get("player/index.html")
    if not index_key:
        raise SystemExit("player/index.html artifact missing for this session")
    player_dir = storage.resolve_local_path(index_key).parent
    handler = functools.partial(RangeRequestHandler, directory=str(player_dir))
    with ThreadingHttpServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"serving v2 player: http://127.0.0.1:{args.port}/")
        print(f"player directory: {player_dir}")
        httpd.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="masterclass-v2")
    parser.add_argument("--storage-root", default="local_adls", help="Local ADLS-shaped storage root.")
    parser.add_argument("--ffmpeg", help="Path to ffmpeg executable.")
    parser.add_argument("--ffprobe", help="Path to ffprobe executable.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-session", help="Create an ADLS-shaped individual-user session manifest.")
    p.add_argument("--user-id", required=True)
    p.add_argument("--source-filename")
    p.add_argument("--repertoire")
    p.add_argument("--movement")
    p.add_argument("--instrument")
    p.add_argument("--instrument-profile")
    p.add_argument("--notes")
    p.set_defaults(func=command_create_session)

    p = sub.add_parser("ingest", help="Create a session and ingest an uploaded/local MP4 into ADLS-shaped storage.")
    p.add_argument("video")
    p.add_argument("--user-id", required=True)
    p.add_argument("--repertoire")
    p.add_argument("--movement")
    p.add_argument("--instrument")
    p.add_argument("--instrument-profile")
    p.add_argument("--notes")
    p.add_argument("--frame-interval", type=float, default=10.0)
    p.set_defaults(func=command_ingest)

    p = sub.add_parser("list-sessions", help="List this individual user's ADLS-backed session manifests.")
    p.add_argument("--user-id", required=True)
    p.set_defaults(func=command_list_sessions)

    p = sub.add_parser("show-session", help="Read one session manifest by user/session id.")
    p.add_argument("--user-id", required=True)
    p.add_argument("session_id")
    p.set_defaults(func=command_show_session)

    p = sub.add_parser("extract-media", help="Worker-style extraction from an already-uploaded source_video artifact.")
    p.add_argument("--user-id", required=True)
    p.add_argument("session_id")
    p.add_argument("--frame-interval", type=float, default=10.0)
    p.set_defaults(func=command_extract_media)

    p = sub.add_parser("analyze", help="Run deterministic audio analysis against ingested ADLS-shaped artifacts.")
    p.add_argument("--user-id", required=True)
    p.add_argument("session_id")
    p.set_defaults(func=command_analyze)

    p = sub.add_parser("evidence-packet", help="Build a storage-backed evidence packet for the teacher agent.")
    p.add_argument("--user-id", required=True)
    p.add_argument("session_id")
    p.set_defaults(func=command_evidence_packet)

    p = sub.add_parser("teach", help="Run the v2 teacher agent through the storage-backed LLM/tool seam.")
    p.add_argument("--user-id", required=True)
    p.add_argument("session_id")
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--max-tool-calls", type=int, default=15)
    p.add_argument("--dry-run", action="store_true", help="Validate the agent seam without calling a paid LLM.")
    p.set_defaults(func=command_teach)

    p = sub.add_parser("serve-player", help="Serve a session's player artifacts from local v2 storage.")
    p.add_argument("--user-id", required=True)
    p.add_argument("session_id")
    p.add_argument("--port", type=int, default=8766)
    p.set_defaults(func=command_serve_player)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
