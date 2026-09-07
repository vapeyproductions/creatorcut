"""Local web application for blind human review of sampled clip candidates."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from creatorcut.dataset import load_jsonl, load_manifest

SCORE_FIELDS = ("hook", "completeness", "payoff", "clarity")
PUBLIC_TASK_FIELDS = (
    "annotation_id",
    "candidate_id",
    "video_id",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "transcript_text",
)
STATIC_DIRECTORY = Path(__file__).with_name("static")


def validate_labels(value: Any) -> dict[str, Any]:
    """Validate and normalize one review submitted by the browser."""
    if not isinstance(value, dict):
        raise ValueError("Review payload must be a JSON object")

    labels: dict[str, Any] = {}
    for field in SCORE_FIELDS:
        score = value.get(field)
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"{field} must be an integer from 1 to 5")
        labels[field] = score

    technically_exportable = value.get("technically_exportable")
    if not isinstance(technically_exportable, bool):
        raise ValueError("technically_exportable must be true or false")
    labels["technically_exportable"] = technically_exportable

    notes = value.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("notes must be text")
    notes = notes.strip()
    if len(notes) > 1000:
        raise ValueError("notes must be 1000 characters or fewer")
    labels["notes"] = notes
    return labels


def parse_byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    """Parse a single HTTP byte range into inclusive start and end offsets."""
    if not value:
        return None
    if not value.startswith("bytes=") or "," in value:
        raise ValueError("Unsupported byte range")

    start_text, separator, end_text = value[6:].partition("-")
    if not separator:
        raise ValueError("Invalid byte range")
    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("Invalid byte range")
        return max(0, size - suffix_length), size - 1

    start = int(start_text)
    end = int(end_text) if end_text else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("Byte range is outside the file")
    return start, min(end, size - 1)


class ReviewStore:
    """Persist review records atomically as a compact JSONL dataset."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        if path.exists():
            self._records = {
                record["annotation_id"]: record for record in load_jsonl(path)
            }

    @property
    def completed_ids(self) -> set[str]:
        with self._lock:
            return set(self._records)

    def save(self, task: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
        """Create or replace one review and atomically rewrite the local dataset."""
        record = {
            "annotation_id": task["annotation_id"],
            "candidate_id": task["candidate_id"],
            "video_id": task["video_id"],
            "start_seconds": task["start_seconds"],
            "end_seconds": task["end_seconds"],
            "duration_seconds": task["duration_seconds"],
            **labels,
            "presentation": 5,
            "reviewed_at": datetime.now(UTC).isoformat(),
        }
        with self._lock:
            self._records[record["annotation_id"]] = record
            self._write()
        return record

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            for annotation_id in sorted(self._records):
                stream.write(json.dumps(self._records[annotation_id], ensure_ascii=False) + "\n")
        temporary_path.replace(self.path)


class AnnotationApplication:
    """Coordinate private tasks, public task payloads, media, and saved reviews."""

    def __init__(
        self,
        queue: list[dict[str, Any]],
        manifest: list[dict[str, Any]],
        store: ReviewStore,
        media_dir: Path,
        batch_size: int = 10,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.queue = queue
        self.tasks = {task["annotation_id"]: task for task in queue}
        if len(self.tasks) != len(queue):
            raise ValueError("Annotation queue contains duplicate annotation IDs")
        self.store = store
        self.batch_size = batch_size
        self.video_paths = {
            item["video_id"]: media_dir / Path(item["local_filename"]).name
            for item in manifest
        }

    def public_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Return review fields without heuristic scores or score-band metadata."""
        return {field: task[field] for field in PUBLIC_TASK_FIELDS}

    def next_tasks(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return the next unfinished batch in stable queue order."""
        completed = self.store.completed_ids
        requested = self.batch_size if limit is None else max(1, min(limit, 50))
        pending = [task for task in self.queue if task["annotation_id"] not in completed]
        return [self.public_task(task) for task in pending[:requested]]

    def stats(self) -> dict[str, int]:
        """Return overall queue progress."""
        completed = len(self.store.completed_ids & self.tasks.keys())
        return {
            "total": len(self.queue),
            "completed": completed,
            "remaining": len(self.queue) - completed,
        }

    def save_review(self, annotation_id: str, value: Any) -> dict[str, Any]:
        """Validate a submitted review against its server-owned task."""
        task = self.tasks.get(annotation_id)
        if task is None:
            raise KeyError(annotation_id)
        return self.store.save(task, validate_labels(value))

    def video_path(self, video_id: str) -> Path:
        """Resolve only manifest-declared media within the configured directory."""
        path = self.video_paths.get(video_id)
        if path is None:
            raise KeyError(video_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


def create_handler(application: AnnotationApplication) -> type[BaseHTTPRequestHandler]:
    """Bind an application instance to a standard-library request handler."""

    class AnnotationHandler(BaseHTTPRequestHandler):
        server_version = "CreatorCutAnnotation/1.0"

        def do_GET(self) -> None:  # noqa: N802
            self._handle_get(send_body=True)

        def do_HEAD(self) -> None:  # noqa: N802
            self._handle_get(send_body=False)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            prefix = "/api/reviews/"
            if not parsed.path.startswith(prefix):
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return

            annotation_id = unquote(parsed.path[len(prefix) :])
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 32_768:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(content_length))
                record = application.save_review(annotation_id, payload)
            except KeyError:
                self._send_json({"error": "Unknown annotation task"}, HTTPStatus.NOT_FOUND)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            else:
                self._send_json(
                    {"saved": True, "annotation_id": record["annotation_id"]},
                    HTTPStatus.CREATED,
                )

        def _handle_get(self, send_body: bool) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_static("annotate.html", send_body)
                return
            if parsed.path == "/assets/styles.css":
                self._send_static("styles.css", send_body)
                return
            if parsed.path == "/assets/app.js":
                self._send_static("app.js", send_body)
                return
            if parsed.path == "/api/tasks":
                values = parse_qs(parsed.query)
                try:
                    limit = int(values["limit"][0]) if "limit" in values else None
                except ValueError:
                    self._send_json({"error": "limit must be an integer"}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(
                    {"tasks": application.next_tasks(limit), "stats": application.stats()},
                    send_body=send_body,
                )
                return
            if parsed.path == "/api/stats":
                self._send_json(application.stats(), send_body=send_body)
                return
            if parsed.path.startswith("/api/video/"):
                video_id = unquote(parsed.path.removeprefix("/api/video/"))
                self._send_video(video_id, send_body)
                return
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND, send_body)

        def _send_static(self, filename: str, send_body: bool) -> None:
            path = STATIC_DIRECTORY / filename
            if not path.is_file():
                self._send_json({"error": "Asset not found"}, HTTPStatus.NOT_FOUND, send_body)
                return
            content = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if send_body:
                self.wfile.write(content)

        def _send_video(self, video_id: str, send_body: bool) -> None:
            try:
                path = application.video_path(video_id)
                size = path.stat().st_size
                byte_range = parse_byte_range(self.headers.get("Range"), size)
            except KeyError:
                self._send_json({"error": "Unknown video"}, HTTPStatus.NOT_FOUND, send_body)
                return
            except FileNotFoundError:
                self._send_json(
                    {"error": "Video file is unavailable"}, HTTPStatus.NOT_FOUND, send_body
                )
                return
            except (ValueError, OverflowError):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return

            start, end = byte_range or (0, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT if byte_range else HTTPStatus.OK
            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if byte_range:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not send_body:
                return

            remaining = end - start + 1
            with path.open("rb") as stream:
                stream.seek(start)
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _send_json(
            self,
            value: Any,
            status: HTTPStatus = HTTPStatus.OK,
            send_body: bool = True,
        ) -> None:
            content = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if send_body:
                self.wfile.write(content)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

    return AnnotationHandler


def main() -> None:
    """Run the local CreatorCut annotation studio."""
    parser = argparse.ArgumentParser(description="Review sampled CreatorCut clips locally")
    parser.add_argument(
        "--queue", type=Path, default=Path("data/processed/annotation_queue.jsonl")
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/videos.json"))
    parser.add_argument("--media-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/annotation_reviews.jsonl")
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    application = AnnotationApplication(
        load_jsonl(args.queue),
        load_manifest(args.manifest),
        ReviewStore(args.output),
        args.media_dir,
        batch_size=args.batch_size,
    )
    server = ThreadingHTTPServer((args.host, args.port), create_handler(application))
    print(f"CreatorCut Annotation Studio: http://{args.host}:{args.port}", flush=True)
    print(f"Reviews save locally to {args.output}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
