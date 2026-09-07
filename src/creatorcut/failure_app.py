"""Local interface for diagnosing out-of-fold top-clip ranking failures."""

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

from creatorcut.annotation_app import parse_byte_range
from creatorcut.dataset import load_jsonl, load_manifest

FAILURE_REASONS = (
    "missing_start_context",
    "incomplete_ending",
    "advertisement_or_sponsor",
    "music_or_intro",
    "too_long",
    "low_energy",
    "weak_hook",
    "weak_payoff",
    "unclear_without_context",
    "other",
)
PREFERENCE_CHOICES = ("model_selected", "human_best", "tie", "neither")
PREFERENCE_CONTEXT = "after_proposed_model_edits"
STATIC_DIRECTORY = Path(__file__).with_name("static")


def validate_failure_review(value: Any) -> dict[str, Any]:
    """Validate the structured diagnosis for one top-selection failure."""
    if not isinstance(value, dict):
        raise ValueError("Failure review payload must be a JSON object")
    reasons = value.get("reasons")
    if not isinstance(reasons, list) or not reasons:
        raise ValueError("Select at least one failure reason")
    if any(not isinstance(reason, str) or reason not in FAILURE_REASONS for reason in reasons):
        raise ValueError("Failure review contains an unknown reason")
    if len(set(reasons)) != len(reasons):
        raise ValueError("Failure reasons must be unique")

    boundary_fixable = value.get("boundary_fixable")
    if not isinstance(boundary_fixable, bool):
        raise ValueError("boundary_fixable must be true or false")

    preferred_clip = value.get("preferred_clip")
    if preferred_clip not in PREFERENCE_CHOICES:
        raise ValueError("preferred_clip must identify one comparison choice")

    adjustments: dict[str, float | None] = {}
    for field in ("start_adjustment_seconds", "end_adjustment_seconds"):
        adjustment = value.get(field)
        if adjustment is None or adjustment == "":
            adjustments[field] = None
            continue
        if isinstance(adjustment, bool) or not isinstance(adjustment, int | float):
            raise ValueError(f"{field} must be numeric or empty")
        adjustment = float(adjustment)
        if not -30.0 <= adjustment <= 30.0:
            raise ValueError(f"{field} must be between -30 and 30 seconds")
        adjustments[field] = adjustment

    notes = value.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("notes must be text")
    notes = notes.strip()
    if len(notes) > 1000:
        raise ValueError("notes must be 1000 characters or fewer")
    return {
        "reasons": reasons,
        "preferred_clip": preferred_clip,
        "preference_context": PREFERENCE_CONTEXT,
        "boundary_fixable": boundary_fixable,
        **adjustments,
        "notes": notes,
    }


class FailureReviewStore:
    """Persist structured failure diagnoses separately from original ratings."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        if path.exists():
            self._records = {record["analysis_id"]: record for record in load_jsonl(path)}

    @property
    def completed_ids(self) -> set[str]:
        with self._lock:
            return {
                analysis_id
                for analysis_id, record in self._records.items()
                if record.get("preferred_clip") in PREFERENCE_CHOICES
            }

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        """Return an existing diagnosis so a new preference pass can preserve it."""
        with self._lock:
            record = self._records.get(analysis_id)
            return dict(record) if record else None

    def save(self, case: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
        record = {
            "analysis_id": case["analysis_id"],
            "video_id": case["video_id"],
            "model_annotation_id": case["model_selected"]["annotation_id"],
            "human_best_annotation_id": case["human_best"]["annotation_id"],
            "regret": case["regret"],
            **diagnosis,
            "reviewed_at": datetime.now(UTC).isoformat(),
        }
        with self._lock:
            self._records[record["analysis_id"]] = record
            self._write()
        return record

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for analysis_id in sorted(self._records):
                stream.write(json.dumps(self._records[analysis_id], ensure_ascii=False) + "\n")
        temporary.replace(self.path)


class FailureAnalysisApplication:
    """Coordinate private failure cases, media access, and saved diagnoses."""

    def __init__(
        self,
        cases: list[dict[str, Any]],
        manifest: list[dict[str, Any]],
        store: FailureReviewStore,
        media_dir: Path,
    ) -> None:
        self.cases = cases
        self.case_by_id = {case["analysis_id"]: case for case in cases}
        if len(self.case_by_id) != len(cases):
            raise ValueError("Failure cases contain duplicate analysis IDs")
        self.store = store
        self.video_paths = {
            item["video_id"]: media_dir / Path(item["local_filename"]).name
            for item in manifest
        }

    def next_cases(
        self, limit: int = 1, include_completed: bool = False
    ) -> list[dict[str, Any]]:
        completed = self.store.completed_ids
        pending = (
            self.cases
            if include_completed
            else [case for case in self.cases if case["analysis_id"] not in completed]
        )
        return [
            {**case, "existing_review": self.store.get(case["analysis_id"])}
            for case in pending[: max(1, min(limit, 20))]
        ]

    def stats(self) -> dict[str, int]:
        completed = len(self.store.completed_ids & self.case_by_id.keys())
        return {
            "total": len(self.cases),
            "completed": completed,
            "remaining": len(self.cases) - completed,
        }

    def save_review(self, analysis_id: str, value: Any) -> dict[str, Any]:
        case = self.case_by_id.get(analysis_id)
        if case is None:
            raise KeyError(analysis_id)
        diagnosis = validate_failure_review(value)
        selected = case["model_selected"]
        adjusted_start = float(selected["start_seconds"]) + (
            diagnosis["start_adjustment_seconds"] or 0.0
        )
        adjusted_end = float(selected["end_seconds"]) + (
            diagnosis["end_adjustment_seconds"] or 0.0
        )
        if adjusted_start < 0 or adjusted_end <= adjusted_start:
            raise ValueError("Adjusted clip boundaries must define a positive interval")
        return self.store.save(case, diagnosis)

    def video_path(self, video_id: str) -> Path:
        path = self.video_paths.get(video_id)
        if path is None:
            raise KeyError(video_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


def create_handler(application: FailureAnalysisApplication) -> type[BaseHTTPRequestHandler]:
    """Bind a failure-analysis application to a local HTTP handler."""

    class FailureHandler(BaseHTTPRequestHandler):
        server_version = "CreatorCutFailureAnalysis/1.0"

        def do_GET(self) -> None:  # noqa: N802
            self._handle_get(send_body=True)

        def do_HEAD(self) -> None:  # noqa: N802
            self._handle_get(send_body=False)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            prefix = "/api/failure-reviews/"
            if not parsed.path.startswith(prefix):
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            analysis_id = unquote(parsed.path[len(prefix) :])
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 32_768:
                    raise ValueError("Invalid request size")
                record = application.save_review(
                    analysis_id, json.loads(self.rfile.read(content_length))
                )
            except KeyError:
                self._send_json({"error": "Unknown failure case"}, HTTPStatus.NOT_FOUND)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            else:
                self._send_json(
                    {"saved": True, "analysis_id": record["analysis_id"]},
                    HTTPStatus.CREATED,
                )

        def _handle_get(self, send_body: bool) -> None:
            parsed = urlparse(self.path)
            static_routes = {
                "/": "failure.html",
                "/assets/failure.css": "failure.css",
                "/assets/failure.js": "failure.js",
            }
            if parsed.path in static_routes:
                self._send_static(static_routes[parsed.path], send_body)
                return
            if parsed.path == "/api/cases":
                values = parse_qs(parsed.query)
                try:
                    limit = int(values.get("limit", ["1"])[0])
                except ValueError:
                    self._send_json({"error": "limit must be an integer"}, HTTPStatus.BAD_REQUEST)
                    return
                include_completed = values.get("include_completed", ["false"])[0].lower() in {
                    "1",
                    "true",
                    "yes",
                }
                self._send_json(
                    {
                        "cases": application.next_cases(limit, include_completed),
                        "stats": application.stats(),
                    },
                    send_body=send_body,
                )
                return
            if parsed.path == "/api/stats":
                self._send_json(application.stats(), send_body=send_body)
                return
            if parsed.path.startswith("/api/video/"):
                self._send_video(
                    unquote(parsed.path.removeprefix("/api/video/")), send_body
                )
                return
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND, send_body)

        def _send_static(self, filename: str, send_body: bool) -> None:
            path = STATIC_DIRECTORY / filename
            if not path.is_file():
                self._send_json({"error": "Asset not found"}, HTTPStatus.NOT_FOUND, send_body)
                return
            content = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            content_type = mimetypes.guess_type(path.name)[0] or "text/plain"
            self.send_header(
                "Content-Type", f"{content_type}; charset=utf-8"
            )
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
                self._send_json({"error": "Video unavailable"}, HTTPStatus.NOT_FOUND, send_body)
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
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
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

    return FailureHandler


def main() -> None:
    """Run the private failure-analysis review interface."""
    parser = argparse.ArgumentParser(description="Review CreatorCut ranking failures locally")
    parser.add_argument(
        "--cases", type=Path, default=Path("data/processed/failure_cases.jsonl")
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/videos.json"))
    parser.add_argument("--media-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/failure_reviews.jsonl")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    application = FailureAnalysisApplication(
        load_jsonl(args.cases),
        load_manifest(args.manifest),
        FailureReviewStore(args.output),
        args.media_dir,
    )
    server = ThreadingHTTPServer((args.host, args.port), create_handler(application))
    print(f"CreatorCut Failure Analysis: http://{args.host}:{args.port}", flush=True)
    print(f"Diagnoses save locally to {args.output}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
