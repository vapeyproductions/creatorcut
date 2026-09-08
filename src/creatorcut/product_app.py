"""Local CreatorCut product website with upload, ranking, export, and feedback APIs."""

from __future__ import annotations

import argparse
import cgi
import json
import logging
import mimetypes
import shutil
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from creatorcut.analytics import MAXIMUM_ANALYTICS_BYTES, parse_youtube_analytics_export
from creatorcut.annotation_app import parse_byte_range
from creatorcut.product_pipeline import ProductProcessor, validate_export_interval
from creatorcut.product_store import ProductStore
from creatorcut.product_worker import ProcessingWorker
from creatorcut.runtime_logging import configure_logging, log_event

STATIC_DIRECTORY = Path(__file__).with_name("static")
MAXIMUM_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}
LOGGER = logging.getLogger("creatorcut.web")


def validate_performance_report(value: Any) -> dict[str, Any]:
    """Validate optional post-publication metrics without inventing missing denominators."""
    if not isinstance(value, dict):
        raise ValueError("Performance report must be a JSON object")
    platform = value.get("platform")
    if platform not in {"tiktok", "instagram", "youtube", "other"}:
        raise ValueError("Choose a supported platform")
    normalized: dict[str, Any] = {"platform": platform}
    for field in ("views", "likes", "comments", "shares"):
        metric = value.get(field)
        if metric in (None, ""):
            normalized[field] = None
        elif isinstance(metric, bool) or not isinstance(metric, int) or metric < 0:
            raise ValueError(f"{field} must be a non-negative whole number")
        else:
            normalized[field] = metric
    average = value.get("average_view_percentage")
    if average in (None, ""):
        normalized["average_view_percentage"] = None
    elif (
        isinstance(average, bool)
        or not isinstance(average, int | float)
        or not 0 <= float(average) <= 100
    ):
        raise ValueError("average_view_percentage must be between 0 and 100")
    else:
        normalized["average_view_percentage"] = float(average)
    published_at = value.get("published_at")
    if published_at not in (None, "") and not isinstance(published_at, str):
        raise ValueError("published_at must be text")
    normalized["published_at"] = published_at or None
    if all(
        normalized[field] is None
        for field in ("views", "likes", "comments", "shares", "average_view_percentage")
    ):
        raise ValueError("Enter at least one performance metric")
    return normalized


class ProductApplication:
    """Coordinate durable product state, private media, and background ML inference."""

    def __init__(
        self,
        store: ProductStore,
        processor: ProductProcessor,
        upload_dir: Path,
        worker_mode: str = "external",
    ) -> None:
        self.store = store
        self.processor = processor
        self.upload_dir = upload_dir
        self.worker_mode = worker_mode
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def accept_upload(
        self,
        creator_name: str,
        creator_id: str | None,
        original_filename: str,
        source: Any,
        source_analytics_filename: str | None = None,
        source_analytics_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Stream an uploaded file to private local storage and queue inference."""
        suffix = Path(original_filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise ValueError("Upload an MP4, MOV, M4V, or WebM video")
        creator = self.store.ensure_creator(creator_name, creator_id)
        upload_key = uuid.uuid4().hex
        upload_path = self.upload_dir / f"{upload_key}{suffix}"
        with upload_path.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if upload_path.stat().st_size == 0:
            upload_path.unlink()
            raise ValueError("The uploaded video is empty")
        video = self.store.create_video(
            creator["id"], Path(original_filename).name, upload_path
        )
        if source_analytics_filename and source_analytics_report:
            self.store.save_analytics_import(
                creator["id"],
                source_analytics_filename,
                "source_video",
                source_analytics_report,
                video_id=video["id"],
            )
            video = self.store.get_video(video["id"])
        log_event(
            LOGGER,
            "video_upload_queued",
            video_id=video["id"],
            creator_id=creator["id"],
            original_filename=video["original_filename"],
        )
        return {"creator": creator, "video": video}

    def health(self) -> dict[str, Any]:
        """Return serving readiness plus persistent queue counters."""
        database_ready = self.store.database_ready()
        model_ready = self.processor.frozen_model_path.is_file()
        return {
            "status": "ok" if database_ready and model_ready else "not_ready",
            "service": "creatorcut-web",
            "database": "ok" if database_ready else "unavailable",
            "frozen_model": "ok" if model_ready else "missing",
            "worker_mode": self.worker_mode,
            "queue": self.store.processing_queue_summary() if database_ready else None,
        }


def create_handler(application: ProductApplication) -> type[BaseHTTPRequestHandler]:
    """Bind the CreatorCut product application to a standard-library HTTP handler."""

    class ProductHandler(BaseHTTPRequestHandler):
        server_version = "CreatorCutProduct/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                self._send_file(STATIC_DIRECTORY / "product.html")
                return
            if path == "/assets/product.css":
                self._send_file(STATIC_DIRECTORY / "product.css")
                return
            if path == "/assets/product.js":
                self._send_file(STATIC_DIRECTORY / "product.js")
                return
            if path == "/api/health/live":
                self._send_json({"status": "ok", "service": "creatorcut-web"})
                return
            if path in {"/api/health", "/api/health/ready"}:
                health = application.health()
                self._send_json(
                    health,
                    HTTPStatus.OK
                    if health["status"] == "ok"
                    else HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            if path.startswith("/api/videos/") and path.endswith("/source"):
                video_id = unquote(path[len("/api/videos/") : -len("/source")]).strip("/")
                try:
                    self._send_file(application.store.media_path_for_video(video_id), ranged=True)
                except (KeyError, FileNotFoundError):
                    self._send_json({"error": "Video not found"}, HTTPStatus.NOT_FOUND)
                return
            if path.startswith("/api/videos/"):
                video_id = unquote(path[len("/api/videos/") :]).strip("/")
                try:
                    video = application.store.get_video(video_id)
                    video["creator_summary"] = application.store.creator_summary(
                        video["creator_id"]
                    )
                    self._send_json(video)
                except KeyError:
                    self._send_json({"error": "Video not found"}, HTTPStatus.NOT_FOUND)
                return
            if path.startswith("/api/creators/") and path.endswith("/videos"):
                creator_id = unquote(
                    path[len("/api/creators/") : -len("/videos")]
                ).strip("/")
                self._send_json({"videos": application.store.list_videos(creator_id)})
                return
            if path.startswith("/api/creators/") and path.endswith("/model-report"):
                creator_id = unquote(
                    path[len("/api/creators/") : -len("/model-report")]
                ).strip("/")
                try:
                    self._send_json(application.store.creator_ml_report(creator_id))
                except KeyError:
                    self._send_json({"error": "Creator not found"}, HTTPStatus.NOT_FOUND)
                return
            if path.startswith("/api/exports/"):
                filename = Path(unquote(path[len("/api/exports/") :])).name
                export_path = application.processor.work_dir / "exports" / filename
                if not export_path.is_file():
                    self._send_json({"error": "Export not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._send_file(export_path, download_name=filename)
                return
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path == "/api/uploads":
                    self._handle_upload()
                    return
                if path.startswith("/api/clips/") and path.endswith("/decision"):
                    clip_id = unquote(path[len("/api/clips/") : -len("/decision")]).strip("/")
                    self._handle_decision(clip_id)
                    return
                if path.startswith("/api/clips/") and path.endswith("/export"):
                    clip_id = unquote(path[len("/api/clips/") : -len("/export")]).strip("/")
                    self._handle_export(clip_id)
                    return
                if path.startswith("/api/clips/") and path.endswith("/performance"):
                    clip_id = unquote(path[len("/api/clips/") : -len("/performance")]).strip("/")
                    self._handle_performance(clip_id)
                    return
                if path.startswith("/api/clips/") and path.endswith("/analytics"):
                    clip_id = unquote(path[len("/api/clips/") : -len("/analytics")]).strip("/")
                    self._handle_analytics_import("published_clip", clip_id)
                    return
                if path.startswith("/api/videos/") and path.endswith("/analytics"):
                    video_id = unquote(path[len("/api/videos/") : -len("/analytics")]).strip("/")
                    self._handle_analytics_import("source_video", video_id)
                    return
                if path.startswith("/api/videos/") and path.endswith("/custom-clips"):
                    video_id = unquote(
                        path[len("/api/videos/") : -len("/custom-clips")]
                    ).strip("/")
                    self._handle_custom_clip(video_id)
                    return
                if path.startswith("/api/videos/") and path.endswith("/review-complete"):
                    video_id = unquote(
                        path[len("/api/videos/") : -len("/review-complete")]
                    ).strip("/")
                    self._handle_review_complete(video_id)
                    return
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            except KeyError:
                self._send_json({"error": "Record not found"}, HTTPStatus.NOT_FOUND)
            except ValueError as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self._send_json(
                    {"error": f"The request failed: {str(error)[:300]}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _handle_upload(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAXIMUM_UPLOAD_BYTES:
                raise ValueError("Upload size must be between 1 byte and 5 GB")
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise ValueError("Upload request must use multipart form data")
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": str(content_length),
                },
                keep_blank_values=True,
            )
            video_field = form["video"] if "video" in form else None
            if video_field is None or not video_field.filename:
                raise ValueError("Choose a video to upload")
            analytics_filename = None
            analytics_report = None
            analytics_field = form["source_analytics"] if "source_analytics" in form else None
            if analytics_field is not None and analytics_field.filename:
                analytics_filename = analytics_field.filename
                payload = analytics_field.file.read(MAXIMUM_ANALYTICS_BYTES + 1)
                analytics_report = parse_youtube_analytics_export(
                    analytics_filename, payload
                )
            result = application.accept_upload(
                str(form.getfirst("creator_name", "Local creator")),
                form.getfirst("creator_id") or None,
                video_field.filename,
                video_field.file,
                analytics_filename,
                analytics_report,
            )
            self._send_json(result, HTTPStatus.ACCEPTED)

        def _handle_decision(self, clip_id: str) -> None:
            value = self._read_json()
            if value.get("action") != "reject":
                raise ValueError("The decision action must be reject")
            clip = application.store.get_clip(clip_id)
            application.store.record_editorial_event(
                clip_id,
                "reject",
                float(clip["start_seconds"]),
                float(clip["end_seconds"]),
                {"rank": clip["rank"]},
            )
            self._send_json({"saved": True})

        def _handle_export(self, clip_id: str) -> None:
            value = self._read_json()
            export_format = value.get("export_format", "original")
            clip = application.store.get_clip(clip_id)
            video = application.store.get_video(clip["video_id"])
            start, end = validate_export_interval(
                clip,
                value.get("start_seconds"),
                value.get("end_seconds"),
                float(video["duration_seconds"]),
            )
            export_path = application.processor.export_clip(
                clip_id, start, end, export_format
            )
            edited = (
                abs(start - float(clip["start_seconds"])) >= 0.05
                or abs(end - float(clip["end_seconds"])) >= 0.05
            )
            application.store.record_editorial_event(
                clip_id,
                "download_edited" if edited else "download_original",
                start,
                end,
                {
                    "rank": clip["rank"],
                    "start_adjustment_seconds": round(start - clip["start_seconds"], 3),
                    "end_adjustment_seconds": round(end - clip["end_seconds"], 3),
                    "export_format": export_format,
                },
            )
            self._send_json(
                {"download_url": f"/api/exports/{quote(export_path.name)}", "edited": edited}
            )

        def _handle_performance(self, clip_id: str) -> None:
            report = validate_performance_report(self._read_json())
            application.processor.ensure_clip_semantic_embedding(clip_id)
            application.store.save_performance_report(clip_id, report)
            self._send_json({"saved": True}, HTTPStatus.CREATED)

        def _handle_custom_clip(self, video_id: str) -> None:
            value = self._read_json()
            clip_id = application.processor.create_custom_clip(
                video_id,
                value.get("start_seconds"),
                value.get("end_seconds"),
            )
            video = application.store.get_video(video_id)
            video["creator_summary"] = application.store.creator_summary(
                video["creator_id"]
            )
            self._send_json(
                {"saved": True, "clip_id": clip_id, "video": video},
                HTTPStatus.CREATED,
            )

        def _handle_review_complete(self, video_id: str) -> None:
            self._read_json()
            unselected_count = application.store.complete_recommendation_review(
                video_id
            )
            video = application.store.get_video(video_id)
            video["creator_summary"] = application.store.creator_summary(
                video["creator_id"]
            )
            self._send_json(
                {"saved": True, "unselected_count": unselected_count, "video": video},
                HTTPStatus.CREATED,
            )

        def _handle_analytics_import(self, report_role: str, target_id: str) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAXIMUM_ANALYTICS_BYTES + 65_536:
                raise ValueError("Analytics uploads must be 20 MB or smaller")
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise ValueError("Analytics upload must use multipart form data")
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": str(content_length),
                },
                keep_blank_values=True,
            )
            analytics_field = form["analytics"] if "analytics" in form else None
            if analytics_field is None or not analytics_field.filename:
                raise ValueError("Choose a YouTube Studio ZIP or CSV export")
            payload = analytics_field.file.read(MAXIMUM_ANALYTICS_BYTES + 1)
            report = parse_youtube_analytics_export(analytics_field.filename, payload)
            if report_role == "published_clip":
                clip = application.store.get_clip(target_id)
                creator_id = clip["creator_id"]
                application.processor.ensure_clip_semantic_embedding(target_id)
                saved = application.store.save_analytics_import(
                    creator_id,
                    analytics_field.filename,
                    report_role,
                    report,
                    clip_id=target_id,
                )
            else:
                video = application.store.get_video(target_id)
                saved = application.store.save_analytics_import(
                    video["creator_id"],
                    analytics_field.filename,
                    report_role,
                    report,
                    video_id=target_id,
                )
            self._send_json({"saved": True, "import": saved}, HTTPStatus.CREATED)

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 65_536:
                raise ValueError("Invalid request size")
            value = json.loads(self.rfile.read(content_length))
            if not isinstance(value, dict):
                raise ValueError("Request body must be a JSON object")
            return value

        def _send_json(
            self, value: Any, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _send_file(
            self,
            path: Path,
            ranged: bool = False,
            download_name: str | None = None,
        ) -> None:
            if not path.is_file():
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            size = path.stat().st_size
            byte_range = None
            if ranged:
                try:
                    byte_range = parse_byte_range(self.headers.get("Range"), size)
                except (ValueError, TypeError):
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            start, end = byte_range or (0, size - 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT if byte_range else HTTPStatus.OK)
            self.send_header(
                "Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            )
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            if byte_range:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if download_name:
                self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def log_message(self, format: str, *args: Any) -> None:
            log_event(
                LOGGER,
                "http_request",
                client=self.client_address[0],
                request=self.requestline,
                response=format % args,
            )

    return ProductHandler


def main() -> None:
    """Start the local CreatorCut product website."""
    parser = argparse.ArgumentParser(description="Run the CreatorCut product website")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--database", type=Path, default=Path("data/product/creatorcut.sqlite"))
    parser.add_argument("--upload-dir", type=Path, default=Path("data/product/uploads"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/product/work"))
    parser.add_argument(
        "--frozen-model", type=Path, default=Path("models/frozen_model_v1.json")
    )
    parser.add_argument("--model-cache", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--semantic-cache", type=Path, default=Path("artifacts/huggingface"))
    parser.add_argument(
        "--worker-mode",
        choices=("embedded", "external"),
        default="embedded",
        help="Embedded keeps local use one-command; external is for separate web/worker services.",
    )
    parser.add_argument("--worker-id")
    parser.add_argument("--lease-seconds", type=float, default=120.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    store = ProductStore(args.database)
    processor = ProductProcessor(
        store,
        args.work_dir,
        args.frozen_model,
        args.model_cache,
        args.semantic_cache,
    )
    application = ProductApplication(store, processor, args.upload_dir, args.worker_mode)
    worker_stop = threading.Event()
    worker_thread = None
    if args.worker_mode == "embedded":
        worker = ProcessingWorker(
            store,
            processor,
            worker_id=args.worker_id,
            lease_seconds=args.lease_seconds,
            poll_seconds=args.poll_seconds,
        )
        worker_thread = threading.Thread(
            target=worker.run_forever,
            args=(worker_stop,),
            name="creatorcut-embedded-worker",
            daemon=True,
        )
        worker_thread.start()
    server = ThreadingHTTPServer((args.host, args.port), create_handler(application))
    log_event(
        LOGGER,
        "web_server_started",
        address=f"http://{args.host}:{args.port}",
        worker_mode=args.worker_mode,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker_stop.set()
        if worker_thread is not None:
            worker_thread.join(timeout=5)


if __name__ == "__main__":
    main()
