"""Local CreatorCut product website with upload, ranking, export, and feedback APIs."""

from __future__ import annotations

import argparse
import cgi
import hmac
import json
import logging
import mimetypes
import shutil
import threading
import time
import uuid
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from creatorcut.analytics import MAXIMUM_ANALYTICS_BYTES, parse_youtube_analytics_export
from creatorcut.annotation_app import parse_byte_range
from creatorcut.feedback_export import build_feedback_snapshot
from creatorcut.product_pipeline import ProductProcessor, validate_export_interval
from creatorcut.product_store import ProductStore
from creatorcut.product_worker import ProcessingWorker
from creatorcut.runtime_logging import configure_logging, log_event

STATIC_DIRECTORY = Path(__file__).with_name("static")
MAXIMUM_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024
MAXIMUM_REPURPOSING_TEXT_BYTES = 8_000
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}
LOGGER = logging.getLogger("creatorcut.web")
SESSION_COOKIE_NAME = "creatorcut_session"
SESSION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
LOGIN_FAILURE_LIMIT = 5
LOGIN_FAILURE_WINDOW_SECONDS = 15 * 60


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
        admin_dashboard_enabled: bool = False,
        secure_cookies: bool = False,
        allow_first_admin_registration: bool = False,
    ) -> None:
        self.store = store
        self.processor = processor
        self.upload_dir = upload_dir
        self.worker_mode = worker_mode
        self.admin_dashboard_enabled = admin_dashboard_enabled
        self.secure_cookies = secure_cookies
        self.allow_first_admin_registration = allow_first_admin_registration
        self._login_lock = threading.Lock()
        self._login_failures: dict[str, list[float]] = {}
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def login_allowed(self, key: str) -> bool:
        """Bound password-verification work per source and login identifier."""
        cutoff = time.monotonic() - LOGIN_FAILURE_WINDOW_SECONDS
        with self._login_lock:
            recent = [value for value in self._login_failures.get(key, []) if value > cutoff]
            self._login_failures[key] = recent
            return len(recent) < LOGIN_FAILURE_LIMIT

    def record_login_failure(self, key: str) -> None:
        with self._login_lock:
            self._login_failures.setdefault(key, []).append(time.monotonic())

    def clear_login_failures(self, key: str) -> None:
        with self._login_lock:
            self._login_failures.pop(key, None)

    def accept_upload(
        self,
        creator: dict[str, Any],
        original_filename: str,
        source: Any,
        source_analytics_filename: str | None = None,
        source_analytics_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Stream an uploaded file to private local storage and queue inference."""
        suffix = Path(original_filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise ValueError("Upload an MP4, MOV, M4V, or WebM video")
        upload_key = uuid.uuid4().hex
        upload_path = self.upload_dir / f"{upload_key}{suffix}"
        with upload_path.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if upload_path.stat().st_size == 0:
            upload_path.unlink()
            raise ValueError("The uploaded video is empty")
        creator_id = creator["creator_id"]
        video = self.store.create_video(
            creator_id, Path(original_filename).name, upload_path
        )
        if source_analytics_filename and source_analytics_report:
            self.store.save_analytics_import(
                creator_id,
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
            creator_id=creator_id,
            original_filename=video["original_filename"],
        )
        return {"video": video}

    def health(self) -> dict[str, Any]:
        """Return serving readiness plus persistent queue counters."""
        database_ready = self.store.database_ready()
        model_ready = self.processor.frozen_model_path.is_file()
        try:
            release = self.processor.serving_release_status()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            release = {"status": "invalid", "error": str(error)[:500]}
        release_ready = release["status"] in {
            "frozen",
            "unmanaged_test_configuration",
        }
        return {
            "status": (
                "ok" if database_ready and model_ready and release_ready else "not_ready"
            ),
            "service": "creatorcut-web",
            "database": "ok" if database_ready else "unavailable",
            "frozen_model": "ok" if model_ready else "missing",
            "serving_release": release,
            "worker_mode": self.worker_mode,
            "queue": self.store.processing_queue_summary() if database_ready else None,
        }


def create_handler(application: ProductApplication) -> type[BaseHTTPRequestHandler]:
    """Bind the CreatorCut product application to a standard-library HTTP handler."""

    class ProductHandler(BaseHTTPRequestHandler):
        server_version = "CreatorCutProduct/0.1"

        def _session_token(self) -> str | None:
            cookie = SimpleCookie()
            try:
                cookie.load(self.headers.get("Cookie", ""))
            except Exception:
                return None
            morsel = cookie.get(SESSION_COOKIE_NAME)
            return morsel.value if morsel else None

        def _current_account(self) -> dict[str, Any] | None:
            return application.store.account_for_session(self._session_token())

        def _require_account(self, *, require_csrf: bool = False) -> dict[str, Any] | None:
            account = self._current_account()
            if account is None:
                self._send_json({"error": "Sign in to continue"}, HTTPStatus.UNAUTHORIZED)
                return None
            if require_csrf and not hmac.compare_digest(
                self.headers.get("X-CSRF-Token", ""), account["csrf_token"]
            ):
                self._send_json(
                    {"error": "The security token is missing or expired. Refresh and try again."},
                    HTTPStatus.FORBIDDEN,
                )
                return None
            return account

        def _require_admin(self) -> dict[str, Any] | None:
            if not application.admin_dashboard_enabled:
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return None
            account = self._require_account()
            if account is None:
                return None
            if not account["is_admin"]:
                self._send_json(
                    {"error": "Administrator access is required"},
                    HTTPStatus.FORBIDDEN,
                )
                return None
            return account

        @staticmethod
        def _account_response(account: dict[str, Any]) -> dict[str, Any]:
            return {
                key: account[key]
                for key in ("creator_id", "display_name", "email", "is_admin")
            }

        def _require_video_owner(
            self, video_id: str, account: dict[str, Any]
        ) -> dict[str, Any]:
            video = application.store.get_video(video_id)
            if video["creator_id"] != account["creator_id"]:
                raise KeyError(video_id)
            return video

        def _require_clip_owner(
            self, clip_id: str, account: dict[str, Any]
        ) -> dict[str, Any]:
            clip = application.store.get_clip(clip_id)
            if clip["creator_id"] != account["creator_id"]:
                raise KeyError(clip_id)
            return clip

        def _session_cookie(self, token: str, *, clear: bool = False) -> str:
            value = "" if clear else token
            max_age = 0 if clear else SESSION_MAX_AGE_SECONDS
            cookie = (
                f"{SESSION_COOKIE_NAME}={value}; Path=/; HttpOnly; SameSite=Lax; "
                f"Max-Age={max_age}"
            )
            if application.secure_cookies:
                cookie += "; Secure"
            return cookie

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
            if path == "/api/config":
                account = self._current_account()
                self._send_json(
                    {
                        "admin_dashboard_enabled": bool(
                            application.admin_dashboard_enabled
                            and account
                            and account["is_admin"]
                        )
                    }
                )
                return
            if path == "/api/auth/session":
                account = self._current_account()
                if account is None:
                    self._send_json({"authenticated": False})
                else:
                    self._send_json(
                        {
                            "authenticated": True,
                            "account": self._account_response(account),
                            "csrf_token": account["csrf_token"],
                            "expires_at": account["session_expires_at"],
                        }
                    )
                return
            if path == "/admin":
                if self._require_admin() is None:
                    return
                self._send_file(STATIC_DIRECTORY / "admin.html")
                return
            if path == "/assets/admin.css":
                if self._require_admin() is None:
                    return
                self._send_file(STATIC_DIRECTORY / "admin.css")
                return
            if path == "/assets/admin.js":
                if self._require_admin() is None:
                    return
                self._send_file(STATIC_DIRECTORY / "admin.js")
                return
            if path == "/api/admin/model-report":
                if self._require_admin() is None:
                    return
                report = application.store.admin_ml_report()
                report["serving_release"] = application.health()["serving_release"]
                self._send_json(report)
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
            account = self._require_account()
            if account is None:
                return
            if path.startswith("/api/videos/") and path.endswith("/source"):
                video_id = unquote(path[len("/api/videos/") : -len("/source")]).strip("/")
                try:
                    self._require_video_owner(video_id, account)
                    self._send_file(
                        application.store.media_path_for_video(video_id), ranged=True
                    )
                except (KeyError, FileNotFoundError):
                    self._send_json({"error": "Video not found"}, HTTPStatus.NOT_FOUND)
                return
            if path.startswith("/api/videos/"):
                video_id = unquote(path[len("/api/videos/") :]).strip("/")
                try:
                    video = self._require_video_owner(video_id, account)
                    video["creator_summary"] = application.store.creator_summary(
                        video["creator_id"]
                    )
                    self._send_json(video)
                except KeyError:
                    self._send_json({"error": "Video not found"}, HTTPStatus.NOT_FOUND)
                return
            if path == "/api/account/videos":
                self._send_json(
                    {"videos": application.store.list_videos(account["creator_id"])}
                )
                return
            if path == "/api/account/model-report":
                try:
                    report = application.store.creator_ml_report(account["creator_id"])
                    report["serving_release"] = application.health()["serving_release"]
                    self._send_json(report)
                except KeyError:
                    self._send_json({"error": "Creator not found"}, HTTPStatus.NOT_FOUND)
                return
            if path == "/api/account/feedback-export":
                try:
                    snapshot = build_feedback_snapshot(
                        application.store, account["creator_id"]
                    )
                    snapshot["serving_release"] = (
                        application.processor.serving_release_status()
                    )
                    self._send_json(
                        snapshot,
                        download_name="creatorcut-feedback-snapshot.json",
                    )
                except KeyError:
                    self._send_json({"error": "Creator not found"}, HTTPStatus.NOT_FOUND)
                return
            if path.startswith("/api/exports/"):
                token = unquote(path[len("/api/exports/") :]).strip("/")
                try:
                    download = application.store.export_download_for_account(
                        token, account["creator_id"]
                    )
                except KeyError:
                    self._send_json({"error": "Export not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._send_file(
                    download["path"], download_name=download["download_name"]
                )
                return
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path == "/api/auth/register":
                    self._handle_register()
                    return
                if path == "/api/auth/login":
                    self._handle_login()
                    return
                account = self._require_account(require_csrf=True)
                if account is None:
                    return
                if path == "/api/auth/logout":
                    self._handle_logout()
                    return
                if path == "/api/uploads":
                    self._handle_upload(account)
                    return
                if path.startswith("/api/clips/") and path.endswith("/decision"):
                    clip_id = unquote(path[len("/api/clips/") : -len("/decision")]).strip("/")
                    self._handle_decision(clip_id, account)
                    return
                if path.startswith("/api/clips/") and path.endswith("/export"):
                    clip_id = unquote(path[len("/api/clips/") : -len("/export")]).strip("/")
                    self._handle_export(clip_id, account)
                    return
                if path.startswith("/api/clips/") and path.endswith("/performance"):
                    clip_id = unquote(path[len("/api/clips/") : -len("/performance")]).strip("/")
                    self._handle_performance(clip_id, account)
                    return
                if path.startswith("/api/clips/") and path.endswith("/analytics"):
                    clip_id = unquote(path[len("/api/clips/") : -len("/analytics")]).strip("/")
                    self._handle_analytics_import("published_clip", clip_id, account)
                    return
                if path.startswith("/api/clips/") and path.endswith("/repurpose-feedback"):
                    clip_id = unquote(
                        path[len("/api/clips/") : -len("/repurpose-feedback")]
                    ).strip("/")
                    self._handle_repurposing_feedback(clip_id, account)
                    return
                if path.startswith("/api/clips/") and path.endswith("/repurpose"):
                    clip_id = unquote(
                        path[len("/api/clips/") : -len("/repurpose")]
                    ).strip("/")
                    self._handle_repurpose(clip_id, account)
                    return
                if path.startswith("/api/videos/") and path.endswith("/analytics"):
                    video_id = unquote(path[len("/api/videos/") : -len("/analytics")]).strip("/")
                    self._handle_analytics_import("source_video", video_id, account)
                    return
                if path.startswith("/api/videos/") and path.endswith("/custom-clips"):
                    video_id = unquote(
                        path[len("/api/videos/") : -len("/custom-clips")]
                    ).strip("/")
                    self._handle_custom_clip(video_id, account)
                    return
                if path.startswith("/api/videos/") and path.endswith("/review-complete"):
                    video_id = unquote(
                        path[len("/api/videos/") : -len("/review-complete")]
                    ).strip("/")
                    self._handle_review_complete(video_id, account)
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

        def _handle_register(self) -> None:
            value = self._read_json()
            account = application.store.register_account(
                value.get("display_name"),
                value.get("email"),
                value.get("password"),
                admin_if_first=application.allow_first_admin_registration,
            )
            session = application.store.create_auth_session(account["creator_id"])
            log_event(
                LOGGER,
                "creator_account_registered",
                creator_id=account["creator_id"],
                administrator=account["is_admin"],
            )
            self._send_json(
                {
                    "authenticated": True,
                    "account": account,
                    "csrf_token": session["csrf_token"],
                    "expires_at": session["expires_at"],
                },
                HTTPStatus.CREATED,
                headers={"Set-Cookie": self._session_cookie(session["token"])},
            )

        def _handle_login(self) -> None:
            value = self._read_json()
            login_key = (
                f"{self.client_address[0]}:"
                f"{str(value.get('email', '')).strip().casefold()[:254]}"
            )
            if not application.login_allowed(login_key):
                self._send_json(
                    {"error": "Too many sign-in attempts. Try again later."},
                    HTTPStatus.TOO_MANY_REQUESTS,
                    headers={"Retry-After": str(LOGIN_FAILURE_WINDOW_SECONDS)},
                )
                return
            account = application.store.authenticate_account(
                value.get("email"), value.get("password")
            )
            if account is None:
                application.record_login_failure(login_key)
                self._send_json(
                    {"error": "Email or password is incorrect"},
                    HTTPStatus.UNAUTHORIZED,
                )
                return
            application.clear_login_failures(login_key)
            session = application.store.create_auth_session(account["creator_id"])
            log_event(LOGGER, "creator_account_login", creator_id=account["creator_id"])
            self._send_json(
                {
                    "authenticated": True,
                    "account": account,
                    "csrf_token": session["csrf_token"],
                    "expires_at": session["expires_at"],
                },
                headers={"Set-Cookie": self._session_cookie(session["token"])},
            )

        def _handle_logout(self) -> None:
            application.store.delete_auth_session(self._session_token())
            self._send_json(
                {"authenticated": False},
                headers={"Set-Cookie": self._session_cookie("", clear=True)},
            )

        def _handle_upload(self, account: dict[str, Any]) -> None:
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
                account,
                video_field.filename,
                video_field.file,
                analytics_filename,
                analytics_report,
            )
            self._send_json(result, HTTPStatus.ACCEPTED)

        def _handle_decision(self, clip_id: str, account: dict[str, Any]) -> None:
            value = self._read_json()
            if value.get("action") != "reject":
                raise ValueError("The decision action must be reject")
            clip = self._require_clip_owner(clip_id, account)
            application.store.record_editorial_event(
                clip_id,
                "reject",
                float(clip["start_seconds"]),
                float(clip["end_seconds"]),
                {"rank": clip["rank"]},
            )
            self._send_json({"saved": True})

        def _handle_export(self, clip_id: str, account: dict[str, Any]) -> None:
            value = self._read_json()
            export_format = value.get("export_format", "original")
            clip = self._require_clip_owner(clip_id, account)
            video = self._require_video_owner(clip["video_id"], account)
            start, end = validate_export_interval(
                clip,
                value.get("start_seconds"),
                value.get("end_seconds"),
                float(video["duration_seconds"]),
            )
            export_path, reframe_metadata = application.processor.export_clip(
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
                    "reframing": reframe_metadata,
                },
            )
            download = application.store.create_export_download(
                account["creator_id"], clip_id, export_path
            )
            self._send_json(
                {
                    "download_url": f"/api/exports/{download['token']}",
                    "edited": edited,
                    "reframing": reframe_metadata,
                }
            )

        def _handle_performance(self, clip_id: str, account: dict[str, Any]) -> None:
            self._require_clip_owner(clip_id, account)
            report = validate_performance_report(self._read_json())
            application.processor.ensure_clip_semantic_embedding(clip_id)
            application.store.save_performance_report(clip_id, report)
            self._send_json({"saved": True}, HTTPStatus.CREATED)

        def _handle_repurpose(self, clip_id: str, account: dict[str, Any]) -> None:
            self._require_clip_owner(clip_id, account)
            self._read_json()
            pack = application.processor.create_repurposing_pack(clip_id)
            self._send_json({"pack": pack}, HTTPStatus.CREATED)

        def _handle_repurposing_feedback(
            self, clip_id: str, account: dict[str, Any]
        ) -> None:
            self._require_clip_owner(clip_id, account)
            value = self._read_json()
            pack_id = value.get("pack_id")
            generated_text = value.get("generated_text")
            final_text = value.get("final_text")
            if not isinstance(pack_id, str) or not pack_id:
                raise ValueError("A repurposing pack ID is required")
            if not isinstance(generated_text, str):
                raise ValueError("Generated text is required")
            if len(generated_text.encode("utf-8")) > MAXIMUM_REPURPOSING_TEXT_BYTES:
                raise ValueError("Generated text is too long")
            if final_text is not None:
                if not isinstance(final_text, str):
                    raise ValueError("Final text must be text")
                if len(final_text.encode("utf-8")) > MAXIMUM_REPURPOSING_TEXT_BYTES:
                    raise ValueError("Final text is too long")
            saved = application.store.record_repurposing_feedback(
                clip_id,
                pack_id,
                value.get("platform"),
                value.get("action"),
                generated_text,
                final_text,
            )
            self._send_json({"saved": True, "feedback": saved}, HTTPStatus.CREATED)

        def _handle_custom_clip(self, video_id: str, account: dict[str, Any]) -> None:
            self._require_video_owner(video_id, account)
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

        def _handle_review_complete(
            self, video_id: str, account: dict[str, Any]
        ) -> None:
            self._require_video_owner(video_id, account)
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

        def _handle_analytics_import(
            self, report_role: str, target_id: str, account: dict[str, Any]
        ) -> None:
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
                self._require_clip_owner(target_id, account)
                application.processor.ensure_clip_semantic_embedding(target_id)
                saved = application.store.save_analytics_import(
                    account["creator_id"],
                    analytics_field.filename,
                    report_role,
                    report,
                    clip_id=target_id,
                )
            else:
                self._require_video_owner(target_id, account)
                saved = application.store.save_analytics_import(
                    account["creator_id"],
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
            self,
            value: Any,
            status: HTTPStatus = HTTPStatus.OK,
            download_name: str | None = None,
            headers: dict[str, str] | None = None,
        ) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers()
            if download_name:
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{download_name}"'
                )
            for name, header_value in (headers or {}).items():
                self.send_header(name, header_value)
            self.end_headers()
            self.wfile.write(payload)

        def _send_security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; "
                "form-action 'self'; img-src 'self' data:; media-src 'self' blob:; "
                "script-src 'self'; style-src 'self'",
            )
            if application.secure_cookies:
                self.send_header(
                    "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
                )

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
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers()
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
    parser.add_argument(
        "--serving-release",
        type=Path,
        default=Path("models/serving_release_v1.json"),
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
    parser.add_argument(
        "--enable-admin-dashboard",
        action="store_true",
        help="Expose the local observatory to authenticated administrators.",
    )
    args = parser.parse_args()
    if args.host not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        parser.error(
            "The credentialed local server may only bind to a loopback host; "
            "use managed HTTPS and authentication for a public deployment"
        )

    configure_logging(args.log_level)
    store = ProductStore(args.database)
    processor = ProductProcessor(
        store,
        args.work_dir,
        args.frozen_model,
        args.model_cache,
        args.semantic_cache,
        args.serving_release,
    )
    application = ProductApplication(
        store,
        processor,
        args.upload_dir,
        worker_mode=args.worker_mode,
        admin_dashboard_enabled=args.enable_admin_dashboard,
        secure_cookies=False,
        allow_first_admin_registration=args.enable_admin_dashboard,
    )
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
