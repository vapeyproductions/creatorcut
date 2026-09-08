"""Durable product records and conservative per-creator preference updates."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

from creatorcut.account_security import (
    DUMMY_PASSWORD_VERIFIER,
    create_password_verifier,
    normalize_display_name,
    normalize_email,
    session_credentials,
    session_token_digest,
    verify_password,
)
from creatorcut.analytics import performance_values

EDITORIAL_EVENT_WEIGHTS = {
    "download_original": 1.0,
    "download_edited": 1.0,
    "custom_created": 1.0,
    "review_closed_unselected": -0.35,
    "reject": -1.0,
}
PERSONALIZATION_PRIOR_STRENGTH = 8.0
MINIMUM_EDITORIAL_DECISIONS = 3
MAXIMUM_PERSONALIZATION_ADJUSTMENT = 0.35
PERFORMANCE_PRIOR_STRENGTH = 12.0
MAXIMUM_PERFORMANCE_ADJUSTMENT = 0.25
MAXIMUM_STRUCTURED_PERFORMANCE_ADJUSTMENT = 0.15
MAXIMUM_SEMANTIC_PERFORMANCE_ADJUSTMENT = 0.15
MINIMUM_PERFORMANCE_EXAMPLES = 5
MINIMUM_PERFORMANCE_VIEWS = 50
MAXIMUM_SOURCE_RETENTION_ADJUSTMENT = 0.20
MINIMUM_SOURCE_RETENTION_POINTS = 10
MINIMUM_SOURCE_RETENTION_VIEWS = 100

PREFERENCE_FEATURE_NAMES = ("hook", "completeness", "payoff", "clarity", "duration")
SEMANTIC_TREND_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "for",
    "from",
    "had",
    "has",
    "have",
    "here",
    "how",
    "into",
    "its",
    "just",
    "like",
    "more",
    "not",
    "now",
    "really",
    "that",
    "the",
    "their",
    "then",
    "there",
    "they",
    "this",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def utc_now() -> str:
    """Return a stable UTC timestamp for persisted product events."""
    return datetime.now(UTC).isoformat()


def preference_features(clip: dict[str, Any]) -> list[float]:
    """Map a ranked clip into a small, interpretable creator-preference vector."""
    targets = clip.get("predicted_targets", {})
    return [
        (float(targets.get(field, 3.0)) - 3.0) / 2.0
        for field in ("hook", "completeness", "payoff", "clarity")
    ] + [(float(clip["duration_seconds"]) - 40.0) / 20.0]


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    """Summarize a numeric operational signal without inventing empty values."""
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return {
            "count": 0,
            "minimum": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "maximum": None,
        }

    def percentile(fraction: float) -> float:
        position = (len(clean) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return clean[lower]
        weight = position - lower
        return clean[lower] * (1 - weight) + clean[upper] * weight

    return {
        "count": len(clean),
        "minimum": round(clean[0], 3),
        "p25": round(percentile(0.25), 3),
        "median": round(percentile(0.5), 3),
        "mean": round(sum(clean) / len(clean), 3),
        "p75": round(percentile(0.75), 3),
        "maximum": round(clean[-1], 3),
    }


class ProductStore:
    """Persist uploads, ranked clips, decisions, edits, and performance reports in SQLite."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS creator_profiles (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS account_credentials (
                creator_id TEXT PRIMARY KEY REFERENCES creator_profiles(id) ON DELETE CASCADE,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_algorithm TEXT NOT NULL,
                password_iterations INTEGER NOT NULL,
                password_salt TEXT NOT NULL,
                password_digest TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0 CHECK(is_admin IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_digest TEXT PRIMARY KEY,
                creator_id TEXT NOT NULL
                    REFERENCES account_credentials(creator_id) ON DELETE CASCADE,
                csrf_token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS export_downloads (
                token TEXT PRIMARY KEY,
                creator_id TEXT NOT NULL REFERENCES creator_profiles(id) ON DELETE CASCADE,
                clip_id TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
                file_path TEXT NOT NULL,
                download_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                creator_id TEXT NOT NULL REFERENCES creator_profiles(id),
                original_filename TEXT NOT NULL,
                media_path TEXT NOT NULL,
                status TEXT NOT NULL,
                duration_seconds REAL,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS clips (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                rank INTEGER NOT NULL,
                start_seconds REAL NOT NULL,
                end_seconds REAL NOT NULL,
                duration_seconds REAL NOT NULL,
                transcript_text TEXT NOT NULL,
                global_score REAL NOT NULL,
                personalized_score REAL NOT NULL,
                predicted_targets_json TEXT NOT NULL,
                explanation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(video_id, rank)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS feedback_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                creator_id TEXT NOT NULL REFERENCES creator_profiles(id),
                video_id TEXT NOT NULL REFERENCES videos(id),
                clip_id TEXT NOT NULL REFERENCES clips(id),
                event_type TEXT NOT NULL,
                start_seconds REAL,
                end_seconds REAL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS performance_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                creator_id TEXT NOT NULL REFERENCES creator_profiles(id),
                clip_id TEXT NOT NULL REFERENCES clips(id),
                platform TEXT NOT NULL,
                views INTEGER,
                likes INTEGER,
                comments INTEGER,
                shares INTEGER,
                average_view_percentage REAL,
                published_at TEXT,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS analytics_imports (
                id TEXT PRIMARY KEY,
                creator_id TEXT NOT NULL REFERENCES creator_profiles(id),
                video_id TEXT REFERENCES videos(id) ON DELETE CASCADE,
                clip_id TEXT REFERENCES clips(id) ON DELETE CASCADE,
                report_role TEXT NOT NULL,
                platform TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                report_json TEXT NOT NULL,
                recognized_row_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS processing_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL UNIQUE REFERENCES videos(id) ON DELETE CASCADE,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                available_at TEXT NOT NULL,
                lease_owner TEXT,
                lease_expires_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS repurposing_packs (
                id TEXT PRIMARY KEY,
                creator_id TEXT NOT NULL REFERENCES creator_profiles(id),
                clip_id TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
                algorithm_version TEXT NOT NULL,
                pack_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS repurposing_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                creator_id TEXT NOT NULL REFERENCES creator_profiles(id),
                clip_id TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
                pack_id TEXT NOT NULL REFERENCES repurposing_packs(id) ON DELETE CASCADE,
                platform TEXT NOT NULL,
                action TEXT NOT NULL,
                generated_text TEXT NOT NULL,
                final_text TEXT,
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_videos_creator_id ON videos(creator_id)",
            "CREATE INDEX IF NOT EXISTS idx_auth_sessions_creator_id ON auth_sessions(creator_id)",
            "CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry ON auth_sessions(expires_at)",
            """
            CREATE INDEX IF NOT EXISTS idx_export_downloads_expiry
            ON export_downloads(expires_at)
            """,
            "CREATE INDEX IF NOT EXISTS idx_clips_video_id ON clips(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_creator_id ON feedback_events(creator_id)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_clip_id ON feedback_events(clip_id)",
            """
            CREATE INDEX IF NOT EXISTS idx_performance_creator_id
            ON performance_reports(creator_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_analytics_creator_id
            ON analytics_imports(creator_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_analytics_video_id
            ON analytics_imports(video_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_processing_jobs_claim
            ON processing_jobs(status, available_at, created_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_repurposing_packs_creator
            ON repurposing_packs(creator_id, created_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_repurposing_feedback_creator
            ON repurposing_feedback(creator_id, created_at)
            """,
        )
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            for statement in statements:
                connection.execute(statement)
            self._add_missing_columns(connection)
            self._backfill_processing_jobs(connection)
            connection.execute("PRAGMA optimize")

    @staticmethod
    def _backfill_processing_jobs(connection: sqlite3.Connection) -> None:
        """Give databases from older builds a durable job record per upload."""
        now = utc_now()
        connection.execute(
            """
            INSERT OR IGNORE INTO processing_jobs(
                video_id, status, attempt_count, max_attempts, available_at,
                created_at, updated_at, completed_at
            )
            SELECT id,
                   CASE
                       WHEN status = 'ready' THEN 'succeeded'
                       WHEN status = 'failed' THEN 'failed'
                       ELSE 'queued'
                   END,
                   0, 3, ?, created_at, updated_at,
                   CASE WHEN status IN ('ready', 'failed') THEN updated_at ELSE NULL END
            FROM videos
            """,
            (now,),
        )

    @staticmethod
    def _add_missing_columns(connection: sqlite3.Connection) -> None:
        """Apply additive migrations to databases created by earlier local builds."""
        migrations = {
            "videos": {
                "publishability_summary_json": "TEXT NOT NULL DEFAULT '{}'",
            },
            "clips": {
                "global_rank": "INTEGER",
                "ranking_model_version": "TEXT",
                "origin": "TEXT NOT NULL DEFAULT 'model'",
                "editorial_adjustment": "REAL NOT NULL DEFAULT 0",
                "performance_adjustment": "REAL NOT NULL DEFAULT 0",
                "semantic_performance_adjustment": "REAL NOT NULL DEFAULT 0",
                "source_retention_adjustment": "REAL NOT NULL DEFAULT 0",
                "semantic_embedding_json": "TEXT",
                "publishability_json": "TEXT NOT NULL DEFAULT '{}'",
                "publishability_adjustment": "REAL NOT NULL DEFAULT 0",
            },
            "performance_reports": {
                "engaged_views": "INTEGER",
                "watch_time_hours": "REAL",
                "average_view_duration_seconds": "REAL",
                "subscribers_gained": "INTEGER",
                "subscribers_lost": "INTEGER",
                "subscribers_net": "INTEGER",
                "shown_in_feed": "INTEGER",
                "chose_to_view_percentage": "REAL",
                "thumbnail_impressions": "INTEGER",
                "thumbnail_ctr": "REAL",
                "analytics_import_id": "TEXT",
            },
        }
        for table, columns in migrations.items():
            existing = {
                row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
            }
            for name, declaration in columns.items():
                if name not in existing:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    def ensure_creator(self, display_name: str, creator_id: str | None = None) -> dict[str, str]:
        """Create a local creator profile or return the existing profile."""
        normalized_name = display_name.strip() or "Local creator"
        normalized_id = creator_id or f"creator_{uuid.uuid4().hex}"
        with self._write_lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT id, display_name FROM creator_profiles WHERE id = ?", (normalized_id,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO creator_profiles(id, display_name, created_at) VALUES (?, ?, ?)",
                    (normalized_id, normalized_name, utc_now()),
                )
                return {"id": normalized_id, "display_name": normalized_name}
            return {"id": existing["id"], "display_name": existing["display_name"]}

    @staticmethod
    def _public_account(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "creator_id": row["creator_id"],
            "display_name": row["display_name"],
            "email": row["email"],
            "is_admin": bool(row["is_admin"]),
        }

    def register_account(
        self,
        display_name: Any,
        email: Any,
        password: Any,
        *,
        is_admin: bool = False,
        admin_if_first: bool = False,
    ) -> dict[str, Any]:
        """Create a credentialed creator profile in one transaction."""
        checked_name = normalize_display_name(display_name)
        checked_email = normalize_email(email)
        verifier = create_password_verifier(password)
        creator_id = f"creator_{uuid.uuid4().hex}"
        now = utc_now()
        try:
            with self._write_lock, self._connect() as connection:
                granted_admin = bool(is_admin)
                if admin_if_first and not granted_admin:
                    granted_admin = (
                        connection.execute(
                            "SELECT COUNT(*) FROM account_credentials"
                        ).fetchone()[0]
                        == 0
                    )
                connection.execute(
                    "INSERT INTO creator_profiles(id, display_name, created_at) VALUES (?, ?, ?)",
                    (creator_id, checked_name, now),
                )
                connection.execute(
                    """
                    INSERT INTO account_credentials(
                        creator_id, email, password_algorithm, password_iterations,
                        password_salt, password_digest, is_admin, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        creator_id,
                        checked_email,
                        verifier["algorithm"],
                        verifier["iterations"],
                        verifier["salt"],
                        verifier["digest"],
                        int(granted_admin),
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as error:
            if "email" in str(error).casefold() or "unique" in str(error).casefold():
                raise ValueError("An account with that email already exists") from error
            raise
        return {
            "creator_id": creator_id,
            "display_name": checked_name,
            "email": checked_email,
            "is_admin": granted_admin,
        }

    def authenticate_account(self, email: Any, password: Any) -> dict[str, Any] | None:
        """Verify credentials while doing equivalent password work for unknown emails."""
        try:
            checked_email = normalize_email(email)
        except ValueError:
            checked_email = "invalid@example.invalid"
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT account_credentials.creator_id, account_credentials.email,
                       account_credentials.password_algorithm,
                       account_credentials.password_iterations,
                       account_credentials.password_salt,
                       account_credentials.password_digest,
                       account_credentials.is_admin, creator_profiles.display_name
                FROM account_credentials
                JOIN creator_profiles ON creator_profiles.id = account_credentials.creator_id
                WHERE account_credentials.email = ? COLLATE NOCASE
                """,
                (checked_email,),
            ).fetchone()
        verifier = (
            {
                "algorithm": row["password_algorithm"],
                "iterations": row["password_iterations"],
                "salt": row["password_salt"],
                "digest": row["password_digest"],
            }
            if row is not None
            else DUMMY_PASSWORD_VERIFIER
        )
        if not verify_password(password, verifier) or row is None:
            return None
        return self._public_account(row)

    def create_auth_session(self, creator_id: str) -> dict[str, Any]:
        """Issue an opaque seven-day session and delete expired records."""
        session = session_credentials()
        with self._write_lock, self._connect() as connection:
            account = connection.execute(
                "SELECT 1 FROM account_credentials WHERE creator_id = ?", (creator_id,)
            ).fetchone()
            if account is None:
                raise KeyError(creator_id)
            connection.execute(
                "DELETE FROM auth_sessions WHERE expires_at <= ?", (session["created_at"],)
            )
            connection.execute(
                """
                INSERT INTO auth_sessions(
                    token_digest, creator_id, csrf_token, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session["token_digest"],
                    creator_id,
                    session["csrf_token"],
                    session["created_at"],
                    session["expires_at"],
                ),
            )
        return {
            "token": session["token"],
            "csrf_token": session["csrf_token"],
            "expires_at": session["expires_at"],
        }

    def account_for_session(self, token: str | None) -> dict[str, Any] | None:
        """Resolve a valid opaque session to its creator and authorization role."""
        if not token or len(token) > 256:
            return None
        digest = session_token_digest(token)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT account_credentials.creator_id, account_credentials.email,
                       account_credentials.is_admin, creator_profiles.display_name,
                       auth_sessions.csrf_token, auth_sessions.expires_at
                FROM auth_sessions
                JOIN account_credentials
                  ON account_credentials.creator_id = auth_sessions.creator_id
                JOIN creator_profiles ON creator_profiles.id = account_credentials.creator_id
                WHERE auth_sessions.token_digest = ? AND auth_sessions.expires_at > ?
                """,
                (digest, utc_now()),
            ).fetchone()
        if row is None:
            return None
        return {
            **self._public_account(row),
            "csrf_token": row["csrf_token"],
            "session_expires_at": row["expires_at"],
        }

    def delete_auth_session(self, token: str | None) -> None:
        """Revoke a browser session without exposing whether it existed."""
        if not token or len(token) > 256:
            return
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM auth_sessions WHERE token_digest = ?",
                (session_token_digest(token),),
            )

    def authenticated_account_count(self) -> int:
        """Count credentialed accounts, excluding legacy local-only profiles."""
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM account_credentials").fetchone()[0])

    def has_admin_account(self) -> bool:
        """Return whether an administrator has been explicitly provisioned."""
        with self._connect() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM account_credentials WHERE is_admin = 1 LIMIT 1"
                ).fetchone()
                is not None
            )

    def promote_account_to_admin(self, email: Any) -> dict[str, Any]:
        """Grant the administrator role to an existing credentialed account."""
        checked_email = normalize_email(email)
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE account_credentials SET is_admin = 1, updated_at = ?
                WHERE email = ? COLLATE NOCASE
                """,
                (utc_now(), checked_email),
            )
            row = connection.execute(
                """
                SELECT account_credentials.creator_id, account_credentials.email,
                       account_credentials.is_admin, creator_profiles.display_name
                FROM account_credentials
                JOIN creator_profiles ON creator_profiles.id = account_credentials.creator_id
                WHERE account_credentials.email = ? COLLATE NOCASE
                """,
                (checked_email,),
            ).fetchone()
            if row is None:
                raise KeyError(checked_email)
        return self._public_account(row)

    def create_export_download(
        self, creator_id: str, clip_id: str, file_path: Path
    ) -> dict[str, str]:
        """Create a short-lived, owner-bound download instead of exposing a file name."""
        token = uuid.uuid4().hex + uuid.uuid4().hex
        created = datetime.now(UTC)
        expires = created + timedelta(hours=24)
        with self._write_lock, self._connect() as connection:
            owner = connection.execute(
                """
                SELECT videos.creator_id
                FROM clips JOIN videos ON videos.id = clips.video_id
                WHERE clips.id = ?
                """,
                (clip_id,),
            ).fetchone()
            if owner is None or owner["creator_id"] != creator_id:
                raise KeyError(clip_id)
            connection.execute(
                "DELETE FROM export_downloads WHERE expires_at <= ?", (created.isoformat(),)
            )
            connection.execute(
                """
                INSERT INTO export_downloads(
                    token, creator_id, clip_id, file_path, download_name, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token,
                    creator_id,
                    clip_id,
                    file_path.as_posix(),
                    file_path.name,
                    created.isoformat(),
                    expires.isoformat(),
                ),
            )
        return {
            "token": token,
            "download_name": file_path.name,
            "expires_at": expires.isoformat(),
        }

    def export_download_for_account(
        self, token: str, creator_id: str
    ) -> dict[str, Any]:
        """Resolve a download only when the current creator owns it and it remains valid."""
        if len(token) != 64 or not all(character in "0123456789abcdef" for character in token):
            raise KeyError(token)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT file_path, download_name, expires_at
                FROM export_downloads
                WHERE token = ? AND creator_id = ? AND expires_at > ?
                """,
                (token, creator_id, utc_now()),
            ).fetchone()
        if row is None:
            raise KeyError(token)
        return {
            "path": Path(row["file_path"]),
            "download_name": row["download_name"],
            "expires_at": row["expires_at"],
        }

    def create_video(
        self, creator_id: str, original_filename: str, media_path: Path
    ) -> dict[str, Any]:
        """Register an uploaded source before background inference begins."""
        video_id = f"upload_{uuid.uuid4().hex}"
        created_at = utc_now()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO videos(
                    id, creator_id, original_filename, media_path, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    video_id,
                    creator_id,
                    original_filename,
                    media_path.as_posix(),
                    created_at,
                    created_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO processing_jobs(
                    video_id, status, attempt_count, max_attempts, available_at,
                    created_at, updated_at
                ) VALUES (?, 'queued', 0, 3, ?, ?, ?)
                """,
                (video_id, created_at, created_at, created_at),
            )
        return self.get_video(video_id)

    def update_video(
        self,
        video_id: str,
        status: str,
        duration_seconds: float | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update processing state without changing the registered source identity."""
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE videos
                SET status = ?, duration_seconds = COALESCE(?, duration_seconds),
                    error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, duration_seconds, error_message, utc_now(), video_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(video_id)
            if status == "ready":
                connection.execute(
                    """
                    UPDATE processing_jobs
                    SET status = 'succeeded', lease_owner = NULL,
                        lease_expires_at = NULL, completed_at = ?, updated_at = ?
                    WHERE video_id = ? AND status != 'succeeded'
                    """,
                    (utc_now(), utc_now(), video_id),
                )

    def save_video_publishability_summary(
        self, video_id: str, summary: dict[str, Any]
    ) -> None:
        """Persist full candidate-gate counts, including removed candidates."""
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE videos SET publishability_summary_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(summary, sort_keys=True), utc_now(), video_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(video_id)

    def claim_next_processing_job(
        self,
        worker_id: str,
        lease_seconds: float = 120.0,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Atomically claim the oldest available job and recover expired leases."""
        if not worker_id.strip():
            raise ValueError("worker_id must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        claimed_at = now or datetime.now(UTC)
        claimed_at_text = claimed_at.isoformat()
        lease_expires_at = (claimed_at + timedelta(seconds=lease_seconds)).isoformat()
        with self._write_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            expired = connection.execute(
                """
                SELECT id, video_id, attempt_count, max_attempts
                FROM processing_jobs
                WHERE status = 'running' AND lease_expires_at <= ?
                """,
                (claimed_at_text,),
            ).fetchall()
            for job in expired:
                exhausted = job["attempt_count"] >= job["max_attempts"]
                connection.execute(
                    """
                    UPDATE processing_jobs
                    SET status = ?, lease_owner = NULL, lease_expires_at = NULL,
                        last_error = ?, completed_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        "failed" if exhausted else "queued",
                        "Worker lease expired before the job completed",
                        claimed_at_text if exhausted else None,
                        claimed_at_text,
                        job["id"],
                    ),
                )
                connection.execute(
                    """
                    UPDATE videos SET status = ?, error_message = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        "failed" if exhausted else "queued",
                        "Processing worker stopped; the job was recovered"
                        if not exhausted
                        else "Processing failed after all retry attempts",
                        claimed_at_text,
                        job["video_id"],
                    ),
                )
            job = connection.execute(
                """
                SELECT * FROM processing_jobs
                WHERE status = 'queued' AND available_at <= ?
                  AND attempt_count < max_attempts
                ORDER BY available_at, created_at, id
                LIMIT 1
                """,
                (claimed_at_text,),
            ).fetchone()
            if job is None:
                return None
            cursor = connection.execute(
                """
                UPDATE processing_jobs
                SET status = 'running', attempt_count = attempt_count + 1,
                    lease_owner = ?, lease_expires_at = ?,
                    started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (
                    worker_id,
                    lease_expires_at,
                    claimed_at_text,
                    claimed_at_text,
                    job["id"],
                ),
            )
            if cursor.rowcount != 1:
                return None
            connection.execute(
                """
                UPDATE videos SET status = 'queued', error_message = NULL, updated_at = ?
                WHERE id = ?
                """,
                (claimed_at_text, job["video_id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM processing_jobs WHERE id = ?", (job["id"],)
            ).fetchone()
        return dict(claimed)

    def extend_processing_lease(
        self,
        job_id: int,
        worker_id: str,
        lease_seconds: float = 120.0,
        now: datetime | None = None,
    ) -> bool:
        """Extend a running job lease while expensive inference is active."""
        renewed_at = now or datetime.now(UTC)
        expires_at = (renewed_at + timedelta(seconds=lease_seconds)).isoformat()
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE processing_jobs
                SET lease_expires_at = ?, updated_at = ?
                WHERE id = ? AND status = 'running' AND lease_owner = ?
                """,
                (expires_at, renewed_at.isoformat(), job_id, worker_id),
            )
        return cursor.rowcount == 1

    def complete_processing_job(self, job_id: int, worker_id: str) -> None:
        """Mark a claimed job complete after its video reaches ready state."""
        completed_at = utc_now()
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE processing_jobs
                SET status = 'succeeded', lease_owner = NULL,
                    lease_expires_at = NULL, completed_at = ?, updated_at = ?
                WHERE id = ? AND status IN ('running', 'succeeded')
                  AND (lease_owner = ? OR lease_owner IS NULL)
                """,
                (completed_at, completed_at, job_id, worker_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Processing job lease is no longer owned by this worker")

    def fail_processing_job(
        self,
        job_id: int,
        worker_id: str,
        error: str,
        retry_delay_seconds: float = 0.0,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Retry a claimed job when possible, otherwise persist terminal failure."""
        failed_at = now or datetime.now(UTC)
        failed_at_text = failed_at.isoformat()
        with self._write_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                """
                SELECT * FROM processing_jobs
                WHERE id = ? AND status = 'running' AND lease_owner = ?
                """,
                (job_id, worker_id),
            ).fetchone()
            if job is None:
                raise RuntimeError("Processing job lease is no longer owned by this worker")
            retrying = job["attempt_count"] < job["max_attempts"]
            available_at = (
                failed_at + timedelta(seconds=max(0.0, retry_delay_seconds))
            ).isoformat()
            connection.execute(
                """
                UPDATE processing_jobs
                SET status = ?, available_at = ?, lease_owner = NULL,
                    lease_expires_at = NULL, last_error = ?, completed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    "queued" if retrying else "failed",
                    available_at,
                    str(error)[:500],
                    None if retrying else failed_at_text,
                    failed_at_text,
                    job_id,
                ),
            )
            connection.execute(
                """
                UPDATE videos SET status = ?, error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    "queued" if retrying else "failed",
                    f"Retry scheduled after attempt {job['attempt_count']}"
                    if retrying
                    else str(error)[:500],
                    failed_at_text,
                    job["video_id"],
                ),
            )
            saved = connection.execute(
                "SELECT * FROM processing_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return dict(saved)

    def processing_job_for_video(self, video_id: str) -> dict[str, Any] | None:
        """Return operational job state without exposing local paths."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, status, attempt_count, max_attempts, available_at,
                       lease_owner, lease_expires_at, last_error, created_at,
                       updated_at, started_at, completed_at
                FROM processing_jobs WHERE video_id = ?
                """,
                (video_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def processing_queue_summary(self) -> dict[str, Any]:
        """Return small operational counters for health checks and monitoring."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM processing_jobs GROUP BY status"
            ).fetchall()
            oldest = connection.execute(
                "SELECT MIN(created_at) AS created_at FROM processing_jobs WHERE status = 'queued'"
            ).fetchone()["created_at"]
        counts = {row["status"]: row["count"] for row in rows}
        return {
            "queued": counts.get("queued", 0),
            "running": counts.get("running", 0),
            "succeeded": counts.get("succeeded", 0),
            "failed": counts.get("failed", 0),
            "oldest_queued_at": oldest,
        }

    def database_ready(self) -> bool:
        """Probe the serving database without changing application records."""
        try:
            with self._connect() as connection:
                return connection.execute("SELECT 1").fetchone()[0] == 1
        except sqlite3.Error:
            return False

    def save_ranked_clips(self, video_id: str, clips: list[dict[str, Any]]) -> None:
        """Persist the three displayed results and one exposure event per result."""
        video = self.get_video(video_id)
        with self._write_lock, self._connect() as connection:
            for clip in clips:
                connection.execute(
                    """
                    INSERT INTO clips(
                        id, video_id, rank, start_seconds, end_seconds, duration_seconds,
                        transcript_text, global_score, personalized_score,
                        predicted_targets_json, explanation, global_rank,
                        ranking_model_version, origin,
                        editorial_adjustment, performance_adjustment,
                        semantic_performance_adjustment, source_retention_adjustment,
                        publishability_json, publishability_adjustment,
                        semantic_embedding_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clip["id"],
                        video_id,
                        clip["rank"],
                        clip["start_seconds"],
                        clip["end_seconds"],
                        clip["duration_seconds"],
                        clip["transcript_text"],
                        clip["global_score"],
                        clip["personalized_score"],
                        json.dumps(clip["predicted_targets"], sort_keys=True),
                        clip["explanation"],
                        clip.get("global_rank"),
                        clip.get("ranking_model_version"),
                        clip.get("origin", "model"),
                        clip.get("editorial_adjustment", 0.0),
                        clip.get("performance_adjustment", 0.0),
                        clip.get("semantic_performance_adjustment", 0.0),
                        clip.get("source_retention_adjustment", 0.0),
                        json.dumps(clip.get("publishability", {}), sort_keys=True),
                        clip.get("publishability_adjustment", 0.0),
                        json.dumps(clip.get("semantic_embedding"))
                        if clip.get("semantic_embedding") is not None
                        else None,
                        utc_now(),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO feedback_events(
                        creator_id, video_id, clip_id, event_type, start_seconds,
                        end_seconds, payload_json, created_at
                    ) VALUES (?, ?, ?, 'presented', ?, ?, ?, ?)
                    """,
                    (
                        video["creator_id"],
                        video_id,
                        clip["id"],
                        clip["start_seconds"],
                        clip["end_seconds"],
                        json.dumps(
                            {
                                "rank": clip["rank"],
                                "global_rank": clip.get("global_rank"),
                                "ranking_model_version": clip.get(
                                    "ranking_model_version"
                                ),
                                "global_score": clip["global_score"],
                                "personalized_score": clip["personalized_score"],
                                "editorial_adjustment": clip.get(
                                    "editorial_adjustment", 0.0
                                ),
                                "performance_adjustment": clip.get(
                                    "performance_adjustment", 0.0
                                ),
                                "semantic_performance_adjustment": clip.get(
                                    "semantic_performance_adjustment", 0.0
                                ),
                                "source_retention_adjustment": clip.get(
                                    "source_retention_adjustment", 0.0
                                ),
                                "publishability": clip.get("publishability", {}),
                                "publishability_adjustment": clip.get(
                                    "publishability_adjustment", 0.0
                                ),
                            },
                            sort_keys=True,
                        ),
                        utc_now(),
                    ),
                )

    def save_custom_clip(self, video_id: str, clip: dict[str, Any]) -> str:
        """Persist a creator-authored interval as preference evidence, not a model impression."""
        video = self.get_video(video_id)
        clip_id = f"{video_id}_custom_{uuid.uuid4().hex}"
        with self._write_lock, self._connect() as connection:
            next_rank = connection.execute(
                "SELECT COALESCE(MAX(rank), 0) + 1 FROM clips WHERE video_id = ?",
                (video_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO clips(
                    id, video_id, rank, start_seconds, end_seconds, duration_seconds,
                    transcript_text, global_score, personalized_score,
                    predicted_targets_json, explanation, global_rank,
                    ranking_model_version, origin, editorial_adjustment,
                    performance_adjustment, semantic_performance_adjustment,
                    source_retention_adjustment, publishability_json,
                    publishability_adjustment, semantic_embedding_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clip_id,
                    video_id,
                    next_rank,
                    clip["start_seconds"],
                    clip["end_seconds"],
                    clip["duration_seconds"],
                    clip["transcript_text"],
                    clip["global_score"],
                    clip["personalized_score"],
                    json.dumps(clip["predicted_targets"], sort_keys=True),
                    clip["explanation"],
                    None,
                    clip.get("ranking_model_version"),
                    "creator",
                    clip.get("editorial_adjustment", 0.0),
                    clip.get("performance_adjustment", 0.0),
                    clip.get("semantic_performance_adjustment", 0.0),
                    clip.get("source_retention_adjustment", 0.0),
                    json.dumps(clip.get("publishability", {}), sort_keys=True),
                    clip.get("publishability_adjustment", 0.0),
                    json.dumps(clip["semantic_embedding"]),
                    utc_now(),
                ),
            )
            connection.execute(
                """
                INSERT INTO feedback_events(
                    creator_id, video_id, clip_id, event_type, start_seconds,
                    end_seconds, payload_json, created_at
                ) VALUES (?, ?, ?, 'custom_created', ?, ?, ?, ?)
                """,
                (
                    video["creator_id"],
                    video_id,
                    clip_id,
                    clip["start_seconds"],
                    clip["end_seconds"],
                    json.dumps(
                        {
                            "origin": "creator",
                            "ranking_model_version": clip.get(
                                "ranking_model_version"
                            ),
                            "global_score": clip["global_score"],
                            "personalized_score": clip["personalized_score"],
                            "publishability": clip.get("publishability", {}),
                            "publishability_adjustment": clip.get(
                                "publishability_adjustment", 0.0
                            ),
                        },
                        sort_keys=True,
                    ),
                    utc_now(),
                ),
            )
        return clip_id

    def complete_recommendation_review(self, video_id: str) -> int:
        """Record explicitly closed, unselected model options as weak negative evidence."""
        video = self.get_video(video_id)
        with self._write_lock, self._connect() as connection:
            positive = connection.execute(
                """
                SELECT COUNT(*) FROM feedback_events
                WHERE video_id = ?
                  AND event_type IN (
                      'download_original', 'download_edited', 'custom_created'
                  )
                """,
                (video_id,),
            ).fetchone()[0]
            if positive == 0:
                raise ValueError(
                    "Choose or add at least one clip before closing the recommendation review"
                )
            unresolved = connection.execute(
                """
                SELECT clips.* FROM clips
                WHERE clips.video_id = ? AND clips.origin = 'model'
                  AND NOT EXISTS (
                      SELECT 1 FROM feedback_events
                      WHERE feedback_events.clip_id = clips.id
                        AND feedback_events.event_type IN (
                            'download_original', 'download_edited', 'reject',
                            'review_closed_unselected'
                        )
                  )
                ORDER BY clips.rank
                """,
                (video_id,),
            ).fetchall()
            for clip in unresolved:
                connection.execute(
                    """
                    INSERT INTO feedback_events(
                        creator_id, video_id, clip_id, event_type, start_seconds,
                        end_seconds, payload_json, created_at
                    ) VALUES (?, ?, ?, 'review_closed_unselected', ?, ?, ?, ?)
                    """,
                    (
                        video["creator_id"],
                        video_id,
                        clip["id"],
                        clip["start_seconds"],
                        clip["end_seconds"],
                        json.dumps(
                            {
                                "rank": clip["rank"],
                                "ranking_model_version": clip[
                                    "ranking_model_version"
                                ],
                                "reason": "review_closed_after_positive_selection",
                            },
                            sort_keys=True,
                        ),
                        utc_now(),
                    ),
                )
        return len(unresolved)

    def get_video(self, video_id: str) -> dict[str, Any]:
        """Return one upload and its ranked clips without exposing its local media path."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
            if row is None:
                raise KeyError(video_id)
            video = dict(row)
            clips = connection.execute(
                "SELECT * FROM clips WHERE video_id = ? ORDER BY rank", (video_id,)
            ).fetchall()
        video.pop("media_path")
        video["publishability_summary"] = json.loads(
            video.pop("publishability_summary_json", "{}")
        )
        video["clips"] = [self._public_clip(dict(clip)) for clip in clips]
        video["job"] = self.processing_job_for_video(video_id)
        video["source_analytics"] = self.source_analytics_summary(video_id)
        return video

    def media_path_for_video(self, video_id: str) -> Path:
        """Resolve the private local path for a registered source."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT media_path FROM videos WHERE id = ?", (video_id,)
            ).fetchone()
        if row is None:
            raise KeyError(video_id)
        return Path(row["media_path"])

    def list_videos(self, creator_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """List recent creator runs without exposing private filesystem paths."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, original_filename, status, duration_seconds, created_at, updated_at
                FROM videos
                WHERE creator_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (creator_id, max(1, min(limit, 50))),
            ).fetchall()
        return [dict(row) for row in rows]

    def creator_ml_report(self, creator_id: str) -> dict[str, Any]:
        """Summarize serving, lineage, labels, edits, and adaptation for one creator."""
        with self._connect() as connection:
            creator = connection.execute(
                "SELECT id, display_name, created_at FROM creator_profiles WHERE id = ?",
                (creator_id,),
            ).fetchone()
            if creator is None:
                raise KeyError(creator_id)
            video_rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count FROM videos
                WHERE creator_id = ? GROUP BY status
                """,
                (creator_id,),
            ).fetchall()
            publishability_rows = connection.execute(
                """
                SELECT publishability_summary_json FROM videos
                WHERE creator_id = ? AND publishability_summary_json != '{}'
                """,
                (creator_id,),
            ).fetchall()
            job_rows = connection.execute(
                """
                SELECT processing_jobs.status, COUNT(*) AS count
                FROM processing_jobs
                JOIN videos ON videos.id = processing_jobs.video_id
                WHERE videos.creator_id = ?
                GROUP BY processing_jobs.status
                """,
                (creator_id,),
            ).fetchall()
            event_rows = connection.execute(
                """
                SELECT event_type, COUNT(*) AS count,
                       COUNT(DISTINCT clip_id) AS clip_count
                FROM feedback_events
                WHERE creator_id = ? GROUP BY event_type
                """,
                (creator_id,),
            ).fetchall()
            clip_totals = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN origin = 'model' THEN 1 ELSE 0 END) AS model_total,
                       SUM(CASE WHEN origin = 'creator' THEN 1 ELSE 0 END) AS custom_total
                FROM clips
                JOIN videos ON videos.id = clips.video_id
                WHERE videos.creator_id = ?
                """,
                (creator_id,),
            ).fetchone()
            selected_model_clips = connection.execute(
                """
                SELECT COUNT(DISTINCT feedback_events.clip_id)
                FROM feedback_events
                JOIN clips ON clips.id = feedback_events.clip_id
                WHERE feedback_events.creator_id = ? AND clips.origin = 'model'
                  AND feedback_events.event_type IN ('download_original', 'download_edited')
                """,
                (creator_id,),
            ).fetchone()[0]
            edits = connection.execute(
                """
                SELECT feedback_events.start_seconds, feedback_events.end_seconds,
                       clips.start_seconds AS proposed_start,
                       clips.end_seconds AS proposed_end
                FROM feedback_events
                JOIN clips ON clips.id = feedback_events.clip_id
                WHERE feedback_events.creator_id = ?
                  AND feedback_events.event_type = 'download_edited'
                """,
                (creator_id,),
            ).fetchall()
            lineage_rows = connection.execute(
                """
                SELECT COALESCE(clips.ranking_model_version, 'unversioned') AS model_version,
                       COUNT(*) AS clip_count,
                       COUNT(DISTINCT clips.video_id) AS video_count,
                       MIN(clips.created_at) AS first_seen,
                       MAX(clips.created_at) AS last_seen
                FROM clips
                JOIN videos ON videos.id = clips.video_id
                WHERE videos.creator_id = ? AND clips.origin = 'model'
                GROUP BY COALESCE(clips.ranking_model_version, 'unversioned')
                ORDER BY last_seen DESC
                """,
                (creator_id,),
            ).fetchall()
            adjustment_row = connection.execute(
                """
                SELECT AVG(ABS(editorial_adjustment)) AS editorial_mean_absolute,
                       MAX(ABS(editorial_adjustment)) AS editorial_max_absolute,
                       AVG(ABS(performance_adjustment)) AS performance_mean_absolute,
                       MAX(ABS(performance_adjustment)) AS performance_max_absolute,
                       AVG(ABS(semantic_performance_adjustment)) AS semantic_mean_absolute,
                       MAX(ABS(semantic_performance_adjustment)) AS semantic_max_absolute,
                       AVG(ABS(source_retention_adjustment)) AS retention_mean_absolute,
                       MAX(ABS(source_retention_adjustment)) AS retention_max_absolute,
                       AVG(ABS(publishability_adjustment)) AS publishability_mean_absolute,
                       MAX(ABS(publishability_adjustment)) AS publishability_max_absolute
                FROM clips
                JOIN videos ON videos.id = clips.video_id
                WHERE videos.creator_id = ? AND clips.origin = 'model'
                """,
                (creator_id,),
            ).fetchone()
            analytics_count = connection.execute(
                "SELECT COUNT(*) FROM analytics_imports WHERE creator_id = ?",
                (creator_id,),
            ).fetchone()[0]
            performance_count = connection.execute(
                "SELECT COUNT(*) FROM performance_reports WHERE creator_id = ?",
                (creator_id,),
            ).fetchone()[0]
            repurposing_pack_count = connection.execute(
                "SELECT COUNT(*) FROM repurposing_packs WHERE creator_id = ?",
                (creator_id,),
            ).fetchone()[0]
            repurposing_feedback_rows = connection.execute(
                """
                SELECT action, COUNT(*) AS count FROM repurposing_feedback
                WHERE creator_id = ? GROUP BY action
                """,
                (creator_id,),
            ).fetchall()
            recent_rows = connection.execute(
                """
                SELECT feedback_events.event_type, feedback_events.start_seconds,
                       feedback_events.end_seconds, feedback_events.created_at,
                       clips.rank, clips.origin, videos.original_filename
                FROM feedback_events
                JOIN clips ON clips.id = feedback_events.clip_id
                JOIN videos ON videos.id = feedback_events.video_id
                WHERE feedback_events.creator_id = ?
                ORDER BY feedback_events.id DESC LIMIT 20
                """,
                (creator_id,),
            ).fetchall()

        video_counts = {row["status"]: row["count"] for row in video_rows}
        job_counts = {row["status"]: row["count"] for row in job_rows}
        event_counts = {
            row["event_type"]: {
                "events": row["count"],
                "distinct_clips": row["clip_count"],
            }
            for row in event_rows
        }
        presented_count = event_counts.get("presented", {}).get("distinct_clips", 0)
        absolute_boundary_changes = [
            abs(float(row["start_seconds"]) - float(row["proposed_start"]))
            + abs(float(row["end_seconds"]) - float(row["proposed_end"]))
            for row in edits
        ]
        creator_summary = self.creator_summary(creator_id)
        repurposing_feedback_counts = {
            row["action"]: row["count"] for row in repurposing_feedback_rows
        }
        publishability_summaries = [
            json.loads(row["publishability_summary_json"])
            for row in publishability_rows
        ]
        return {
            "report_schema": "creatorcut_ml_operations_report_v1",
            "generated_at": utc_now(),
            "creator": dict(creator),
            "serving": {
                "video_count": sum(video_counts.values()),
                "video_status_counts": video_counts,
                "job_status_counts": job_counts,
            },
            "feedback": {
                "clip_count": int(clip_totals["total"] or 0),
                "model_clip_count": int(clip_totals["model_total"] or 0),
                "custom_clip_count": int(clip_totals["custom_total"] or 0),
                "presented_model_clip_count": presented_count,
                "selected_model_clip_count": selected_model_clips,
                "model_clip_selection_rate": round(
                    selected_model_clips / presented_count, 4
                )
                if presented_count
                else None,
                "edited_download_count": len(edits),
                "median_total_boundary_change_seconds": round(
                    median(absolute_boundary_changes), 3
                )
                if absolute_boundary_changes
                else None,
                "event_counts": event_counts,
                "analytics_import_count": analytics_count,
                "performance_report_count": performance_count,
                "repurposing_pack_count": repurposing_pack_count,
                "repurposing_feedback_counts": repurposing_feedback_counts,
            },
            "adaptation": creator_summary,
            "lineage": [dict(row) for row in lineage_rows],
            "adjustments": {
                key: round(float(value or 0.0), 4)
                for key, value in dict(adjustment_row).items()
            },
            "publishability": {
                "assessed_video_count": len(publishability_summaries),
                "candidate_count": sum(
                    int(summary.get("candidate_count", 0))
                    for summary in publishability_summaries
                ),
                "blocked_candidate_count": sum(
                    int(summary.get("blocked_count", 0))
                    for summary in publishability_summaries
                ),
                "rule_versions": sorted(
                    {
                        summary["rule_version"]
                        for summary in publishability_summaries
                        if summary.get("rule_version")
                    }
                ),
            },
            "recent_events": [dict(row) for row in recent_rows],
            "global_model_policy": {
                "frozen": True,
                "production_feedback_auto_trains_global_model": False,
                "promotion_requires_unseen_video_evaluation": True,
            },
        }

    def admin_ml_report(self) -> dict[str, Any]:
        """Aggregate privacy-conscious serving and feedback signals across creators."""
        with self._connect() as connection:
            creators = connection.execute(
                """
                SELECT id, display_name, created_at FROM creator_profiles
                ORDER BY created_at, id
                """
            ).fetchall()
            credential_rows = connection.execute(
                """
                SELECT creator_id, email, is_admin, created_at
                FROM account_credentials ORDER BY created_at, creator_id
                """
            ).fetchall()
            videos = connection.execute(
                """
                SELECT id, creator_id, original_filename, status, duration_seconds
                FROM videos ORDER BY created_at, id
                """
            ).fetchall()
            clips = connection.execute(
                """
                SELECT clips.id, videos.creator_id, clips.rank, clips.origin,
                       clips.duration_seconds, clips.global_score,
                       clips.personalized_score
                FROM clips JOIN videos ON videos.id = clips.video_id
                ORDER BY clips.created_at, clips.id
                """
            ).fetchall()
            events = connection.execute(
                """
                SELECT feedback_events.creator_id, feedback_events.clip_id,
                       feedback_events.event_type, feedback_events.start_seconds,
                       feedback_events.end_seconds, feedback_events.payload_json,
                       clips.rank, clips.origin,
                       clips.start_seconds AS proposed_start,
                       clips.end_seconds AS proposed_end
                FROM feedback_events
                JOIN clips ON clips.id = feedback_events.clip_id
                ORDER BY feedback_events.id
                """
            ).fetchall()
            analytics = connection.execute(
                """
                SELECT creator_id, report_role, original_filename, report_json,
                       recognized_row_count
                FROM analytics_imports ORDER BY created_at, id
                """
            ).fetchall()
            performance = connection.execute(
                "SELECT * FROM performance_reports ORDER BY id"
            ).fetchall()
            jobs = connection.execute(
                "SELECT status, attempt_count FROM processing_jobs ORDER BY id"
            ).fetchall()
            repurposing = connection.execute(
                """
                SELECT creator_id, action, platform FROM repurposing_feedback
                ORDER BY id
                """
            ).fetchall()

        credentials = {row["creator_id"]: row for row in credential_rows}
        accounts: dict[str, dict[str, Any]] = {
            row["id"]: {
                "creator_id": row["id"],
                "display_name": row["display_name"],
                "created_at": row["created_at"],
                "account_status": (
                    "authenticated" if row["id"] in credentials else "legacy_local_profile"
                ),
                "email": credentials[row["id"]]["email"]
                if row["id"] in credentials
                else None,
                "role": (
                    "administrator"
                    if row["id"] in credentials and credentials[row["id"]]["is_admin"]
                    else "creator"
                ),
                "video_count": 0,
                "model_clip_count": 0,
                "custom_clip_count": 0,
                "presented_clip_ids": set(),
                "selected_clip_ids": set(),
                "rejected_clip_ids": set(),
                "edited_download_count": 0,
                "total_boundary_changes": [],
                "analytics_import_count": 0,
                "performance_report_count": 0,
                "repurposing_feedback_count": 0,
            }
            for row in creators
        }
        source_extensions: Counter[str] = Counter()
        video_statuses: Counter[str] = Counter()
        source_durations: list[float] = []
        for row in videos:
            accounts[row["creator_id"]]["video_count"] += 1
            suffix = Path(row["original_filename"]).suffix.casefold() or "no_extension"
            source_extensions[suffix] += 1
            video_statuses[row["status"]] += 1
            if row["duration_seconds"] is not None:
                source_durations.append(float(row["duration_seconds"]))

        clip_origins: Counter[str] = Counter()
        clip_durations: list[float] = []
        global_scores: list[float] = []
        personalized_scores: list[float] = []
        for row in clips:
            account = accounts[row["creator_id"]]
            key = "custom_clip_count" if row["origin"] == "creator" else "model_clip_count"
            account[key] += 1
            clip_origins[row["origin"]] += 1
            clip_durations.append(float(row["duration_seconds"]))
            global_scores.append(float(row["global_score"]))
            personalized_scores.append(float(row["personalized_score"]))

        presented_by_rank: dict[int, set[str]] = {}
        selected_by_rank: dict[int, set[str]] = {}
        event_counts: Counter[str] = Counter()
        export_formats: Counter[str] = Counter()
        total_boundary_changes: list[float] = []
        start_boundary_changes: list[float] = []
        end_boundary_changes: list[float] = []
        selected_durations: list[float] = []
        for row in events:
            account = accounts[row["creator_id"]]
            event_type = row["event_type"]
            event_counts[event_type] += 1
            if row["origin"] == "model" and event_type == "presented":
                account["presented_clip_ids"].add(row["clip_id"])
                presented_by_rank.setdefault(int(row["rank"]), set()).add(row["clip_id"])
            if row["origin"] == "model" and event_type in {
                "download_original",
                "download_edited",
            }:
                account["selected_clip_ids"].add(row["clip_id"])
                selected_by_rank.setdefault(int(row["rank"]), set()).add(row["clip_id"])
            if event_type in {"download_original", "download_edited"}:
                selected_durations.append(
                    float(row["end_seconds"]) - float(row["start_seconds"])
                )
            if row["origin"] == "model" and event_type == "reject":
                account["rejected_clip_ids"].add(row["clip_id"])
            payload = json.loads(row["payload_json"])
            export_format = payload.get("export_format")
            if isinstance(export_format, str):
                export_formats[export_format] += 1
            if event_type == "download_edited":
                start_change = abs(float(row["start_seconds"]) - float(row["proposed_start"]))
                end_change = abs(float(row["end_seconds"]) - float(row["proposed_end"]))
                total_change = start_change + end_change
                account["edited_download_count"] += 1
                account["total_boundary_changes"].append(total_change)
                start_boundary_changes.append(start_change)
                end_boundary_changes.append(end_change)
                total_boundary_changes.append(total_change)

        analytics_extensions: Counter[str] = Counter()
        analytics_roles: Counter[str] = Counter()
        analytics_report_types: Counter[str] = Counter()
        analytics_metric_coverage: Counter[str] = Counter()
        retention_point_counts: list[float] = []
        recognized_analytics_rows = 0
        for row in analytics:
            accounts[row["creator_id"]]["analytics_import_count"] += 1
            analytics_extensions[
                Path(row["original_filename"]).suffix.casefold() or "no_extension"
            ] += 1
            analytics_roles[row["report_role"]] += 1
            recognized_analytics_rows += int(row["recognized_row_count"])
            report = json.loads(row["report_json"])
            for report_type in report.get("report_types", []):
                analytics_report_types[str(report_type)] += 1
            for metric in report.get("totals", {}):
                analytics_metric_coverage[str(metric)] += 1
            retention_point_counts.append(len(report.get("retention_rows", [])))

        performance_metric_fields = (
            "views",
            "engaged_views",
            "watch_time_hours",
            "average_view_duration_seconds",
            "average_view_percentage",
            "likes",
            "comments",
            "shares",
            "subscribers_net",
            "shown_in_feed",
            "chose_to_view_percentage",
            "thumbnail_impressions",
            "thumbnail_ctr",
        )
        performance_metric_coverage: Counter[str] = Counter()
        performance_metric_values: dict[str, list[float]] = {
            field: [] for field in performance_metric_fields
        }
        for row in performance:
            accounts[row["creator_id"]]["performance_report_count"] += 1
            for field in performance_metric_fields:
                if row[field] is not None:
                    performance_metric_coverage[field] += 1
                    performance_metric_values[field].append(float(row[field]))
        repurposing_actions: Counter[str] = Counter()
        repurposing_platforms: Counter[str] = Counter()
        for row in repurposing:
            accounts[row["creator_id"]]["repurposing_feedback_count"] += 1
            repurposing_actions[row["action"]] += 1
            repurposing_platforms[row["platform"]] += 1

        account_rows: list[dict[str, Any]] = []
        for account in accounts.values():
            presented_count = len(account.pop("presented_clip_ids"))
            selected_count = len(account.pop("selected_clip_ids"))
            rejected_count = len(account.pop("rejected_clip_ids"))
            edit_values = account.pop("total_boundary_changes")
            account_rows.append(
                {
                    **account,
                    "presented_model_clip_count": presented_count,
                    "selected_model_clip_count": selected_count,
                    "rejected_model_clip_count": rejected_count,
                    "selection_rate": round(selected_count / presented_count, 4)
                    if presented_count
                    else None,
                    "median_total_boundary_change_seconds": (
                        numeric_summary(edit_values)["median"]
                    ),
                }
            )

        total_presented = sum(row["presented_model_clip_count"] for row in account_rows)
        total_selected = sum(row["selected_model_clip_count"] for row in account_rows)
        rank_performance = []
        for rank in sorted(set(presented_by_rank) | set(selected_by_rank)):
            presented_count = len(presented_by_rank.get(rank, set()))
            selected_count = len(selected_by_rank.get(rank, set()))
            rank_performance.append(
                {
                    "rank": rank,
                    "presented": presented_count,
                    "selected": selected_count,
                    "selection_rate": round(selected_count / presented_count, 4)
                    if presented_count
                    else None,
                }
            )
        return {
            "schema": "creatorcut_admin_ml_observatory_v1",
            "generated_at": utc_now(),
            "scope": {
                "creator_account_count": len(account_rows),
                "authenticated_account_count": len(credential_rows),
                "legacy_profile_count": len(account_rows) - len(credential_rows),
                "source_video_count": len(videos),
                "clip_count": len(clips),
            },
            "model_behavior": {
                "presented_model_clip_count": total_presented,
                "selected_model_clip_count": total_selected,
                "selection_rate": round(total_selected / total_presented, 4)
                if total_presented
                else None,
                "event_counts": dict(sorted(event_counts.items())),
                "selection_by_display_rank": rank_performance,
                "global_score_distribution": numeric_summary(global_scores),
                "personalized_score_distribution": numeric_summary(personalized_scores),
            },
            "media": {
                "source_file_types": dict(sorted(source_extensions.items())),
                "video_status_counts": dict(sorted(video_statuses.items())),
                "source_duration_seconds": numeric_summary(source_durations),
                "clip_origins": dict(sorted(clip_origins.items())),
                "clip_duration_seconds": numeric_summary(clip_durations),
                "selected_duration_seconds": numeric_summary(selected_durations),
                "export_format_counts": dict(sorted(export_formats.items())),
            },
            "editing": {
                "edited_download_count": len(total_boundary_changes),
                "absolute_start_change_seconds": numeric_summary(start_boundary_changes),
                "absolute_end_change_seconds": numeric_summary(end_boundary_changes),
                "total_boundary_change_seconds": numeric_summary(total_boundary_changes),
            },
            "analytics": {
                "import_count": len(analytics),
                "file_types": dict(sorted(analytics_extensions.items())),
                "report_roles": dict(sorted(analytics_roles.items())),
                "report_types": dict(sorted(analytics_report_types.items())),
                "recognized_row_count": recognized_analytics_rows,
                "metric_coverage": dict(sorted(analytics_metric_coverage.items())),
                "retention_points_per_import": numeric_summary(retention_point_counts),
                "performance_report_count": len(performance),
                "performance_metric_coverage": dict(
                    sorted(performance_metric_coverage.items())
                ),
                "performance_metric_distributions": {
                    field: numeric_summary(values)
                    for field, values in performance_metric_values.items()
                    if values
                },
            },
            "repurposing": {
                "feedback_count": len(repurposing),
                "action_counts": dict(sorted(repurposing_actions.items())),
                "platform_counts": dict(sorted(repurposing_platforms.items())),
            },
            "operations": {
                "job_status_counts": dict(
                    sorted(Counter(row["status"] for row in jobs).items())
                ),
                "job_attempt_distribution": numeric_summary(
                    [float(row["attempt_count"]) for row in jobs]
                ),
            },
            "accounts": account_rows,
            "interpretation": {
                "selection_rate": (
                    "An explicit product-choice signal, not ground-truth model accuracy."
                ),
                "analytics": (
                    "Coverage shows which fields were available; missing metrics remain missing."
                ),
                "privacy": (
                    "This aggregate omits transcript text, media paths, and raw analytics rows."
                ),
            },
        }

    def get_clip(self, clip_id: str) -> dict[str, Any]:
        """Return one ranked clip plus its owning creator and private source path."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT clips.*, videos.creator_id, videos.media_path
                FROM clips JOIN videos ON videos.id = clips.video_id
                WHERE clips.id = ?
                """,
                (clip_id,),
            ).fetchone()
        if row is None:
            raise KeyError(clip_id)
        clip = dict(row)
        clip["predicted_targets"] = json.loads(clip.pop("predicted_targets_json"))
        return clip

    def creator_clip_documents(self, creator_id: str) -> list[str]:
        """Return saved clip transcripts as the creator-local topic corpus."""
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM creator_profiles WHERE id = ?", (creator_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(creator_id)
            rows = connection.execute(
                """
                SELECT clips.transcript_text
                FROM clips JOIN videos ON videos.id = clips.video_id
                WHERE videos.creator_id = ? AND TRIM(clips.transcript_text) != ''
                ORDER BY clips.created_at, clips.id
                """,
                (creator_id,),
            ).fetchall()
        return [str(row["transcript_text"]) for row in rows]

    def save_repurposing_pack(
        self, clip_id: str, pack: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist one generated platform pack with its algorithm lineage."""
        clip = self.get_clip(clip_id)
        algorithm_version = pack.get("algorithm_version")
        if not isinstance(algorithm_version, str) or not algorithm_version:
            raise ValueError("The repurposing pack must identify its algorithm version")
        pack_id = f"repurpose_{uuid.uuid4().hex}"
        created_at = utc_now()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO repurposing_packs(
                    id, creator_id, clip_id, algorithm_version, pack_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    pack_id,
                    clip["creator_id"],
                    clip_id,
                    algorithm_version,
                    json.dumps(pack, sort_keys=True),
                    created_at,
                ),
            )
        return {"id": pack_id, "clip_id": clip_id, "created_at": created_at, **pack}

    def record_repurposing_feedback(
        self,
        clip_id: str,
        pack_id: str,
        platform: str,
        action: str,
        generated_text: str,
        final_text: str | None,
    ) -> dict[str, Any]:
        """Record whether generated platform copy was kept, edited, or rejected."""
        if platform not in {"youtube", "tiktok", "instagram", "linkedin"}:
            raise ValueError("Choose a supported repurposing platform")
        if action not in {"copied_original", "copied_edited", "rejected"}:
            raise ValueError("Choose a supported repurposing action")
        if not isinstance(generated_text, str) or not generated_text.strip():
            raise ValueError("Generated text is required")
        if action.startswith("copied") and (
            not isinstance(final_text, str) or not final_text.strip()
        ):
            raise ValueError("Final text is required when copy is selected")
        clip = self.get_clip(clip_id)
        created_at = utc_now()
        with self._write_lock, self._connect() as connection:
            pack = connection.execute(
                """
                SELECT id, pack_json FROM repurposing_packs
                WHERE id = ? AND clip_id = ? AND creator_id = ?
                """,
                (pack_id, clip_id, clip["creator_id"]),
            ).fetchone()
            if pack is None:
                raise ValueError("The repurposing pack does not belong to this clip")
            stored_pack = json.loads(pack["pack_json"])
            expected_text = (
                stored_pack.get("platforms", {}).get(platform, {}).get("text")
            )
            if generated_text != expected_text:
                raise ValueError("Generated text does not match the stored pack")
            if action == "copied_original" and final_text != generated_text:
                raise ValueError("Edited copy must be recorded as copied_edited")
            if action == "copied_edited" and final_text == generated_text:
                raise ValueError("Unchanged copy must be recorded as copied_original")
            cursor = connection.execute(
                """
                INSERT INTO repurposing_feedback(
                    creator_id, clip_id, pack_id, platform, action,
                    generated_text, final_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clip["creator_id"],
                    clip_id,
                    pack_id,
                    platform,
                    action,
                    generated_text.strip(),
                    final_text.strip() if isinstance(final_text, str) else None,
                    created_at,
                ),
            )
        return {
            "id": cursor.lastrowid,
            "clip_id": clip_id,
            "pack_id": pack_id,
            "platform": platform,
            "action": action,
            "created_at": created_at,
        }

    def save_clip_semantic_embedding(
        self, clip_id: str, embedding: list[float]
    ) -> None:
        """Backfill a frozen transcript embedding for recommendations made before this feature."""
        values = [float(item) for item in embedding]
        if not values or not all(math.isfinite(item) for item in values):
            raise ValueError("The semantic embedding is invalid")
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE clips SET semantic_embedding_json = ? WHERE id = ?",
                (json.dumps(values), clip_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(clip_id)

    def record_editorial_event(
        self,
        clip_id: str,
        event_type: str,
        start_seconds: float,
        end_seconds: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record a user decision with the exact interval that decision concerned."""
        if event_type not in EDITORIAL_EVENT_WEIGHTS:
            raise ValueError(f"Unsupported editorial event: {event_type}")
        clip = self.get_clip(clip_id)
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback_events(
                    creator_id, video_id, clip_id, event_type, start_seconds,
                    end_seconds, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clip["creator_id"],
                    clip["video_id"],
                    clip_id,
                    event_type,
                    start_seconds,
                    end_seconds,
                    json.dumps(payload or {}, sort_keys=True),
                    utc_now(),
                ),
            )

    def save_performance_report(
        self,
        clip_id: str,
        value: dict[str, Any],
        analytics_import_id: str | None = None,
    ) -> None:
        """Store post-publication outcomes separately from editorial preferences."""
        clip = self.get_clip(clip_id)
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO performance_reports(
                    creator_id, clip_id, platform, views, likes, comments, shares,
                    average_view_percentage, published_at, created_at, engaged_views,
                    watch_time_hours, average_view_duration_seconds, subscribers_gained,
                    subscribers_lost, subscribers_net, shown_in_feed, chose_to_view_percentage,
                    thumbnail_impressions, thumbnail_ctr, analytics_import_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clip["creator_id"],
                    clip_id,
                    value["platform"],
                    value.get("views"),
                    value.get("likes"),
                    value.get("comments"),
                    value.get("shares"),
                    value.get("average_view_percentage"),
                    value.get("published_at"),
                    utc_now(),
                    value.get("engaged_views"),
                    value.get("watch_time_hours"),
                    value.get("average_view_duration_seconds"),
                    value.get("subscribers_gained"),
                    value.get("subscribers_lost"),
                    value.get("subscribers_net"),
                    value.get("shown_in_feed"),
                    value.get("chose_to_view_percentage"),
                    value.get("thumbnail_impressions"),
                    value.get("thumbnail_ctr"),
                    analytics_import_id,
                ),
            )

    def save_analytics_import(
        self,
        creator_id: str,
        original_filename: str,
        report_role: str,
        report: dict[str, Any],
        video_id: str | None = None,
        clip_id: str | None = None,
        platform: str = "youtube",
    ) -> dict[str, Any]:
        """Persist a parsed analytics export and link it to a source or published clip."""
        if report_role not in {"source_video", "published_clip"}:
            raise ValueError("Unsupported analytics report role")
        if report_role == "source_video" and (video_id is None or clip_id is not None):
            raise ValueError("Source analytics must be linked to one uploaded video")
        if report_role == "published_clip" and clip_id is None:
            raise ValueError("Short-form analytics must be linked to one clip")
        report_video_ids = {
            str(row["video_id"])
            for row in report.get("report_rows", [])
            if row.get("video_id")
        }
        report_contents = {
            str(row["content"])
            for row in report.get("report_rows", [])
            if row.get("content") and str(row["content"]).casefold() != "total"
        }
        if len(report_video_ids) > 1 or (not report_video_ids and len(report_contents) > 1):
            raise ValueError("Choose a YouTube analytics export filtered to one video or Short")

        if clip_id is not None:
            clip = self.get_clip(clip_id)
            if clip["creator_id"] != creator_id:
                raise ValueError("The analytics creator does not own this clip")
            video_id = clip["video_id"]
        elif video_id is not None:
            video = self.get_video(video_id)
            if video["creator_id"] != creator_id:
                raise ValueError("The analytics creator does not own this video")

        import_id = f"analytics_{uuid.uuid4().hex}"
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analytics_imports(
                    id, creator_id, video_id, clip_id, report_role, platform,
                    original_filename, report_json, recognized_row_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    import_id,
                    creator_id,
                    video_id,
                    clip_id,
                    report_role,
                    platform,
                    Path(original_filename).name,
                    json.dumps(report, sort_keys=True),
                    int(report.get("recognized_row_count", 0)),
                    utc_now(),
                ),
            )

        values = performance_values(report)
        if report_role == "published_clip" and any(
            metric is not None for metric in values.values()
        ):
            self.save_performance_report(
                clip_id,
                {"platform": platform, "published_at": None, **values},
                analytics_import_id=import_id,
            )
        return self.analytics_import_summary(import_id)

    def analytics_import_summary(self, import_id: str) -> dict[str, Any]:
        """Return useful import evidence without exposing every raw analytics row."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM analytics_imports WHERE id = ?", (import_id,)
            ).fetchone()
        if row is None:
            raise KeyError(import_id)
        value = dict(row)
        report = json.loads(value.pop("report_json"))
        totals = report.get("totals", {})
        exposure = totals.get("engaged_views", totals.get("views"))
        if value["report_role"] == "source_video":
            active = (
                len(report.get("retention_rows", [])) >= MINIMUM_SOURCE_RETENTION_POINTS
                and exposure is not None
                and exposure >= MINIMUM_SOURCE_RETENTION_VIEWS
            )
            status = (
                "Source retention is available for bounded ranking adjustments."
                if active
                else "Saved for context; more exposure or a retention curve is needed."
            )
        else:
            values = performance_values(report)
            has_outcome = any(
                values.get(field) is not None
                for field in (
                    "average_view_percentage",
                    "average_view_duration_seconds",
                    "chose_to_view_percentage",
                    "likes",
                    "comments",
                    "shares",
                    "subscribers_gained",
                    "subscribers_lost",
                    "subscribers_net",
                )
            )
            active = (
                exposure is not None
                and exposure >= MINIMUM_PERFORMANCE_VIEWS
                and has_outcome
            )
            status = (
                "This report can contribute after five comparable Shorts are available."
                if active
                else "Saved, but it needs enough exposure and at least one outcome metric."
            )
        return {
            "id": value["id"],
            "report_role": value["report_role"],
            "platform": value["platform"],
            "original_filename": value["original_filename"],
            "recognized_row_count": value["recognized_row_count"],
            "retention_point_count": len(report.get("retention_rows", [])),
            "totals": totals,
            "learning_eligible": active,
            "learning_status": status,
            "warnings": report.get("warnings", []),
            "created_at": value["created_at"],
        }

    def source_analytics_summary(self, video_id: str) -> dict[str, Any] | None:
        """Return the latest analytics import linked to an uploaded source video."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id FROM analytics_imports
                WHERE video_id = ? AND report_role = 'source_video'
                ORDER BY created_at DESC LIMIT 1
                """,
                (video_id,),
            ).fetchone()
        return self.analytics_import_summary(row["id"]) if row is not None else None

    @staticmethod
    def _centered_percentile(value: float, values: list[float]) -> float:
        if len(values) < 2 or max(values) == min(values):
            return 0.0
        below = sum(candidate < value for candidate in values)
        equal = sum(candidate == value for candidate in values)
        rank = below + (equal - 1) / 2
        return 2 * rank / (len(values) - 1) - 1

    @staticmethod
    def _performance_components(row: dict[str, Any]) -> dict[str, float]:
        exposure = row.get("engaged_views") or row.get("views")
        if exposure is None or exposure < MINIMUM_PERFORMANCE_VIEWS:
            return {}
        components: dict[str, float] = {}
        for field in ("average_view_percentage", "chose_to_view_percentage"):
            if row.get(field) is not None:
                components[field] = float(row[field]) / 100.0
        if row.get("average_view_duration_seconds") is not None and row.get(
            "duration_seconds"
        ):
            components["duration_fraction"] = min(
                2.0,
                float(row["average_view_duration_seconds"]) / float(row["duration_seconds"]),
            )
        for field in ("likes", "comments", "shares"):
            if row.get(field) is not None:
                components[f"{field}_per_view"] = float(row[field]) / exposure
        gained = row.get("subscribers_gained")
        lost = row.get("subscribers_lost")
        net = row.get("subscribers_net")
        if gained is not None or lost is not None or net is not None:
            components["net_subscribers_per_view"] = (
                float(net) if net is not None else float(gained or 0) - float(lost or 0)
            ) / exposure
        return components

    @staticmethod
    def _semantic_vector(value: str | None) -> list[float] | None:
        if not value:
            return None
        try:
            vector = [float(item) for item in json.loads(value)]
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not vector or not all(math.isfinite(item) for item in vector):
            return None
        norm = math.sqrt(sum(item * item for item in vector))
        if norm <= 0:
            return None
        return [item / norm for item in vector]

    @staticmethod
    def _semantic_terms(text: str) -> set[str]:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", text.casefold())
            if token not in SEMANTIC_TREND_STOPWORDS
        ]
        terms = set(tokens)
        terms.update(
            f"{left} {right}" for left, right in zip(tokens, tokens[1:], strict=False)
        )
        return terms

    @classmethod
    def _positive_semantic_trends(
        cls, examples: list[dict[str, Any]], targets: dict[str, float]
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        support: dict[str, int] = {}
        for row in examples:
            outcome = targets.get(row["clip_id"])
            if outcome is None:
                continue
            for term in cls._semantic_terms(str(row.get("transcript_text", ""))):
                scores[term] = scores.get(term, 0.0) + outcome
                support[term] = support.get(term, 0) + 1
        ranked = sorted(
            (
                (term, score, support[term])
                for term, score in scores.items()
                if support[term] >= 2 and score > 0
            ),
            key=lambda item: (-item[1], -item[2], item[0]),
        )
        return [
            {"term": term, "support": count, "association": round(score, 3)}
            for term, score, count in ranked[:6]
        ]

    @classmethod
    def _negative_semantic_trends(
        cls, examples: list[dict[str, Any]], targets: dict[str, float]
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        support: dict[str, int] = {}
        for row in examples:
            outcome = targets.get(row["clip_id"])
            if outcome is None:
                continue
            for term in cls._semantic_terms(str(row.get("transcript_text", ""))):
                scores[term] = scores.get(term, 0.0) + outcome
                support[term] = support.get(term, 0) + 1
        ranked = sorted(
            (
                (term, score, support[term])
                for term, score in scores.items()
                if support[term] >= 2 and score < 0
            ),
            key=lambda item: (item[1], -item[2], item[0]),
        )
        return [
            {"term": term, "support": count, "association": round(score, 3)}
            for term, score, count in ranked[:6]
        ]

    @staticmethod
    def _performance_insight_summary(
        example_count: int,
        weights: list[float],
        positive_terms: list[dict[str, Any]],
        negative_terms: list[dict[str, Any]],
    ) -> str | None:
        if example_count < MINIMUM_PERFORMANCE_EXAMPLES:
            return None
        feature_phrases = {
            "hook": ("higher predicted hook", "lower predicted hook"),
            "completeness": (
                "higher predicted completeness",
                "lower predicted completeness",
            ),
            "payoff": ("higher predicted payoff", "lower predicted payoff"),
            "clarity": ("higher predicted clarity", "lower predicted clarity"),
            "duration": ("longer clips", "shorter clips"),
        }
        ranked_features = sorted(
            zip(PREFERENCE_FEATURE_NAMES, weights, strict=True),
            key=lambda item: (-abs(item[1]), item[0]),
        )
        associated_features = [
            feature_phrases[field][0 if weight > 0 else 1]
            for field, weight in ranked_features
            if abs(weight) > 1e-6
        ][:2]
        stronger = list(associated_features)
        if positive_terms:
            stronger.append(
                "themes such as " + ", ".join(item["term"] for item in positive_terms[:3])
            )
        sentences = [
            f"Across {example_count} eligible Shorts, stronger outcomes are currently associated "
            + ("with " + "; ".join(stronger) if stronger else "with no stable feature yet")
            + "."
        ]
        if negative_terms:
            sentences.append(
                "Lower-performing clips more often contain themes such as "
                + ", ".join(item["term"] for item in negative_terms[:3])
                + "."
            )
        sentences.append("These are within-channel associations, not causal claims.")
        return " ".join(sentences)

    @staticmethod
    def _semantic_outcome_adjustment(
        candidate_embedding: list[float] | None,
        examples: list[tuple[list[float], float]],
        shrinkage: float,
    ) -> float:
        if candidate_embedding is None or not examples or shrinkage <= 0:
            return 0.0
        norm = math.sqrt(sum(item * item for item in candidate_embedding))
        if norm <= 0 or not math.isfinite(norm):
            return 0.0
        normalized = [item / norm for item in candidate_embedding]
        if any(len(vector) != len(normalized) for vector, _ in examples):
            return 0.0
        similarities = [
            sum(left * right for left, right in zip(normalized, vector, strict=True))
            for vector, _ in examples
        ]
        maximum = max(similarities)
        weights = [math.exp((similarity - maximum) / 0.12) for similarity in similarities]
        predicted_outcome = sum(
            weight * outcome
            for weight, (_, outcome) in zip(weights, examples, strict=True)
        ) / sum(weights)
        return max(
            -MAXIMUM_SEMANTIC_PERFORMANCE_ADJUSTMENT,
            min(
                MAXIMUM_SEMANTIC_PERFORMANCE_ADJUSTMENT,
                MAXIMUM_SEMANTIC_PERFORMANCE_ADJUSTMENT
                * shrinkage
                * predicted_outcome,
            ),
        )

    def _performance_preference_profile(
        self, creator_id: str
    ) -> tuple[list[float], list[tuple[list[float], float]], dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT performance_reports.*,
                       COALESCE(
                           (
                               SELECT event.end_seconds - event.start_seconds
                               FROM feedback_events AS event
                               WHERE event.clip_id = performance_reports.clip_id
                                 AND event.event_type IN ('download_original', 'download_edited')
                               ORDER BY event.id DESC LIMIT 1
                           ),
                           clips.duration_seconds
                       ) AS duration_seconds,
                       clips.predicted_targets_json, clips.transcript_text,
                       clips.semantic_embedding_json
                FROM performance_reports
                JOIN clips ON clips.id = performance_reports.clip_id
                WHERE performance_reports.creator_id = ?
                  AND performance_reports.platform = 'youtube'
                ORDER BY performance_reports.id
                """,
                (creator_id,),
            ).fetchall()
        latest_by_clip = {row["clip_id"]: dict(row) for row in rows}
        examples = list(latest_by_clip.values())
        components_by_clip = {
            row["clip_id"]: self._performance_components(row) for row in examples
        }
        examples = [row for row in examples if components_by_clip[row["clip_id"]]]
        component_values: dict[str, list[float]] = {}
        for row in examples:
            for field, value in components_by_clip[row["clip_id"]].items():
                component_values.setdefault(field, []).append(value)

        targets: dict[str, float] = {}
        for row in examples:
            percentiles = [
                self._centered_percentile(value, component_values[field])
                for field, value in components_by_clip[row["clip_id"]].items()
                if len(component_values[field]) >= 2
            ]
            if percentiles:
                targets[row["clip_id"]] = sum(percentiles) / len(percentiles)

        eligible = [row for row in examples if row["clip_id"] in targets]
        active = (
            len(eligible) >= MINIMUM_PERFORMANCE_EXAMPLES
            and max(targets.values(), default=0.0) != min(targets.values(), default=0.0)
        )
        shrinkage = (
            len(eligible) / (len(eligible) + PERFORMANCE_PRIOR_STRENGTH) if active else 0.0
        )
        weights = [0.0] * len(PREFERENCE_FEATURE_NAMES)
        if active:
            for row in eligible:
                features = preference_features(
                    {
                        "duration_seconds": row["duration_seconds"],
                        "predicted_targets": json.loads(row["predicted_targets_json"]),
                    }
                )
                outcome = targets[row["clip_id"]]
                weights = [
                    weight + outcome * feature
                    for weight, feature in zip(weights, features, strict=True)
                ]
            weights = [shrinkage * weight / len(eligible) for weight in weights]
        semantic_examples = []
        if active:
            for row in eligible:
                vector = self._semantic_vector(row.get("semantic_embedding_json"))
                if vector is not None:
                    semantic_examples.append((vector, targets[row["clip_id"]]))
        semantic_example_count = len(semantic_examples)
        semantic_active = semantic_example_count >= MINIMUM_PERFORMANCE_EXAMPLES
        if not semantic_active:
            semantic_examples = []
        positive_semantic_trends = (
            self._positive_semantic_trends(eligible, targets) if semantic_active else []
        )
        negative_semantic_trends = (
            self._negative_semantic_trends(eligible, targets) if semantic_active else []
        )
        return weights, semantic_examples, {
            "active": active,
            "eligible_clip_count": len(eligible),
            "minimum_clip_count": MINIMUM_PERFORMANCE_EXAMPLES,
            "minimum_views": MINIMUM_PERFORMANCE_VIEWS,
            "shrinkage": shrinkage,
            "feature_weights": dict(zip(PREFERENCE_FEATURE_NAMES, weights, strict=True)),
            "semantic_active": semantic_active,
            "semantic_example_count": semantic_example_count,
            "positive_semantic_trends": positive_semantic_trends,
            "negative_semantic_trends": negative_semantic_trends,
            "insight_summary": self._performance_insight_summary(
                len(eligible),
                weights,
                positive_semantic_trends,
                negative_semantic_trends,
            ),
        }

    def personalize_candidates(
        self, creator_id: str, candidates: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply a bounded, shrinkage-weighted creator preference adjustment."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT feedback_events.id, feedback_events.clip_id,
                       feedback_events.event_type, clips.duration_seconds,
                       clips.predicted_targets_json
                FROM feedback_events JOIN clips ON clips.id = feedback_events.clip_id
                WHERE feedback_events.creator_id = ?
                  AND feedback_events.event_type IN (
                      'download_original', 'download_edited', 'custom_created',
                      'review_closed_unselected', 'reject'
                  )
                ORDER BY feedback_events.id
                """,
                (creator_id,),
            ).fetchall()

        latest_by_clip = {row["clip_id"]: dict(row) for row in rows}
        examples = list(latest_by_clip.values())
        weights = [0.0] * 5
        for example in examples:
            features = preference_features(
                {
                    "duration_seconds": example["duration_seconds"],
                    "predicted_targets": json.loads(example["predicted_targets_json"]),
                }
            )
            outcome = EDITORIAL_EVENT_WEIGHTS[example["event_type"]]
            weights = [
                weight + outcome * feature
                for weight, feature in zip(weights, features, strict=True)
            ]

        count = len(examples)
        active = count >= MINIMUM_EDITORIAL_DECISIONS
        shrinkage = count / (count + PERSONALIZATION_PRIOR_STRENGTH) if active else 0.0
        if active:
            weights = [shrinkage * weight / count for weight in weights]
        else:
            weights = [0.0] * len(weights)

        (
            performance_weights,
            semantic_examples,
            performance_metadata,
        ) = self._performance_preference_profile(creator_id)
        personalized: list[dict[str, Any]] = []
        for candidate in candidates:
            features = preference_features(candidate)
            raw_editorial_adjustment = sum(
                weight * feature for weight, feature in zip(weights, features, strict=True)
            ) / len(features)
            editorial_adjustment = max(
                -MAXIMUM_PERSONALIZATION_ADJUSTMENT,
                min(MAXIMUM_PERSONALIZATION_ADJUSTMENT, raw_editorial_adjustment),
            )
            raw_performance_adjustment = sum(
                weight * feature
                for weight, feature in zip(performance_weights, features, strict=True)
            ) / len(features)
            performance_adjustment = max(
                -MAXIMUM_STRUCTURED_PERFORMANCE_ADJUSTMENT,
                min(
                    MAXIMUM_STRUCTURED_PERFORMANCE_ADJUSTMENT,
                    raw_performance_adjustment,
                ),
            )
            semantic_performance_adjustment = self._semantic_outcome_adjustment(
                candidate.get("semantic_embedding"),
                semantic_examples,
                performance_metadata["shrinkage"],
            )
            combined_performance = performance_adjustment + semantic_performance_adjustment
            if abs(combined_performance) > MAXIMUM_PERFORMANCE_ADJUSTMENT:
                scale = MAXIMUM_PERFORMANCE_ADJUSTMENT / abs(combined_performance)
                performance_adjustment *= scale
                semantic_performance_adjustment *= scale
            personalized.append(
                {
                    **candidate,
                    "editorial_adjustment": editorial_adjustment,
                    "performance_adjustment": performance_adjustment,
                    "semantic_performance_adjustment": semantic_performance_adjustment,
                    "source_retention_adjustment": 0.0,
                    "personalized_score": (
                        float(candidate["global_score"])
                        + editorial_adjustment
                        + performance_adjustment
                        + semantic_performance_adjustment
                    ),
                }
            )
        return personalized, {
            "decision_count": count,
            "active": active,
            "shrinkage": shrinkage,
            "maximum_adjustment": MAXIMUM_PERSONALIZATION_ADJUSTMENT,
            "feature_weights": dict(zip(PREFERENCE_FEATURE_NAMES, weights, strict=True)),
            "performance": performance_metadata,
        }

    def apply_source_retention_signal(
        self,
        video_id: str,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply a small within-source retention boost when a reliable curve is available."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT report_json FROM analytics_imports
                WHERE video_id = ? AND report_role = 'source_video'
                ORDER BY created_at DESC LIMIT 1
                """,
                (video_id,),
            ).fetchone()
        if row is None:
            return candidates, {"active": False, "reason": "No source analytics imported."}
        report = json.loads(row["report_json"])
        points = report.get("retention_rows", [])
        totals = report.get("totals", {})
        exposure = totals.get("engaged_views", totals.get("views"))
        if len(points) < MINIMUM_SOURCE_RETENTION_POINTS or not exposure:
            return candidates, {
                "active": False,
                "reason": "A timestamped retention curve and source exposure are required.",
            }
        if exposure < MINIMUM_SOURCE_RETENTION_VIEWS:
            return candidates, {
                "active": False,
                "reason": f"At least {MINIMUM_SOURCE_RETENTION_VIEWS} source views are required.",
            }

        field = (
            "relative_retention_performance"
            if sum("relative_retention_performance" in point for point in points)
            >= MINIMUM_SOURCE_RETENTION_POINTS
            else "audience_watch_ratio"
        )
        usable = [float(point[field]) for point in points if field in point]
        if len(usable) < MINIMUM_SOURCE_RETENTION_POINTS:
            return candidates, {"active": False, "reason": "The retention curve is incomplete."}
        center = 0.5 if field == "relative_retention_performance" else median(usable)
        spread = math.sqrt(sum((value - center) ** 2 for value in usable) / len(usable))
        spread = max(spread, 0.05)
        video = self.get_video(video_id)
        duration = float(video["duration_seconds"] or 0)
        if duration <= 0:
            return candidates, {"active": False, "reason": "Source duration is unavailable."}
        adjusted: list[dict[str, Any]] = []
        for candidate in candidates:
            start_ratio = float(candidate["start_seconds"]) / duration
            end_ratio = float(candidate["end_seconds"]) / duration
            clip_values = [
                float(point[field])
                for point in points
                if field in point
                and start_ratio <= float(point["elapsed_video_time_ratio"]) <= end_ratio
            ]
            if not clip_values:
                adjustment = 0.0
            else:
                z_score = (sum(clip_values) / len(clip_values) - center) / spread
                adjustment = MAXIMUM_SOURCE_RETENTION_ADJUSTMENT * math.tanh(z_score / 2)
            adjusted.append(
                {
                    **candidate,
                    "source_retention_adjustment": adjustment,
                    "personalized_score": float(candidate["personalized_score"]) + adjustment,
                }
            )
        return adjusted, {
            "active": True,
            "metric": field,
            "retention_point_count": len(usable),
            "source_views": exposure,
            "maximum_adjustment": MAXIMUM_SOURCE_RETENTION_ADJUSTMENT,
        }

    def creator_summary(self, creator_id: str) -> dict[str, Any]:
        """Summarize data available for personalization without exposing event records."""
        with self._connect() as connection:
            decision_count = connection.execute(
                """
                SELECT COUNT(DISTINCT clip_id) FROM feedback_events
                WHERE creator_id = ?
                  AND event_type IN (
                      'download_original', 'download_edited', 'custom_created',
                      'review_closed_unselected', 'reject'
                  )
                """,
                (creator_id,),
            ).fetchone()[0]
            performance_count = connection.execute(
                "SELECT COUNT(*) FROM performance_reports WHERE creator_id = ?", (creator_id,)
            ).fetchone()[0]
            analytics_count = connection.execute(
                "SELECT COUNT(*) FROM analytics_imports WHERE creator_id = ?", (creator_id,)
            ).fetchone()[0]
        _, _, performance = self._performance_preference_profile(creator_id)
        return {
            "decision_count": decision_count,
            "editorial_minimum_decision_count": MINIMUM_EDITORIAL_DECISIONS,
            "performance_report_count": performance_count,
            "analytics_import_count": analytics_count,
            "personalization_active": decision_count >= MINIMUM_EDITORIAL_DECISIONS,
            "performance_personalization_active": performance["active"],
            "performance_eligible_clip_count": performance["eligible_clip_count"],
            "performance_minimum_clip_count": performance["minimum_clip_count"],
            "semantic_performance_active": performance["semantic_active"],
            "semantic_performance_example_count": performance["semantic_example_count"],
            "positive_semantic_trends": performance["positive_semantic_trends"],
            "negative_semantic_trends": performance["negative_semantic_trends"],
            "performance_insight_summary": performance["insight_summary"],
        }

    def feedback_training_records(
        self, creator_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Build an auditable offline snapshot from the exact records used while serving."""
        with self._connect() as connection:
            if creator_id is None:
                clips = connection.execute(
                    """
                    SELECT clips.*, videos.creator_id
                    FROM clips JOIN videos ON videos.id = clips.video_id
                    ORDER BY videos.created_at, clips.video_id, clips.rank
                    """
                ).fetchall()
            else:
                clips = connection.execute(
                    """
                    SELECT clips.*, videos.creator_id
                    FROM clips JOIN videos ON videos.id = clips.video_id
                    WHERE videos.creator_id = ?
                    ORDER BY videos.created_at, clips.video_id, clips.rank
                    """,
                    (creator_id,),
                ).fetchall()
            records = []
            for clip_row in clips:
                clip = dict(clip_row)
                events = [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT event_type, start_seconds, end_seconds, payload_json, created_at
                        FROM feedback_events WHERE clip_id = ? ORDER BY id
                        """,
                        (clip["id"],),
                    ).fetchall()
                ]
                for event in events:
                    event["payload"] = json.loads(event.pop("payload_json"))
                labeled = [
                    event
                    for event in events
                    if event["event_type"] in EDITORIAL_EVENT_WEIGHTS
                ]
                latest_label = labeled[-1] if labeled else None
                performance_row = connection.execute(
                    """
                    SELECT * FROM performance_reports
                    WHERE clip_id = ? ORDER BY id DESC LIMIT 1
                    """,
                    (clip["id"],),
                ).fetchone()
                performance = dict(performance_row) if performance_row else None
                if performance:
                    for field in ("id", "creator_id", "clip_id"):
                        performance.pop(field, None)
                repurposing_feedback = [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT repurposing_feedback.platform,
                               repurposing_feedback.action,
                               repurposing_feedback.generated_text,
                               repurposing_feedback.final_text,
                               repurposing_packs.algorithm_version,
                               repurposing_feedback.created_at
                        FROM repurposing_feedback
                        JOIN repurposing_packs
                          ON repurposing_packs.id = repurposing_feedback.pack_id
                        WHERE repurposing_feedback.clip_id = ?
                        ORDER BY repurposing_feedback.id
                        """,
                        (clip["id"],),
                    ).fetchall()
                ]
                final_start = (
                    latest_label["start_seconds"]
                    if latest_label and latest_label["start_seconds"] is not None
                    else clip["start_seconds"]
                )
                final_end = (
                    latest_label["end_seconds"]
                    if latest_label and latest_label["end_seconds"] is not None
                    else clip["end_seconds"]
                )
                records.append(
                    {
                        "creator_id": clip["creator_id"],
                        "video_id": clip["video_id"],
                        "clip_id": clip["id"],
                        "origin": clip["origin"],
                        "display_rank": clip["rank"],
                        "global_rank": clip["global_rank"],
                        "proposed_start_seconds": clip["start_seconds"],
                        "proposed_end_seconds": clip["end_seconds"],
                        "final_start_seconds": final_start,
                        "final_end_seconds": final_end,
                        "start_delta_seconds": final_start - clip["start_seconds"],
                        "end_delta_seconds": final_end - clip["end_seconds"],
                        "transcript_text": clip["transcript_text"],
                        "semantic_embedding": self._semantic_vector(
                            clip["semantic_embedding_json"]
                        ),
                        "ranking_model_version": clip["ranking_model_version"],
                        "predicted_targets": json.loads(clip["predicted_targets_json"]),
                        "global_score": clip["global_score"],
                        "personalized_score": clip["personalized_score"],
                        "adjustments": {
                            "editorial": clip["editorial_adjustment"],
                            "performance": clip["performance_adjustment"],
                            "semantic_performance": clip[
                                "semantic_performance_adjustment"
                            ],
                            "source_retention": clip[
                                "source_retention_adjustment"
                            ],
                            "publishability": clip["publishability_adjustment"],
                        },
                        "publishability": json.loads(clip["publishability_json"]),
                        "editorial_label": latest_label["event_type"]
                        if latest_label
                        else None,
                        "editorial_weight": EDITORIAL_EVENT_WEIGHTS.get(
                            latest_label["event_type"] if latest_label else ""
                        ),
                        "events": events,
                        "latest_performance": performance,
                        "repurposing_feedback": repurposing_feedback,
                    }
                )
        return records

    @staticmethod
    def _public_clip(clip: dict[str, Any]) -> dict[str, Any]:
        clip["predicted_targets"] = json.loads(clip.pop("predicted_targets_json"))
        clip["publishability"] = json.loads(clip.pop("publishability_json", "{}"))
        clip.pop("semantic_embedding_json", None)
        return clip
