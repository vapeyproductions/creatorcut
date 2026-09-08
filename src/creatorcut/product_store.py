"""Durable product records and conservative per-creator preference updates."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

from creatorcut.analytics import performance_values

EDITORIAL_EVENT_WEIGHTS = {
    "download_original": 1.0,
    "download_edited": 1.0,
    "custom_created": 1.0,
    "review_closed_unselected": -0.35,
    "reject": -1.0,
}
PERSONALIZATION_PRIOR_STRENGTH = 8.0
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
            "CREATE INDEX IF NOT EXISTS idx_videos_creator_id ON videos(creator_id)",
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
        active = count >= 3
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
            "performance_report_count": performance_count,
            "analytics_import_count": analytics_count,
            "personalization_active": decision_count >= 3,
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
                    }
                )
        return records

    @staticmethod
    def _public_clip(clip: dict[str, Any]) -> dict[str, Any]:
        clip["predicted_targets"] = json.loads(clip.pop("predicted_targets_json"))
        clip["publishability"] = json.loads(clip.pop("publishability_json", "{}"))
        clip.pop("semantic_embedding_json", None)
        return clip
