"""Durable product records and conservative per-creator preference updates."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EDITORIAL_EVENT_WEIGHTS = {
    "download_original": 1.0,
    "download_edited": 1.0,
    "reject": -1.0,
}
PERSONALIZATION_PRIOR_STRENGTH = 8.0
MAXIMUM_PERSONALIZATION_ADJUSTMENT = 0.35


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
            "CREATE INDEX IF NOT EXISTS idx_videos_creator_id ON videos(creator_id)",
            "CREATE INDEX IF NOT EXISTS idx_clips_video_id ON clips(video_id)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_creator_id ON feedback_events(creator_id)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_clip_id ON feedback_events(clip_id)",
            """
            CREATE INDEX IF NOT EXISTS idx_performance_creator_id
            ON performance_reports(creator_id)
            """,
        )
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            for statement in statements:
                connection.execute(statement)
            connection.execute("PRAGMA optimize")

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
                        predicted_targets_json, explanation, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        json.dumps({"rank": clip["rank"]}),
                        utc_now(),
                    ),
                )

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
        video["clips"] = [self._public_clip(dict(clip)) for clip in clips]
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

    def save_performance_report(self, clip_id: str, value: dict[str, Any]) -> None:
        """Store post-publication outcomes separately from editorial preferences."""
        clip = self.get_clip(clip_id)
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO performance_reports(
                    creator_id, clip_id, platform, views, likes, comments, shares,
                    average_view_percentage, published_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )

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
                      'download_original', 'download_edited', 'reject'
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

        personalized: list[dict[str, Any]] = []
        for candidate in candidates:
            features = preference_features(candidate)
            raw_adjustment = sum(
                weight * feature for weight, feature in zip(weights, features, strict=True)
            ) / len(features)
            adjustment = max(
                -MAXIMUM_PERSONALIZATION_ADJUSTMENT,
                min(MAXIMUM_PERSONALIZATION_ADJUSTMENT, raw_adjustment),
            )
            personalized.append(
                {
                    **candidate,
                    "personalized_score": float(candidate["global_score"]) + adjustment,
                }
            )
        return personalized, {
            "decision_count": count,
            "active": active,
            "shrinkage": shrinkage,
            "maximum_adjustment": MAXIMUM_PERSONALIZATION_ADJUSTMENT,
        }

    def creator_summary(self, creator_id: str) -> dict[str, Any]:
        """Summarize data available for personalization without exposing event records."""
        with self._connect() as connection:
            decision_count = connection.execute(
                """
                SELECT COUNT(DISTINCT clip_id) FROM feedback_events
                WHERE creator_id = ?
                  AND event_type IN ('download_original', 'download_edited', 'reject')
                """,
                (creator_id,),
            ).fetchone()[0]
            performance_count = connection.execute(
                "SELECT COUNT(*) FROM performance_reports WHERE creator_id = ?", (creator_id,)
            ).fetchone()[0]
        return {
            "decision_count": decision_count,
            "performance_report_count": performance_count,
            "personalization_active": decision_count >= 3,
        }

    @staticmethod
    def _public_clip(clip: dict[str, Any]) -> dict[str, Any]:
        clip["predicted_targets"] = json.loads(clip.pop("predicted_targets_json"))
        return clip
