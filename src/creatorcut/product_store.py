"""Durable product records and conservative per-creator preference updates."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from creatorcut.analytics import performance_values

EDITORIAL_EVENT_WEIGHTS = {
    "download_original": 1.0,
    "download_edited": 1.0,
    "reject": -1.0,
}
PERSONALIZATION_PRIOR_STRENGTH = 8.0
MAXIMUM_PERSONALIZATION_ADJUSTMENT = 0.35
PERFORMANCE_PRIOR_STRENGTH = 12.0
MAXIMUM_PERFORMANCE_ADJUSTMENT = 0.25
MINIMUM_PERFORMANCE_EXAMPLES = 5
MINIMUM_PERFORMANCE_VIEWS = 50
MAXIMUM_SOURCE_RETENTION_ADJUSTMENT = 0.20
MINIMUM_SOURCE_RETENTION_POINTS = 10
MINIMUM_SOURCE_RETENTION_VIEWS = 100

PREFERENCE_FEATURE_NAMES = ("hook", "completeness", "payoff", "clarity", "duration")


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
        )
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            for statement in statements:
                connection.execute(statement)
            self._add_missing_columns(connection)
            connection.execute("PRAGMA optimize")

    @staticmethod
    def _add_missing_columns(connection: sqlite3.Connection) -> None:
        """Apply additive migrations to databases created by earlier local builds."""
        migrations = {
            "clips": {
                "global_rank": "INTEGER",
                "ranking_model_version": "TEXT",
                "editorial_adjustment": "REAL NOT NULL DEFAULT 0",
                "performance_adjustment": "REAL NOT NULL DEFAULT 0",
                "source_retention_adjustment": "REAL NOT NULL DEFAULT 0",
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
                        predicted_targets_json, explanation, global_rank,
                        ranking_model_version,
                        editorial_adjustment, performance_adjustment,
                        source_retention_adjustment, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        clip.get("editorial_adjustment", 0.0),
                        clip.get("performance_adjustment", 0.0),
                        clip.get("source_retention_adjustment", 0.0),
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

    def _performance_preference_profile(
        self, creator_id: str
    ) -> tuple[list[float], dict[str, Any]]:
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
                       clips.predicted_targets_json
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
        return weights, {
            "active": active,
            "eligible_clip_count": len(eligible),
            "minimum_clip_count": MINIMUM_PERFORMANCE_EXAMPLES,
            "minimum_views": MINIMUM_PERFORMANCE_VIEWS,
            "shrinkage": shrinkage,
            "feature_weights": dict(zip(PREFERENCE_FEATURE_NAMES, weights, strict=True)),
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

        performance_weights, performance_metadata = self._performance_preference_profile(
            creator_id
        )
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
                -MAXIMUM_PERFORMANCE_ADJUSTMENT,
                min(MAXIMUM_PERFORMANCE_ADJUSTMENT, raw_performance_adjustment),
            )
            personalized.append(
                {
                    **candidate,
                    "editorial_adjustment": editorial_adjustment,
                    "performance_adjustment": performance_adjustment,
                    "source_retention_adjustment": 0.0,
                    "personalized_score": (
                        float(candidate["global_score"])
                        + editorial_adjustment
                        + performance_adjustment
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
                  AND event_type IN ('download_original', 'download_edited', 'reject')
                """,
                (creator_id,),
            ).fetchone()[0]
            performance_count = connection.execute(
                "SELECT COUNT(*) FROM performance_reports WHERE creator_id = ?", (creator_id,)
            ).fetchone()[0]
            analytics_count = connection.execute(
                "SELECT COUNT(*) FROM analytics_imports WHERE creator_id = ?", (creator_id,)
            ).fetchone()[0]
        _, performance = self._performance_preference_profile(creator_id)
        return {
            "decision_count": decision_count,
            "performance_report_count": performance_count,
            "analytics_import_count": analytics_count,
            "personalization_active": decision_count >= 3,
            "performance_personalization_active": performance["active"],
            "performance_eligible_clip_count": performance["eligible_clip_count"],
            "performance_minimum_clip_count": performance["minimum_clip_count"],
        }

    @staticmethod
    def _public_clip(clip: dict[str, Any]) -> dict[str, Any]:
        clip["predicted_targets"] = json.loads(clip.pop("predicted_targets_json"))
        return clip
