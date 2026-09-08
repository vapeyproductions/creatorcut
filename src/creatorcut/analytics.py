"""Parse YouTube, Instagram, and TikTok analytics into one stable schema."""

from __future__ import annotations

import csv
import io
import math
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

SUPPORTED_ANALYTICS_PLATFORMS = {"youtube", "instagram", "tiktok"}

MAXIMUM_ANALYTICS_BYTES = 20 * 1024 * 1024
MAXIMUM_UNCOMPRESSED_ANALYTICS_BYTES = 50 * 1024 * 1024
MAXIMUM_CSV_FILES = 30
MAXIMUM_REPORT_ROWS = 20_000


def normalize_header(value: str) -> str:
    """Normalize YouTube UI and API column names without depending on locale punctuation."""
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = value.casefold().replace("%", " percentage ")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


FIELD_ALIASES = {
    "date": {"date", "day"},
    "content": {
        "content",
        "video",
        "video title",
        "post",
        "post caption",
        "caption",
        "description",
        "reel",
    },
    "video_id": {"video id", "post id", "media id", "reel id", "item id"},
    "channel_id": {"channel id"},
    "creator_content_type": {"creator content type", "content type"},
    "subscribed_status": {"subscribed status"},
    "country_code": {"country code", "country"},
    "traffic_source_type": {"traffic source type"},
    "traffic_source_detail": {"traffic source detail"},
    "device_type": {"device type"},
    "operating_system": {"operating system"},
    "playback_location_type": {"playback location type"},
    "age_group": {"age group"},
    "gender": {"gender"},
    "sharing_service": {"sharing service"},
    "subtitle_language": {"subtitle language"},
    "views": {
        "views",
        "video views",
        "post views",
        "reel views",
        "reels views",
        "plays",
        "video play count",
        "initial plays",
        "initial views",
    },
    "engaged_views": {"engaged views"},
    "watch_time_hours": {"watch time hours"},
    "watch_time_minutes": {"watch time minutes", "estimated minutes watched"},
    "watch_time_seconds": {
        "watch time",
        "watch time seconds",
        "total watch time",
        "total play time",
        "total watch duration",
        "video view total time",
        "ig reels video view total time",
    },
    "average_view_duration_seconds": {
        "average view duration",
        "average view duration seconds",
        "average watch time",
        "average watch time seconds",
        "avg watch time",
        "ig reels avg watch time",
    },
    "average_view_percentage": {
        "average percentage viewed percentage",
        "average percentage viewed",
        "average view percentage",
        "average view duration percentage",
    },
    "likes": {"likes"},
    "dislikes": {"dislikes"},
    "comments": {"comments"},
    "shares": {"shares"},
    "saves": {"saves", "saved"},
    "reach": {"reach", "accounts reached", "unique viewers", "reached audience"},
    "follows": {"follows", "new follows", "new followers", "followers gained"},
    "profile_visits": {"profile visits", "profile views"},
    "replays": {
        "replays",
        "replay count",
        "clips replays count",
        "aggregated all plays count",
    },
    "total_interactions": {"total interactions", "interactions", "engagement"},
    "completion_rate_percentage": {
        "watched full video percentage",
        "watched full video",
        "full video watched percentage",
        "completion rate percentage",
        "completion rate",
        "completed video percentage",
    },
    "subscribers": {"subscribers"},
    "subscribers_gained": {"subscribers gained"},
    "subscribers_lost": {"subscribers lost"},
    "shown_in_feed": {"shown in feed", "shorts shown in feed"},
    "chose_to_view_percentage": {
        "how many chose to view percentage",
        "chose to view percentage",
        "stayed to watch percentage",
        "stayed to watch",
    },
    "thumbnail_impressions": {
        "thumbnail impressions",
        "impressions",
        "video thumbnail impressions",
    },
    "thumbnail_ctr": {
        "thumbnail click through rate percentage",
        "impressions click through rate percentage",
        "click through rate percentage",
        "video thumbnail impressions ctr",
    },
    "elapsed_video_time_ratio": {
        "elapsed video time ratio",
        "elapsed video time percentage",
    },
    "audience_watch_ratio": {
        "audience watch ratio",
        "audience watch ratio percentage",
    },
    "relative_retention_performance": {
        "relative retention performance",
        "relative retention performance percentage",
    },
}

ALIAS_TO_FIELD = {
    alias: field for field, aliases in FIELD_ALIASES.items() for alias in aliases
}

INTEGER_FIELDS = {
    "views",
    "engaged_views",
    "likes",
    "dislikes",
    "comments",
    "shares",
    "subscribers",
    "subscribers_gained",
    "subscribers_lost",
    "shown_in_feed",
    "thumbnail_impressions",
    "saves",
    "reach",
    "follows",
    "profile_visits",
    "replays",
    "total_interactions",
}

DIMENSION_FIELDS = {
    "date",
    "content",
    "video_id",
    "channel_id",
    "creator_content_type",
    "subscribed_status",
    "country_code",
    "traffic_source_type",
    "traffic_source_detail",
    "device_type",
    "operating_system",
    "playback_location_type",
    "age_group",
    "gender",
    "sharing_service",
    "subtitle_language",
}

PERCENTAGE_FIELDS = {
    "average_view_percentage",
    "chose_to_view_percentage",
    "thumbnail_ctr",
    "completion_rate_percentage",
}


def _parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {"—", "-", "n/a", "N/A"}:
        return None
    text = text.removesuffix("%").strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _parse_duration_seconds(value: Any) -> float | None:
    number = _parse_number(value)
    text = str(value).strip() if value is not None else ""
    unit_parts = re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", text.casefold())
    if unit_parts:
        multiplier = {"h": 3600.0, "m": 60.0, "s": 1.0}
        return sum(float(amount) * multiplier[unit] for amount, unit in unit_parts)
    if ":" not in text:
        return number
    try:
        parts = [float(part) for part in text.split(":")]
    except ValueError:
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def _canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for header, value in row.items():
        normalized_header = normalize_header(header or "")
        field = ALIAS_TO_FIELD.get(normalized_header)
        if field is None:
            continue
        if field in DIMENSION_FIELDS:
            cleaned = str(value or "").strip()
            canonical[field] = cleaned or None
        elif field in {"average_view_duration_seconds", "watch_time_seconds"}:
            canonical[field] = _parse_duration_seconds(value)
        else:
            number = _parse_number(value)
            if (
                number is not None
                and field
                in {
                    "elapsed_video_time_ratio",
                    "audience_watch_ratio",
                    "relative_retention_performance",
                }
                and "percentage" in normalized_header
            ):
                number /= 100.0
            if field in INTEGER_FIELDS and number is not None:
                canonical[field] = int(round(number))
            else:
                canonical[field] = number
    return {key: value for key, value in canonical.items() if value is not None}


def _decode_csv(value: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("The analytics CSV text encoding is not supported")


def _csv_files(filename: str, payload: bytes) -> list[tuple[str, bytes]]:
    suffix = Path(filename).suffix.casefold()
    if suffix in {".csv", ".tsv"}:
        return [(Path(filename).name, payload)]
    if suffix == ".xlsx":
        return _xlsx_csv_files(filename, payload)
    if suffix != ".zip":
        raise ValueError("Upload a .csv, .tsv, .xlsx, or supported analytics .zip export")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir() and Path(member.filename).suffix.casefold() == ".csv"
            ]
            if not members:
                raise ValueError("The ZIP archive does not contain a CSV report")
            if len(members) > MAXIMUM_CSV_FILES:
                raise ValueError("The analytics archive contains too many CSV files")
            if sum(member.file_size for member in members) > MAXIMUM_UNCOMPRESSED_ANALYTICS_BYTES:
                raise ValueError("The analytics archive expands beyond 50 MB")
            return [(Path(member.filename).name, archive.read(member)) for member in members]
    except zipfile.BadZipFile as error:
        raise ValueError("The analytics ZIP archive is invalid") from error


def _column_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    result = 0
    for character in letters.upper():
        result = result * 26 + ord(character) - ord("A") + 1
    return max(0, result - 1)


def _xlsx_csv_files(filename: str, payload: bytes) -> list[tuple[str, bytes]]:
    """Read ordinary Google Sheets/Excel exports without introducing a workbook runtime."""
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    relationship_namespace = {
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships"
    }
    office_relationship = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            shared: list[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
                shared = [
                    "".join(node.text or "" for node in item.findall(".//main:t", namespace))
                    for item in root.findall("main:si", namespace)
                ]
            workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
            relationships = ElementTree.fromstring(
                archive.read("xl/_rels/workbook.xml.rels")
            )
            targets = {
                item.attrib["Id"]: item.attrib["Target"]
                for item in relationships.findall("rel:Relationship", relationship_namespace)
            }
            sheets = workbook.findall("main:sheets/main:sheet", namespace)
            if len(sheets) > MAXIMUM_CSV_FILES:
                raise ValueError("The analytics workbook contains too many sheets")
            converted: list[tuple[str, bytes]] = []
            for sheet in sheets:
                target = targets.get(sheet.attrib.get(office_relationship, ""))
                if not target:
                    continue
                worksheet_path = target.lstrip("/")
                if not worksheet_path.startswith("xl/"):
                    worksheet_path = f"xl/{worksheet_path}"
                root = ElementTree.fromstring(archive.read(worksheet_path))
                rows: list[list[str]] = []
                for row in root.findall(".//main:sheetData/main:row", namespace):
                    values: dict[int, str] = {}
                    for cell in row.findall("main:c", namespace):
                        index = _column_index(cell.attrib.get("r", "A1"))
                        cell_type = cell.attrib.get("t")
                        if cell_type == "inlineStr":
                            value = "".join(
                                node.text or ""
                                for node in cell.findall(".//main:t", namespace)
                            )
                        else:
                            node = cell.find("main:v", namespace)
                            value = node.text if node is not None and node.text is not None else ""
                            if cell_type == "s" and value:
                                value = shared[int(value)]
                        values[index] = value
                    if values:
                        rows.append([values.get(index, "") for index in range(max(values) + 1)])
                if not rows:
                    continue
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerows(rows)
                converted.append(
                    (
                        f"{Path(filename).stem}-{sheet.attrib.get('name', 'Sheet')}.csv",
                        output.getvalue().encode(),
                    )
                )
            if not converted:
                raise ValueError("The analytics workbook does not contain readable rows")
            return converted
    except (KeyError, IndexError, ElementTree.ParseError, zipfile.BadZipFile) as error:
        raise ValueError("The analytics workbook is invalid") from error


def _retention_ratio(value: float) -> float:
    """Accept API ratios (0–1) and Studio percentages (0–100)."""
    return value / 100.0 if value > 1.0 else value


def _report_type(fields: set[str]) -> str:
    if "elapsed_video_time_ratio" in fields:
        return "audience_retention"
    if "traffic_source_type" in fields:
        return "traffic_sources"
    if "device_type" in fields or "operating_system" in fields:
        return "device_and_os"
    if "age_group" in fields or "gender" in fields:
        return "demographics"
    if "thumbnail_impressions" in fields:
        return "reach"
    if "date" in fields:
        return "daily_performance"
    return "summary"


def parse_analytics_export(
    platform: str, filename: str, payload: bytes
) -> dict[str, Any]:
    """Parse one supported creator-platform export without a fixed column order."""
    if platform not in SUPPORTED_ANALYTICS_PLATFORMS:
        raise ValueError("Choose YouTube, Instagram Reels, or TikTok analytics")
    if not payload:
        raise ValueError("The analytics export is empty")
    if len(payload) > MAXIMUM_ANALYTICS_BYTES:
        raise ValueError("Analytics exports must be 20 MB or smaller")

    totals: dict[str, float | int] = {}
    daily_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, float]] = []
    report_rows: list[dict[str, Any]] = []
    report_types: set[str] = set()
    recognized_rows = 0
    parsed_files: list[str] = []

    for csv_name, csv_bytes in _csv_files(filename, payload):
        text = _decode_csv(csv_bytes)
        delimiter = "\t" if Path(csv_name).suffix.casefold() == ".tsv" else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        if not reader.fieldnames:
            continue
        parsed_files.append(csv_name)
        canonical_fields = {
            ALIAS_TO_FIELD[normalized]
            for header in reader.fieldnames
            if (normalized := normalize_header(header or "")) in ALIAS_TO_FIELD
        }
        report_types.add(_report_type(canonical_fields))
        for row_index, row in enumerate(reader, start=1):
            if row_index > MAXIMUM_REPORT_ROWS:
                raise ValueError("An analytics CSV contains more than 20,000 rows")
            canonical = _canonical_row(row)
            if not canonical:
                continue
            recognized_rows += 1
            report_rows.append(canonical)
            content = str(canonical.get("content", "")).casefold()
            numeric = {
                key: value
                for key, value in canonical.items()
                if key not in DIMENSION_FIELDS | {"elapsed_video_time_ratio"}
            }
            if content == "total" or (
                len(reader.fieldnames) > 1 and numeric and len(text.splitlines()) <= 3
            ):
                totals.update(numeric)
            if "date" in canonical:
                daily_rows.append(canonical)
            if "elapsed_video_time_ratio" in canonical:
                point: dict[str, float] = {
                    "elapsed_video_time_ratio": _retention_ratio(
                        float(canonical["elapsed_video_time_ratio"])
                    )
                }
                for field in ("audience_watch_ratio", "relative_retention_performance"):
                    if field in canonical:
                        point[field] = float(canonical[field])
                if len(point) > 1 and 0 <= point["elapsed_video_time_ratio"] <= 1:
                    retention_rows.append(point)

    if recognized_rows == 0:
        label = {
            "youtube": "YouTube",
            "instagram": "Instagram",
            "tiktok": "TikTok",
        }[platform]
        raise ValueError(f"No supported {label} analytics columns were found")

    simple_daily_rows = [
        row
        for row in daily_rows
        if not (set(row) & (DIMENSION_FIELDS - {"date", "content"}))
    ]
    if not totals and simple_daily_rows:
        for field in INTEGER_FIELDS:
            values = [row[field] for row in simple_daily_rows if field in row]
            if values:
                totals[field] = int(sum(values))
        for field in ("watch_time_hours", "watch_time_minutes"):
            values = [row[field] for row in simple_daily_rows if field in row]
            if values:
                totals[field] = float(sum(values))
        weighted_metrics = {
            "average_view_percentage": "engaged_views",
            "average_view_duration_seconds": "engaged_views",
            "chose_to_view_percentage": "shown_in_feed",
            "thumbnail_ctr": "thumbnail_impressions",
        }
        for field, preferred_weight in weighted_metrics.items():
            pairs = []
            for row in simple_daily_rows:
                if field not in row:
                    continue
                weight = row.get(preferred_weight, row.get("views", 0))
                if weight:
                    pairs.append((float(row[field]), float(weight)))
            if pairs:
                totals[field] = sum(value * weight for value, weight in pairs) / sum(
                    weight for _, weight in pairs
                )

    retention_rows.sort(key=lambda row: row["elapsed_video_time_ratio"])
    warnings: list[str] = []
    exposure = totals.get("engaged_views", totals.get("views"))
    if exposure is not None and exposure < 50:
        warnings.append("Too little exposure to influence ranking yet")
    if not retention_rows:
        warnings.append("No timestamped audience-retention curve was included")

    return {
        "schema_version": 2,
        "platform": platform,
        "source": f"{platform}_creator_analytics_export",
        "source_files": parsed_files,
        "totals": totals,
        "daily_rows": daily_rows,
        "retention_rows": retention_rows,
        "report_rows": report_rows,
        "report_types": sorted(report_types),
        "recognized_row_count": recognized_rows,
        "warnings": warnings,
    }


def parse_youtube_analytics_export(filename: str, payload: bytes) -> dict[str, Any]:
    """Retain the original YouTube-specific API for older callers and snapshots."""
    return parse_analytics_export("youtube", filename, payload)


def performance_values(report: dict[str, Any]) -> dict[str, Any]:
    """Map parsed totals into the performance-report columns used by the product store."""
    totals = report.get("totals", {})
    fields = (
        "views",
        "engaged_views",
        "watch_time_hours",
        "watch_time_seconds",
        "average_view_duration_seconds",
        "average_view_percentage",
        "likes",
        "comments",
        "shares",
        "saves",
        "reach",
        "follows",
        "profile_visits",
        "replays",
        "total_interactions",
        "completion_rate_percentage",
        "subscribers_gained",
        "subscribers_lost",
        "shown_in_feed",
        "chose_to_view_percentage",
        "thumbnail_impressions",
        "thumbnail_ctr",
    )
    values = {field: totals.get(field) for field in fields}
    values["subscribers_net"] = totals.get("subscribers")
    if values["watch_time_hours"] is None and totals.get("watch_time_minutes") is not None:
        values["watch_time_hours"] = float(totals["watch_time_minutes"]) / 60.0
    if values["watch_time_hours"] is None and totals.get("watch_time_seconds") is not None:
        values["watch_time_hours"] = float(totals["watch_time_seconds"]) / 3600.0
    return values
