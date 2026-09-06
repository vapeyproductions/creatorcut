"""Dataset validation and transcript-to-label joining."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCORE_FIELDS = ("hook", "completeness", "payoff", "clarity", "presentation")
REQUIRED_ANNOTATION_FIELDS = (
    "annotation_id",
    "video_id",
    "start_seconds",
    "end_seconds",
    *SCORE_FIELDS,
    "technically_exportable",
    "notes",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON objects from a JSON Lines file."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load the source-video manifest."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    return value


def validate_dataset(
    manifest_path: Path,
    annotations_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Validate identifiers, timestamps, score ranges, and local video paths."""
    manifest = load_manifest(manifest_path)
    annotations = load_jsonl(annotations_path)
    errors: list[str] = []
    warnings: list[str] = []

    video_ids = [item.get("video_id") for item in manifest]
    duplicate_video_ids = [key for key, count in Counter(video_ids).items() if count > 1]
    if duplicate_video_ids:
        errors.append(f"Duplicate video IDs: {duplicate_video_ids}")

    known_video_ids = set(video_ids)
    for item in manifest:
        local_filename = item.get("local_filename")
        if not local_filename:
            errors.append(f"{item.get('video_id')}: local_filename is missing")
        elif not (project_root / local_filename).is_file():
            errors.append(f"{item.get('video_id')}: file not found at {local_filename}")

    annotation_ids: list[str] = []
    for row_number, annotation in enumerate(annotations, start=1):
        missing = [field for field in REQUIRED_ANNOTATION_FIELDS if field not in annotation]
        if missing:
            errors.append(f"Annotation row {row_number} is missing: {missing}")
            continue

        annotation_id = str(annotation["annotation_id"])
        annotation_ids.append(annotation_id)
        if annotation["video_id"] not in known_video_ids:
            errors.append(f"{annotation_id}: unknown video_id {annotation['video_id']}")

        start = annotation["start_seconds"]
        end = annotation["end_seconds"]
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            errors.append(f"{annotation_id}: timestamps must be numeric seconds")
        elif start < 0 or end <= start:
            errors.append(f"{annotation_id}: invalid interval [{start}, {end}]")

        for field in SCORE_FIELDS:
            score = annotation[field]
            if not isinstance(score, int) or not 1 <= score <= 5:
                errors.append(f"{annotation_id}: {field} must be an integer from 1 to 5")

    duplicate_annotation_ids = [
        key for key, count in Counter(annotation_ids).items() if count > 1
    ]
    if duplicate_annotation_ids:
        errors.append(f"Duplicate annotation IDs: {duplicate_annotation_ids}")

    for field in SCORE_FIELDS:
        values = {annotation[field] for annotation in annotations if field in annotation}
        if len(values) <= 1:
            warnings.append(f"{field} has no label variance: {sorted(values)}")

    export_values = {
        annotation["technically_exportable"]
        for annotation in annotations
        if "technically_exportable" in annotation
    }
    if len(export_values) <= 1:
        warnings.append(
            "technically_exportable has no label variance and should be treated as a gate, "
            "not a prediction target"
        )

    durations = [row["end_seconds"] - row["start_seconds"] for row in annotations]
    return {
        "videos": len(manifest),
        "annotations": len(annotations),
        "duration_seconds": {
            "minimum": min(durations) if durations else None,
            "maximum": max(durations) if durations else None,
            "mean": sum(durations) / len(durations) if durations else None,
        },
        "warnings": warnings,
        "errors": errors,
    }


def transcript_text_for_interval(
    transcript: dict[str, Any], start_seconds: float, end_seconds: float
) -> tuple[str, int]:
    """Return words that overlap the labeled time interval."""
    words = [
        word
        for segment in transcript["segments"]
        for word in segment.get("words", [])
        if word["end"] > start_seconds and word["start"] < end_seconds
    ]
    text = "".join(word["word"] for word in words).strip()
    return text, len(words)


def build_labeled_dataset(
    annotations_path: Path,
    transcripts_dir: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Attach transcript text to every human-labeled clip."""
    annotations = load_jsonl(annotations_path)
    transcript_cache: dict[str, dict[str, Any]] = {}
    output: list[dict[str, Any]] = []

    for annotation in annotations:
        video_id = annotation["video_id"]
        if video_id not in transcript_cache:
            transcript_path = transcripts_dir / f"{video_id}.json"
            transcript_cache[video_id] = json.loads(transcript_path.read_text(encoding="utf-8"))

        transcript_text, word_count = transcript_text_for_interval(
            transcript_cache[video_id],
            annotation["start_seconds"],
            annotation["end_seconds"],
        )
        output.append(
            {
                **annotation,
                "duration_seconds": annotation["end_seconds"] - annotation["start_seconds"],
                "transcript_text": transcript_text,
                "transcript_word_count": word_count,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in output:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


def main() -> None:
    """Run annotation validation from the command line."""
    parser = argparse.ArgumentParser(description="Validate CreatorCut seed data")
    parser.add_argument("--manifest", type=Path, default=Path("data/videos.json"))
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/annotations/seed_labels.jsonl")
    )
    args = parser.parse_args()

    result = validate_dataset(args.manifest, args.annotations, Path.cwd())
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


def build_dataset_main() -> None:
    """Join transcripts to labels from the command line."""
    parser = argparse.ArgumentParser(description="Attach transcript text to clip labels")
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/annotations/seed_labels.jsonl")
    )
    parser.add_argument(
        "--transcripts-dir", type=Path, default=Path("data/processed/transcripts")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/labeled_clips.jsonl")
    )
    args = parser.parse_args()

    records = build_labeled_dataset(args.annotations, args.transcripts_dir, args.output)
    empty = [record["annotation_id"] for record in records if not record["transcript_text"]]
    print(f"Wrote {len(records)} labeled clips to {args.output}")
    if empty:
        print(f"Warning: {len(empty)} clips have no transcript text: {empty}")
