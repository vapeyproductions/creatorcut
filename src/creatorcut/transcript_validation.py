"""Validate timestamped ASR outputs before candidate generation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"[a-z]+")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate_transcript(transcript: dict[str, Any]) -> dict[str, Any]:
    """Check one transcript's identity, timing, confidence, and content."""
    video_id = transcript.get("video_id")
    duration = transcript.get("media_duration_seconds")
    segments = transcript.get("segments")
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(video_id, str) or not video_id:
        errors.append("video_id must be a non-empty string")
    if not _is_number(duration) or duration <= 0:
        errors.append("media_duration_seconds must be positive")
        duration = None
    if not isinstance(segments, list):
        errors.append("segments must be a list")
        segments = []

    word_count = 0
    probabilities: list[float] = []
    repetitive_segment_count = 0
    previous_segment_start = -1.0
    previous_word_start = -1.0
    for segment_index, segment in enumerate(segments):
        start = segment.get("start")
        end = segment.get("end")
        label = f"segment {segment_index}"
        if not _is_number(start) or not _is_number(end):
            errors.append(f"{label}: start and end must be numeric")
        else:
            if start < 0 or end < start:
                errors.append(f"{label}: invalid interval [{start}, {end}]")
            if start < previous_segment_start:
                errors.append(f"{label}: segments are not ordered by start time")
            if duration is not None and end > duration + 1.0:
                errors.append(f"{label}: end exceeds media duration")
            previous_segment_start = start

        words = segment.get("words", [])
        if not isinstance(words, list):
            errors.append(f"{label}: words must be a list")
            continue
        for word_index, word in enumerate(words):
            word_count += 1
            word_start = word.get("start")
            word_end = word.get("end")
            probability = word.get("probability")
            word_label = f"{label}, word {word_index}"
            if not _is_number(word_start) or not _is_number(word_end):
                errors.append(f"{word_label}: start and end must be numeric")
            else:
                if word_start < 0 or word_end < word_start:
                    errors.append(f"{word_label}: invalid interval [{word_start}, {word_end}]")
                if word_start < previous_word_start:
                    errors.append(f"{word_label}: words are not ordered by start time")
                if duration is not None and word_end > duration + 1.0:
                    errors.append(f"{word_label}: end exceeds media duration")
                previous_word_start = word_start
            if probability is not None and (
                not _is_number(probability) or not 0.0 <= probability <= 1.0
            ):
                errors.append(f"{word_label}: probability must be between 0 and 1")
            elif probability is not None:
                probabilities.append(float(probability))

        tokens = TOKEN_PATTERN.findall(str(segment.get("text", "")).lower())
        four_grams = [tuple(tokens[index : index + 4]) for index in range(len(tokens) - 3)]
        if max(Counter(four_grams).values(), default=0) >= 8:
            repetitive_segment_count += 1

    if not segments:
        warnings.append("transcript has no segments")
    if word_count == 0:
        warnings.append("transcript has no timestamped words")
    mean_probability = sum(probabilities) / len(probabilities) if probabilities else None
    low_confidence_fraction = (
        sum(probability < 0.5 for probability in probabilities) / len(probabilities)
        if probabilities
        else None
    )
    if mean_probability is not None and mean_probability < 0.8:
        warnings.append(f"mean word confidence {mean_probability:.3f} is below 0.800")
    if repetitive_segment_count:
        warnings.append(
            f"{repetitive_segment_count} segment(s) contain a four-word phrase repeated 8+ times"
        )

    return {
        "video_id": video_id,
        "duration_seconds": duration,
        "segment_count": len(segments),
        "word_count": word_count,
        "mean_word_probability": mean_probability,
        "low_confidence_word_fraction": low_confidence_fraction,
        "repetitive_segment_count": repetitive_segment_count,
        "warnings": warnings,
        "errors": errors,
    }


def validate_transcript_directory(transcripts_dir: Path) -> dict[str, Any]:
    """Validate every JSON transcript in a directory and summarize the corpus."""
    results: list[dict[str, Any]] = []
    for path in sorted(transcripts_dir.glob("*.json")):
        transcript = json.loads(path.read_text(encoding="utf-8"))
        result = validate_transcript(transcript)
        result["path"] = path.as_posix()
        results.append(result)

    errors = [f"{result['path']}: {message}" for result in results for message in result["errors"]]
    warnings = [
        f"{result['path']}: {message}" for result in results for message in result["warnings"]
    ]
    video_ids = [result["video_id"] for result in results]
    duplicates = [video_id for video_id, count in Counter(video_ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate video IDs: {duplicates}")

    return {
        "transcript_count": len(results),
        "total_duration_seconds": sum(result["duration_seconds"] or 0 for result in results),
        "total_segments": sum(result["segment_count"] for result in results),
        "total_words": sum(result["word_count"] for result in results),
        "warnings": warnings,
        "errors": errors,
        "transcripts": results,
    }


def main() -> None:
    """Validate all timestamped transcripts from the command line."""
    parser = argparse.ArgumentParser(description="Validate CreatorCut ASR transcripts")
    parser.add_argument("--transcripts-dir", type=Path, default=Path("data/processed/transcripts"))
    args = parser.parse_args()

    result = validate_transcript_directory(args.transcripts_dir)
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
