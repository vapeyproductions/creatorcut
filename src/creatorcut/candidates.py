"""Generate sentence-aligned clip candidates and evaluate interval recall."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from creatorcut.dataset import load_jsonl

CORE_SCORE_FIELDS = ("hook", "completeness", "payoff", "clarity")
SENTENCE_END = re.compile(r"[.!?][\"'”’)]*$")


@dataclass(frozen=True)
class SentenceUnit:
    """A transcript unit with clean temporal and textual boundaries."""

    start: float
    end: float
    text: str
    word_count: int


def flatten_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten timestamped words from all ASR segments."""
    return [word for segment in transcript["segments"] for word in segment.get("words", [])]


def words_to_sentence_units(
    words: list[dict[str, Any]],
    pause_split_seconds: float = 1.0,
    maximum_unit_seconds: float = 15.0,
) -> list[SentenceUnit]:
    """Create units at punctuation, long pauses, or a safety duration limit."""
    if not words:
        return []

    units: list[SentenceUnit] = []
    current: list[dict[str, Any]] = []

    for index, word in enumerate(words):
        current.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        current_duration = current[-1]["end"] - current[0]["start"]
        punctuation_boundary = bool(SENTENCE_END.search(word["word"].strip()))
        pause_boundary = bool(
            next_word
            and next_word["start"] - word["end"] >= pause_split_seconds
            and current_duration >= 2.0
        )
        forced_boundary = current_duration >= maximum_unit_seconds
        final_word = next_word is None

        if punctuation_boundary or pause_boundary or forced_boundary or final_word:
            units.append(
                SentenceUnit(
                    start=current[0]["start"],
                    end=current[-1]["end"],
                    text="".join(item["word"] for item in current).strip(),
                    word_count=len(current),
                )
            )
            current = []

    return units


def build_candidate(
    video_id: str,
    units: list[SentenceUnit],
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    """Build one stable candidate record from consecutive sentence units."""
    selected = units[start_index : end_index + 1]
    start = selected[0].start
    end = selected[-1].end
    return {
        "video_id": video_id,
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": end - start,
        "sentence_count": len(selected),
        "word_count": sum(unit.word_count for unit in selected),
        "text": " ".join(unit.text for unit in selected),
        "start_unit_index": start_index,
        "end_unit_index": end_index,
    }


def generate_candidates(
    video_id: str,
    units: list[SentenceUnit],
    minimum_duration: float = 20.0,
    maximum_duration: float = 60.0,
    target_durations: Iterable[float] = (20.0, 30.0, 45.0, 60.0),
) -> list[dict[str, Any]]:
    """Generate a compact set of sentence-aligned durations from every start unit."""
    targets = tuple(target_durations)
    if minimum_duration <= 0 or maximum_duration <= minimum_duration:
        raise ValueError("Expected 0 < minimum_duration < maximum_duration")
    if not targets:
        raise ValueError("At least one target duration is required")

    candidates: list[dict[str, Any]] = []
    seen_boundaries: set[tuple[float, float]] = set()

    for start_index in range(len(units)):
        valid_end_indices: list[int] = []
        for end_index in range(start_index, len(units)):
            duration = units[end_index].end - units[start_index].start
            if duration < minimum_duration:
                continue
            if duration > maximum_duration:
                break
            valid_end_indices.append(end_index)

        for target in targets:
            if not valid_end_indices:
                continue
            end_index = min(
                valid_end_indices,
                key=lambda value: abs((units[value].end - units[start_index].start) - target),
            )
            boundary_key = (units[start_index].start, units[end_index].end)
            if boundary_key in seen_boundaries:
                continue
            seen_boundaries.add(boundary_key)
            candidates.append(build_candidate(video_id, units, start_index, end_index))

    candidates.sort(key=lambda item: (item["start_seconds"], item["end_seconds"]))
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_id"] = f"{video_id}_candidate_{index:04d}"
    return candidates


def interval_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Calculate intersection over union for two temporal intervals."""
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0 else 0.0


def core_relevance(annotation: dict[str, Any]) -> float:
    """Calculate the provisional mean of the four content-quality labels."""
    return sum(annotation[field] for field in CORE_SCORE_FIELDS) / len(CORE_SCORE_FIELDS)


def evaluate_candidate_recall(
    annotations: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    strong_threshold: float = 4.0,
) -> dict[str, Any]:
    """Measure whether generated intervals overlap human-selected clips."""
    candidates_by_video: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        candidates_by_video.setdefault(candidate["video_id"], []).append(candidate)

    matches: list[dict[str, Any]] = []
    for annotation in annotations:
        video_candidates = candidates_by_video.get(annotation["video_id"], [])
        scored = [
            (
                interval_iou(
                    (annotation["start_seconds"], annotation["end_seconds"]),
                    (candidate["start_seconds"], candidate["end_seconds"]),
                ),
                candidate,
            )
            for candidate in video_candidates
        ]
        best_iou, best_candidate = max(scored, key=lambda item: item[0]) if scored else (0.0, None)
        relevance = core_relevance(annotation)
        matches.append(
            {
                "annotation_id": annotation["annotation_id"],
                "video_id": annotation["video_id"],
                "human_start_seconds": annotation["start_seconds"],
                "human_end_seconds": annotation["end_seconds"],
                "core_relevance": relevance,
                "strong": relevance >= strong_threshold,
                "best_iou": best_iou,
                "best_candidate_id": best_candidate["candidate_id"] if best_candidate else None,
                "candidate_start_seconds": (
                    best_candidate["start_seconds"] if best_candidate else None
                ),
                "candidate_end_seconds": best_candidate["end_seconds"] if best_candidate else None,
            }
        )

    strong_matches = [match for match in matches if match["strong"]]

    def recall(rows: list[dict[str, Any]], threshold: float) -> float | None:
        if not rows:
            return None
        return sum(row["best_iou"] >= threshold for row in rows) / len(rows)

    return {
        "candidate_count": len(candidates),
        "annotation_count": len(matches),
        "strong_threshold": strong_threshold,
        "strong_annotation_count": len(strong_matches),
        "recall": {
            "all_at_iou_0.50": recall(matches, 0.50),
            "all_at_iou_0.70": recall(matches, 0.70),
            "strong_at_iou_0.50": recall(strong_matches, 0.50),
            "strong_at_iou_0.70": recall(strong_matches, 0.70),
        },
        "matches": matches,
    }


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_main() -> None:
    """Generate candidates for every transcript from the command line."""
    parser = argparse.ArgumentParser(description="Generate sentence-aligned clip candidates")
    parser.add_argument(
        "--transcripts-dir", type=Path, default=Path("data/processed/transcripts")
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/candidates.jsonl"))
    parser.add_argument("--minimum-duration", type=float, default=20.0)
    parser.add_argument("--maximum-duration", type=float, default=60.0)
    parser.add_argument(
        "--target-duration", type=float, action="append", dest="target_durations"
    )
    args = parser.parse_args()

    all_candidates: list[dict[str, Any]] = []
    for transcript_path in sorted(args.transcripts_dir.glob("*.json")):
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        words = flatten_words(transcript)
        units = words_to_sentence_units(words)
        candidates = generate_candidates(
            transcript["video_id"],
            units,
            minimum_duration=args.minimum_duration,
            maximum_duration=args.maximum_duration,
            target_durations=args.target_durations or (20.0, 30.0, 45.0, 60.0),
        )
        print(
            f"{transcript['video_id']}: {len(words)} words -> "
            f"{len(units)} sentence units -> {len(candidates)} candidates"
        )
        all_candidates.extend(candidates)

    write_jsonl(all_candidates, args.output)
    print(f"Wrote {len(all_candidates)} candidates to {args.output}")


def evaluate_main() -> None:
    """Evaluate candidate interval recall from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate candidate-generation recall")
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/annotations/seed_labels.jsonl")
    )
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/candidates.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/candidate_evaluation.json")
    )
    parser.add_argument("--strong-threshold", type=float, default=4.0)
    args = parser.parse_args()

    evaluation = evaluate_candidate_recall(
        load_jsonl(args.annotations),
        load_jsonl(args.candidates),
        strong_threshold=args.strong_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    summary = {key: value for key, value in evaluation.items() if key != "matches"}
    print(json.dumps(summary, indent=2))
    print(f"Wrote detailed matches to {args.output}")
