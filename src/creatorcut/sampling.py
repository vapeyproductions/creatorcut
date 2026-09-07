"""Build a diverse, deterministic queue for human clip annotation."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from creatorcut.baseline import score_candidates
from creatorcut.candidates import interval_iou, write_jsonl
from creatorcut.dataset import load_jsonl
from creatorcut.transcript_validation import validate_transcript_directory

SCORE_BANDS = ("low", "medium", "high")
TARGET_DURATIONS = (25.0, 40.0, 55.0)


def partition_score_bands(
    candidates: list[dict[str, Any]],
    score_field: str = "transcript_heuristic_score",
) -> dict[str, list[dict[str, Any]]]:
    """Split ranked candidates into approximately equal low, medium, and high bands."""
    ordered = sorted(candidates, key=lambda item: (item[score_field], item["candidate_id"]))
    bands: dict[str, list[dict[str, Any]]] = {band: [] for band in SCORE_BANDS}
    for index, candidate in enumerate(ordered):
        band_index = min(len(SCORE_BANDS) - 1, index * len(SCORE_BANDS) // len(ordered))
        bands[SCORE_BANDS[band_index]].append(candidate)
    return bands


def _overlaps_selected(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
    maximum_overlap_iou: float,
) -> bool:
    interval = (candidate["start_seconds"], candidate["end_seconds"])
    return any(
        interval_iou(
            interval,
            (existing["start_seconds"], existing["end_seconds"]),
        )
        > maximum_overlap_iou
        for existing in selected
    )


def sample_video_candidates(
    candidates: list[dict[str, Any]],
    samples_per_video: int = 6,
    seed: int = 42,
    maximum_overlap_iou: float = 0.50,
) -> list[dict[str, Any]]:
    """Sample equal score bands while spreading selections over time and duration."""
    if samples_per_video <= 0 or samples_per_video % len(SCORE_BANDS):
        raise ValueError("samples_per_video must be a positive multiple of 3")
    if not 0 <= maximum_overlap_iou <= 1:
        raise ValueError("maximum_overlap_iou must be between 0 and 1")
    if not candidates:
        return []

    video_ids = {candidate["video_id"] for candidate in candidates}
    if len(video_ids) != 1:
        raise ValueError("sample_video_candidates expects candidates from one video")

    bands = partition_score_bands(candidates)
    quota_per_band = samples_per_video // len(SCORE_BANDS)
    for band, pool in bands.items():
        if len(pool) < quota_per_band:
            raise ValueError(f"Not enough {band}-band candidates to sample {quota_per_band}")

    video_id = next(iter(video_ids))
    video_end = max(float(candidate["end_seconds"]) for candidate in candidates)
    band_schedule = [band for band in SCORE_BANDS for _ in range(quota_per_band)]
    random.Random(f"{seed}:{video_id}").shuffle(band_schedule)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for slot, band in enumerate(band_schedule):
        target_position = (slot + 0.5) / samples_per_video
        target_midpoint = target_position * video_end
        target_duration = TARGET_DURATIONS[slot % len(TARGET_DURATIONS)]
        eligible = [
            candidate
            for candidate in bands[band]
            if candidate["candidate_id"] not in selected_ids
            and not _overlaps_selected(candidate, selected, maximum_overlap_iou)
        ]
        if not eligible:
            raise ValueError(
                f"Could not sample {samples_per_video} non-overlapping candidates for {video_id}"
            )

        chosen = min(
            eligible,
            key=lambda candidate: (
                abs((candidate["start_seconds"] + candidate["end_seconds"]) / 2 - target_midpoint)
                / video_end
                + 0.25
                * abs(candidate["duration_seconds"] - target_duration)
                / max(TARGET_DURATIONS),
                candidate["candidate_id"],
            ),
        )
        selected.append(
            {
                **chosen,
                "sampling_score_band": band,
                "sampling_target_position": round(target_position, 4),
                "sampling_target_duration": target_duration,
            }
        )
        selected_ids.add(chosen["candidate_id"])

    return sorted(selected, key=lambda item: item["start_seconds"])


def build_annotation_queue(
    candidates: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    samples_per_video: int = 6,
    seed: int = 42,
    maximum_overlap_iou: float = 0.50,
    excluded_video_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Sample candidates from videos that do not yet have human annotations."""
    unavailable_video_ids = {annotation["video_id"] for annotation in annotations}
    unavailable_video_ids.update(excluded_video_ids or set())
    scored = score_candidates(candidates)
    candidates_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in scored:
        if candidate["video_id"] not in unavailable_video_ids:
            candidates_by_video[candidate["video_id"]].append(candidate)

    queue: list[dict[str, Any]] = []
    for video_id in sorted(candidates_by_video):
        sampled = sample_video_candidates(
            candidates_by_video[video_id],
            samples_per_video=samples_per_video,
            seed=seed,
            maximum_overlap_iou=maximum_overlap_iou,
        )
        for index, candidate in enumerate(sampled, start=1):
            queue.append(
                {
                    "annotation_id": f"{video_id}_sample_{index:02d}",
                    "candidate_id": candidate["candidate_id"],
                    "video_id": video_id,
                    "start_seconds": candidate["start_seconds"],
                    "end_seconds": candidate["end_seconds"],
                    "duration_seconds": candidate["duration_seconds"],
                    "transcript_text": candidate["text"],
                    "sampling": {
                        "strategy": "heuristic_stratified_temporal_v1",
                        "score_band": candidate["sampling_score_band"],
                        "proxy_score": candidate["transcript_heuristic_score"],
                        "target_position": candidate["sampling_target_position"],
                        "target_duration": candidate["sampling_target_duration"],
                        "seed": seed,
                    },
                    "labels": {
                        "hook": None,
                        "completeness": None,
                        "payoff": None,
                        "clarity": None,
                        "technically_exportable": None,
                        "notes": "",
                    },
                }
            )
    return queue


def summarize_queue(queue: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact coverage statistics for an annotation queue."""
    durations = [item["duration_seconds"] for item in queue]
    videos = Counter(item["video_id"] for item in queue)
    bands = Counter(item["sampling"]["score_band"] for item in queue)
    return {
        "annotation_tasks": len(queue),
        "videos": len(videos),
        "tasks_per_video": dict(sorted(videos.items())),
        "score_bands": {band: bands[band] for band in SCORE_BANDS},
        "duration_seconds": {
            "minimum": min(durations) if durations else None,
            "maximum": max(durations) if durations else None,
            "mean": sum(durations) / len(durations) if durations else None,
            "total": sum(durations),
        },
    }


def main() -> None:
    """Generate the next diverse annotation queue from local candidates."""
    parser = argparse.ArgumentParser(description="Sample CreatorCut clips for human annotation")
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/candidates.jsonl"))
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/annotations/seed_labels.jsonl")
    )
    parser.add_argument(
        "--transcripts-dir", type=Path, default=Path("data/processed/transcripts")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/annotation_queue.jsonl")
    )
    parser.add_argument("--samples-per-video", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maximum-overlap-iou", type=float, default=0.50)
    parser.add_argument(
        "--include-warned-transcripts",
        action="store_true",
        help="Include videos whose transcript validator reports quality warnings",
    )
    args = parser.parse_args()

    transcript_validation = validate_transcript_directory(args.transcripts_dir)
    if transcript_validation["errors"]:
        raise SystemExit("Transcript validation contains errors; fix them before sampling")
    warned_video_ids = {
        result["video_id"]
        for result in transcript_validation["transcripts"]
        if result["warnings"]
    }
    excluded_video_ids = set() if args.include_warned_transcripts else warned_video_ids
    queue = build_annotation_queue(
        load_jsonl(args.candidates),
        load_jsonl(args.annotations),
        samples_per_video=args.samples_per_video,
        seed=args.seed,
        maximum_overlap_iou=args.maximum_overlap_iou,
        excluded_video_ids=excluded_video_ids,
    )
    write_jsonl(queue, args.output)
    summary = summarize_queue(queue)
    summary["excluded_transcript_warnings"] = sorted(excluded_video_ids)
    print(json.dumps(summary, indent=2))
    print(f"Wrote annotation queue to {args.output}")


if __name__ == "__main__":
    main()
