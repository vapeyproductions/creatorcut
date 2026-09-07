"""Build a private review queue from out-of-fold top-clip ranking mistakes."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from creatorcut.candidates import CORE_SCORE_FIELDS, write_jsonl
from creatorcut.dataset import load_jsonl

DEFAULT_PREDICTION_FIELD = "pointwise_hybrid_ridge_score"


def _unique_index(
    records: list[dict[str, Any]], key: str, source: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{source} contains a record without a valid {key}")
        if value in indexed:
            raise ValueError(f"{source} contains duplicate {key}: {value}")
        indexed[value] = record
    return indexed


def _clip_payload(record: dict[str, Any], prediction: float) -> dict[str, Any]:
    features = record["features"]
    text = record["transcript_text"]
    lowered = text.lower()
    return {
        "annotation_id": record["annotation_id"],
        "candidate_id": record["candidate_id"],
        "start_seconds": float(record["start_seconds"]),
        "end_seconds": float(record["end_seconds"]),
        "duration_seconds": float(record["duration_seconds"]),
        "transcript_text": text,
        "quality_score": float(record["targets"]["quality_score"]),
        "component_scores": {
            field: float(record["targets"][field]) for field in CORE_SCORE_FIELDS
        },
        "model_score": float(prediction),
        "automatic_signals": {
            "context_dependent_first_word": float(features["clean_start"]) < 0.5,
            "punctuation_complete_ending": float(features["complete_end"]) >= 0.5,
            "longer_than_45_seconds": float(record["duration_seconds"]) > 45.0,
            "low_lexical_hook_signal": float(features["hook_signal"]) < 0.25,
            "intro_or_outro_phrase": float(features["intro_outro"]) >= 0.5,
            "possible_ad_language": any(
                phrase in lowered
                for phrase in (
                    "brought to you by",
                    "promo code",
                    "sponsor",
                    "sponsored by",
                )
            ),
            "possible_music_marker": any(
                marker in lowered for marker in ("[music]", "♪", "intro music")
            ),
        },
    }


def build_failure_cases(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    prediction_field: str = DEFAULT_PREDICTION_FIELD,
) -> list[dict[str, Any]]:
    """Pair each mistaken out-of-fold top selection with a human-best candidate."""
    if not records:
        raise ValueError("Training records are empty")
    predictions_by_id = _unique_index(predictions, "annotation_id", "predictions")
    record_ids = {record.get("annotation_id") for record in records}
    if set(predictions_by_id) != record_ids:
        missing = sorted(record_ids - predictions_by_id.keys())
        extra = sorted(predictions_by_id.keys() - record_ids)
        raise ValueError(f"Prediction IDs do not match records; missing={missing}, extra={extra}")

    by_video: dict[str, list[tuple[int, dict[str, Any], float]]] = defaultdict(list)
    for index, record in enumerate(records):
        prediction = predictions_by_id[record["annotation_id"]]
        if prediction.get("video_id") != record["video_id"]:
            raise ValueError(f"{record['annotation_id']}: prediction video_id does not match")
        score = prediction.get(prediction_field)
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ValueError(
                f"{record['annotation_id']}: prediction field {prediction_field} is not numeric"
            )
        by_video[record["video_id"]].append((index, record, float(score)))

    cases: list[dict[str, Any]] = []
    for video_id, candidates in sorted(by_video.items()):
        selected = max(candidates, key=lambda item: (item[2], -item[0]))
        best_quality = max(float(item[1]["targets"]["quality_score"]) for item in candidates)
        selected_quality = float(selected[1]["targets"]["quality_score"])
        if math.isclose(selected_quality, best_quality):
            continue
        best_candidates = [
            item
            for item in candidates
            if math.isclose(float(item[1]["targets"]["quality_score"]), best_quality)
        ]
        human_best = max(best_candidates, key=lambda item: (item[2], -item[0]))
        target_gaps = {
            field: float(human_best[1]["targets"][field] - selected[1]["targets"][field])
            for field in CORE_SCORE_FIELDS
        }
        cases.append(
            {
                "analysis_id": f"{video_id}_top1_failure",
                "video_id": video_id,
                "prediction_field": prediction_field,
                "regret": best_quality - selected_quality,
                "target_gaps": target_gaps,
                "model_selected": _clip_payload(selected[1], selected[2]),
                "human_best": _clip_payload(human_best[1], human_best[2]),
            }
        )
    return cases


def summarize_failures(
    cases: list[dict[str, Any]], total_videos: int
) -> dict[str, Any]:
    """Aggregate failure prevalence without exposing private transcript text."""
    if total_videos <= 0 or len(cases) > total_videos:
        raise ValueError("total_videos must cover every failure case")
    regrets = [float(case["regret"]) for case in cases]
    target_gaps = {
        field: [float(case["target_gaps"][field]) for case in cases]
        for field in CORE_SCORE_FIELDS
    }
    selected_signals = Counter(
        signal
        for case in cases
        for signal, present in case["model_selected"]["automatic_signals"].items()
        if present
    )
    return {
        "videos": total_videos,
        "top_1_hits": total_videos - len(cases),
        "top_1_failures": len(cases),
        "top_1_hit_rate": (total_videos - len(cases)) / total_videos,
        "mean_regret_on_failures": sum(regrets) / len(regrets) if regrets else 0.0,
        "regret_distribution": {
            str(value): count for value, count in sorted(Counter(regrets).items())
        },
        "mean_human_best_minus_selected_target": {
            field: sum(values) / len(values) if values else 0.0
            for field, values in target_gaps.items()
        },
        "selected_clip_automatic_signal_counts": dict(sorted(selected_signals.items())),
        "interpretation": (
            "Automatic signals are hypotheses for manual review, not validated failure labels."
        ),
    }


def main() -> None:
    """Create the private top-selection failure queue and aggregate summary."""
    parser = argparse.ArgumentParser(description="Build CreatorCut's top-clip failure review")
    parser.add_argument(
        "--records", type=Path, default=Path("data/processed/training_clips.jsonl")
    )
    parser.add_argument(
        "--predictions", type=Path, default=Path("data/processed/pairwise_predictions.jsonl")
    )
    parser.add_argument("--prediction-field", default=DEFAULT_PREDICTION_FIELD)
    parser.add_argument(
        "--cases", type=Path, default=Path("data/processed/failure_cases.jsonl")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("data/processed/failure_summary.json")
    )
    args = parser.parse_args()

    records = load_jsonl(args.records)
    cases = build_failure_cases(
        records, load_jsonl(args.predictions), prediction_field=args.prediction_field
    )
    summary = summarize_failures(cases, len({record["video_id"] for record in records}))
    write_jsonl(cases, args.cases)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {len(cases)} private review cases to {args.cases}")


if __name__ == "__main__":
    main()
