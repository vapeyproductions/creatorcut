"""Create an aggregate public report from the registered external holdout evaluation."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from creatorcut.candidates import CORE_SCORE_FIELDS
from creatorcut.dataset import load_jsonl

REPORT_SCHEMA = "creatorcut_external_holdout_summary_v1"


def percentile_interval(values: list[float]) -> list[float]:
    """Return a deterministic percentile interval from bootstrap draws."""
    ordered = sorted(values)
    return [ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]]


def _video_summary(records: list[dict[str, Any]]) -> dict[str, float | int | bool]:
    best_quality = max(float(record["actual_quality_score"]) for record in records)
    predicted_order = sorted(
        records,
        key=lambda record: (-float(record["predicted_quality_score"]), record["annotation_id"]),
    )
    selected_quality = float(predicted_order[0]["actual_quality_score"])
    top_three_quality = max(
        float(record["actual_quality_score"]) for record in predicted_order[:3]
    )
    best_ties = sum(
        math.isclose(float(record["actual_quality_score"]), best_quality)
        for record in records
    )
    random_top_three_regrets = [
        best_quality
        - max(float(records[index]["actual_quality_score"]) for index in combination)
        for combination in itertools.combinations(range(len(records)), min(3, len(records)))
    ]
    random_top_three_hit = statistics.mean(
        math.isclose(regret, 0.0) for regret in random_top_three_regrets
    )
    return {
        "top_1_hit": math.isclose(selected_quality, best_quality),
        "top_1_regret": best_quality - selected_quality,
        "top_3_hit": math.isclose(top_three_quality, best_quality),
        "top_3_regret": best_quality - top_three_quality,
        "random_top_1_expected_hit": best_ties / len(records),
        "random_top_1_expected_regret": best_quality
        - statistics.mean(float(record["actual_quality_score"]) for record in records),
        "random_top_3_expected_hit": random_top_three_hit,
        "random_top_3_expected_regret": statistics.mean(random_top_three_regrets),
    }


def build_public_holdout_summary(
    evaluation: dict[str, Any],
    reviews: list[dict[str, Any]],
    bootstrap_iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate registered metrics, uncertainty, and labeled exploratory diagnostics."""
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    records = evaluation.get("records", [])
    review_by_id = {review["annotation_id"]: review for review in reviews}
    if len(review_by_id) != len(reviews):
        raise ValueError("Reviews contain duplicate annotation IDs")
    if {record["annotation_id"] for record in records} != set(review_by_id):
        raise ValueError("Evaluation and reviews contain different annotation IDs")

    records_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_video[record["video_id"]].append(record)
    video_summaries = [_video_summary(group) for group in records_by_video.values()]
    rng = random.Random(seed)
    top_1_draws: list[float] = []
    regret_draws: list[float] = []
    for _ in range(bootstrap_iterations):
        sample = [rng.choice(video_summaries) for _ in video_summaries]
        top_1_draws.append(statistics.mean(bool(row["top_1_hit"]) for row in sample))
        regret_draws.append(statistics.mean(float(row["top_1_regret"]) for row in sample))

    target_predictions = [
        float(value)
        for record in records
        for value in record["predicted_targets"].values()
    ]
    boundary_edits = [review for review in reviews if review.get("boundary_edit")]
    quality_deltas = []
    target_deltas: dict[str, list[float]] = {field: [] for field in CORE_SCORE_FIELDS}
    for review in boundary_edits:
        edit = review["boundary_edit"]
        for field in CORE_SCORE_FIELDS:
            target_deltas[field].append(float(edit["scores"][field]) - float(review[field]))
        quality_deltas.append(
            statistics.mean(float(edit["scores"][field]) for field in CORE_SCORE_FIELDS)
            - statistics.mean(float(review[field]) for field in CORE_SCORE_FIELDS)
        )

    mean = statistics.mean
    boundary_diagnostics: dict[str, Any] = {
        "edited_clips": len(boundary_edits),
        "unedited_clips": len(reviews) - len(boundary_edits),
        "interpretation": (
            "Conditional on the reviewer choosing to edit; not an unbiased estimate of the "
            "effect of editing every clip."
        ),
    }
    if boundary_edits:
        boundary_diagnostics.update(
            {
                "mean_quality_change_when_an_edit_was_chosen": mean(quality_deltas),
                "edits_with_higher_quality": sum(delta > 0 for delta in quality_deltas),
                "edits_with_equal_quality": sum(
                    math.isclose(delta, 0.0) for delta in quality_deltas
                ),
                "edits_with_lower_quality": sum(delta < 0 for delta in quality_deltas),
                "mean_target_changes": {
                    field: mean(target_deltas[field]) for field in CORE_SCORE_FIELDS
                },
                "mean_start_adjustment_seconds": mean(
                    float(review["boundary_edit"]["start_adjustment_seconds"])
                    for review in boundary_edits
                ),
                "mean_end_adjustment_seconds": mean(
                    float(review["boundary_edit"]["end_adjustment_seconds"])
                    for review in boundary_edits
                ),
            }
        )
    return {
        "report_schema": REPORT_SCHEMA,
        "status": "reported_before_any_retraining",
        "evaluation_schema": evaluation["evaluation_schema"],
        "evaluated_at": evaluation["evaluated_at"],
        "prediction_commitment_sha256": evaluation["prediction_commitment_sha256"],
        "frozen_model_sha256": evaluation["frozen_model_sha256"],
        "development_dataset_sha256": evaluation["development_dataset_sha256"],
        "dataset": evaluation["dataset"],
        "registered_metrics": {
            "regression": evaluation["regression"],
            "ranking": evaluation["ranking"],
        },
        "video_cluster_bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": seed,
            "top_1_hit_rate_95_percentile_interval": percentile_interval(top_1_draws),
            "mean_top_1_regret_95_percentile_interval": percentile_interval(regret_draws),
        },
        "exploratory_product_metrics": {
            "top_3_hit_rate": mean(bool(row["top_3_hit"]) for row in video_summaries),
            "mean_top_3_regret": mean(float(row["top_3_regret"]) for row in video_summaries),
            "random_top_1_expected_hit_rate": mean(
                float(row["random_top_1_expected_hit"]) for row in video_summaries
            ),
            "random_top_1_expected_regret": mean(
                float(row["random_top_1_expected_regret"]) for row in video_summaries
            ),
            "random_top_3_expected_hit_rate": mean(
                float(row["random_top_3_expected_hit"]) for row in video_summaries
            ),
            "random_top_3_expected_regret": mean(
                float(row["random_top_3_expected_regret"]) for row in video_summaries
            ),
        },
        "calibration_diagnostics": {
            "actual_quality_mean": mean(
                float(record["actual_quality_score"]) for record in records
            ),
            "predicted_quality_mean": mean(
                float(record["predicted_quality_score"]) for record in records
            ),
            "target_predictions_outside_1_to_5": sum(
                value < 1.0 or value > 5.0 for value in target_predictions
            ),
            "target_prediction_count": len(target_predictions),
            "target_prediction_minimum": min(target_predictions),
            "target_prediction_maximum": max(target_predictions),
        },
        "conditional_boundary_edit_diagnostics": boundary_diagnostics,
    }


def main() -> None:
    """Write the public aggregate report without exposing clip-level human labels."""
    parser = argparse.ArgumentParser(description="Summarize CreatorCut external holdout v1")
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("data/processed/holdout_v1/evaluation.json"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/processed/holdout_v1/annotation_reviews.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/holdout_v1/evaluation_summary.json")
    )
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    summary = build_public_holdout_summary(evaluation, load_jsonl(args.reviews))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
