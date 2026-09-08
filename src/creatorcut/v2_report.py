"""Aggregate honest v2 development metrics without publishing private labels."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from creatorcut.dataset import load_jsonl


def _percentile_ranks(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [0.0] * len(values)
    return [sum(other < value for other in values) / (len(values) - 1) for value in values]


def _selection_metrics(
    records: list[dict[str, Any]], scores: dict[str, float], top_k: int = 3
) -> dict[str, Any]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_video[record["video_id"]].append(record)
    pairwise_credit = 0.0
    comparable_pairs = 0
    top_hits = 0
    top_regrets: list[float] = []
    top_k_hits = 0
    top_k_regrets: list[float] = []
    for rows in by_video.values():
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                actual_delta = (
                    float(left["targets"]["quality_score"])
                    - float(right["targets"]["quality_score"])
                )
                if math.isclose(actual_delta, 0.0):
                    continue
                score_delta = scores[left["annotation_id"]] - scores[right["annotation_id"]]
                comparable_pairs += 1
                if math.isclose(score_delta, 0.0):
                    pairwise_credit += 0.5
                elif (actual_delta > 0) == (score_delta > 0):
                    pairwise_credit += 1.0
        ordered = sorted(
            rows,
            key=lambda row: scores[row["annotation_id"]],
            reverse=True,
        )
        best = max(float(row["targets"]["quality_score"]) for row in rows)
        selected = float(ordered[0]["targets"]["quality_score"])
        selected_k = max(
            float(row["targets"]["quality_score"]) for row in ordered[:top_k]
        )
        top_hits += math.isclose(selected, best)
        top_regrets.append(best - selected)
        top_k_hits += math.isclose(selected_k, best)
        top_k_regrets.append(best - selected_k)
    return {
        "pairwise_accuracy": pairwise_credit / comparable_pairs,
        "comparable_pairs": comparable_pairs,
        "top_1_hit_rate": top_hits / len(by_video),
        "mean_top_1_regret": sum(top_regrets) / len(top_regrets),
        f"top_{top_k}_hit_rate": top_k_hits / len(by_video),
        f"mean_top_{top_k}_regret": sum(top_k_regrets) / len(top_k_regrets),
    }


def _random_expectation(records: list[dict[str, Any]], top_k: int = 3) -> dict[str, Any]:
    by_video: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_video[record["video_id"]].append(float(record["targets"]["quality_score"]))
    hit_rates: list[float] = []
    regrets: list[float] = []
    top_k_hit_rates: list[float] = []
    top_k_regrets: list[float] = []
    for values in by_video.values():
        best = max(values)
        hit_rates.append(sum(math.isclose(value, best) for value in values) / len(values))
        regrets.append(sum(best - value for value in values) / len(values))
        combinations = list(itertools.combinations(range(len(values)), min(top_k, len(values))))
        top_k_values = [max(values[index] for index in choice) for choice in combinations]
        top_k_hit_rates.append(
            sum(math.isclose(value, best) for value in top_k_values) / len(top_k_values)
        )
        top_k_regrets.append(sum(best - value for value in top_k_values) / len(top_k_values))
    return {
        "pairwise_accuracy": 0.5,
        "top_1_hit_rate": sum(hit_rates) / len(hit_rates),
        "mean_top_1_regret": sum(regrets) / len(regrets),
        f"top_{top_k}_hit_rate": sum(top_k_hit_rates) / len(top_k_hit_rates),
        f"mean_top_{top_k}_regret": sum(top_k_regrets) / len(top_k_regrets),
    }


def build_v2_summary(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    promoted_review_ids: set[str],
    boundary_evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Report LOOV moment-ranking and grouped boundary experiments."""
    record_by_id = {record["annotation_id"]: record for record in records}
    prediction_by_id = {row["annotation_id"]: row for row in predictions}
    if len(record_by_id) != len(records) or len(prediction_by_id) != len(predictions):
        raise ValueError("records and predictions require unique annotation IDs")
    if set(record_by_id) != set(prediction_by_id):
        raise ValueError("prediction identities do not exactly match training records")

    pointwise = {
        annotation_id: float(row["pointwise_hybrid_ridge_score"])
        for annotation_id, row in prediction_by_id.items()
    }
    pairwise = {
        annotation_id: float(row["pairwise_hybrid_logistic_score"])
        for annotation_id, row in prediction_by_id.items()
    }
    ensemble: dict[str, float] = {}
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_video[record["video_id"]].append(record)
    for rows in by_video.values():
        pointwise_ranks = _percentile_ranks([pointwise[row["annotation_id"]] for row in rows])
        pairwise_ranks = _percentile_ranks([pairwise[row["annotation_id"]] for row in rows])
        for row, first, second in zip(rows, pointwise_ranks, pairwise_ranks, strict=True):
            ensemble[row["annotation_id"]] = (first + second) / 2.0

    def cohort(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "videos": len({record["video_id"] for record in selected}),
            "clips": len(selected),
            "random_expectation": _random_expectation(selected),
            "pointwise_hybrid_ridge": _selection_metrics(selected, pointwise),
            "pairwise_hybrid_logistic": _selection_metrics(selected, pairwise),
            "equal_rank_ensemble": _selection_metrics(selected, ensemble),
        }

    original = [record for record in records if record["annotation_id"] not in promoted_review_ids]
    promoted = [record for record in records if record["annotation_id"] in promoted_review_ids]
    if not original or not promoted:
        raise ValueError("both original and promoted cohorts must be present")
    return {
        "schema": "creatorcut_v2_development_report_v1",
        "status": "development-only; not an external holdout result",
        "moment_ranking": {
            "target": (
                "attainable quality: edited-version scores when available, otherwise original "
                "interval scores"
            ),
            "evaluation": "leave-one-video-out cross-validation across 32 source videos",
            "ensemble": "equal average of within-video pointwise and pairwise percentile ranks",
            "all": cohort(records),
            "original_cohort": cohort(original),
            "promoted_cohort": cohort(promoted),
        },
        "boundary_selection": {
            "evaluation": boundary_evaluation["cross_validation"]["strategy"],
            "metrics": boundary_evaluation["metrics"],
            "conclusion": (
                "Candidate coverage is strong, but the learned selector does not beat no-change. "
                "Semantic context features are required before deployment."
            ),
        },
        "decision": {
            "promote_equal_rank_ensemble_for_next_external_test": True,
            "deploy_learned_boundary_selector": False,
            "retain_boundary_candidate_generator": True,
            "future_claim_requires_new_unseen_videos": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CreatorCut's public-safe v2 summary")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--promoted-reviews", type=Path, required=True)
    parser.add_argument("--boundary-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reviews = load_jsonl(args.promoted_reviews)
    summary = build_v2_summary(
        load_jsonl(args.records),
        load_jsonl(args.predictions),
        {row["annotation_id"] for row in reviews},
        json.loads(args.boundary_evaluation.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
