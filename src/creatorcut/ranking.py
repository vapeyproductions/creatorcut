"""Pairwise learning-to-rank for CreatorCut clip selection."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from creatorcut.candidates import write_jsonl
from creatorcut.dataset import load_jsonl
from creatorcut.semantic import EMBEDDING_SCHEMA, embedding_matrix_for_records
from creatorcut.training import (
    MODEL_FEATURE_FIELDS,
    build_group_folds,
    cross_validate_ridge,
    ranking_metrics,
)

PAIRWISE_MODEL_SCHEMA = "standardized_pairwise_logistic_v1"


def _quality_scores(records: list[dict[str, Any]]) -> np.ndarray:
    scores = np.asarray(
        [float(record["targets"]["quality_score"]) for record in records], dtype=float
    )
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Training records require finite quality scores")
    return scores


def build_preference_matrix(
    records: list[dict[str, Any]],
    standardized_features: np.ndarray,
    quality_scores: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    """Convert within-video non-tied ratings into correctly oriented feature differences."""
    if standardized_features.ndim != 2 or len(standardized_features) != len(records):
        raise ValueError("Expected one feature row per training record")
    if len(quality_scores) != len(records):
        raise ValueError("Expected one quality score per training record")

    by_video: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_video[records[index]["video_id"]].append(index)

    differences: list[np.ndarray] = []
    for video_indices in by_video.values():
        for left_position, left in enumerate(video_indices):
            for right in video_indices[left_position + 1 :]:
                preference = quality_scores[left] - quality_scores[right]
                if math.isclose(float(preference), 0.0, abs_tol=1e-12):
                    continue
                direction = 1.0 if preference > 0 else -1.0
                differences.append(
                    direction * (standardized_features[left] - standardized_features[right])
                )
    if not differences:
        raise ValueError("Training split contains no comparable within-video preferences")
    return np.asarray(differences, dtype=float)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _pairwise_objective(differences: np.ndarray, weights: np.ndarray, l2: float) -> float:
    margins = differences @ weights
    return float(np.logaddexp(0.0, -margins).mean() + 0.5 * l2 * (weights @ weights))


def fit_pairwise_logistic(
    records: list[dict[str, Any]],
    features: np.ndarray,
    quality_scores: np.ndarray,
    train_indices: list[int] | None = None,
    l2: float = 0.1,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Fit a linear Bradley–Terry ranker with L2-regularized logistic loss."""
    if features.ndim != 2 or len(features) != len(records):
        raise ValueError("Expected one feature row per training record")
    if not np.isfinite(features).all():
        raise ValueError("Training features must be finite")
    if l2 <= 0:
        raise ValueError("l2 must be positive")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    indices = list(range(len(records))) if train_indices is None else list(train_indices)
    if not indices:
        raise ValueError("Training indices are empty")

    train_features = features[indices]
    means = train_features.mean(axis=0)
    scales = train_features.std(axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = (features - means) / scales
    differences = build_preference_matrix(records, standardized, quality_scores, indices)

    weights = np.zeros(features.shape[1], dtype=float)
    identity = np.eye(features.shape[1], dtype=float)
    converged = False
    completed_iterations = 0
    for iteration in range(1, maximum_iterations + 1):
        margins = differences @ weights
        mistakes = _sigmoid(-margins)
        gradient = -(differences.T @ mistakes) / len(differences) + l2 * weights
        curvature = _sigmoid(margins) * mistakes
        hessian = (differences.T * curvature) @ differences / len(differences) + l2 * identity
        step = np.linalg.solve(hessian, gradient)

        current_objective = _pairwise_objective(differences, weights, l2)
        step_scale = 1.0
        while step_scale >= 1e-8:
            candidate = weights - step_scale * step
            if _pairwise_objective(differences, candidate, l2) < current_objective:
                break
            step_scale *= 0.5
        if step_scale < 1e-8:
            completed_iterations = iteration
            converged = float(np.linalg.norm(gradient)) <= tolerance
            break

        weights = candidate
        completed_iterations = iteration
        if float(np.linalg.norm(step_scale * step)) <= tolerance * (
            1.0 + float(np.linalg.norm(weights))
        ):
            converged = True
            break

    return {
        "feature_means": means,
        "feature_scales": scales,
        "weights": weights,
        "l2": l2,
        "training_preferences": len(differences),
        "iterations": completed_iterations,
        "converged": converged,
        "objective": _pairwise_objective(differences, weights, l2),
    }


def predict_pairwise_scores(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Return unbounded utilities whose within-video ordering defines the ranking."""
    standardized = (features - model["feature_means"]) / model["feature_scales"]
    return standardized @ model["weights"]


def _ranking_summary(
    groups: list[list[int]], actual: list[float], predicted: list[float]
) -> tuple[float, float, float]:
    pairwise_credit = 0.0
    comparable_pairs = 0
    top_hits = 0
    regrets: list[float] = []
    for indices in groups:
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                actual_difference = actual[left] - actual[right]
                if actual_difference == 0:
                    continue
                predicted_difference = predicted[left] - predicted[right]
                comparable_pairs += 1
                if predicted_difference == 0:
                    pairwise_credit += 0.5
                elif (actual_difference > 0) == (predicted_difference > 0):
                    pairwise_credit += 1.0
        selected = max(indices, key=lambda index: (predicted[index], -index))
        best_actual = max(actual[index] for index in indices)
        selected_actual = actual[selected]
        top_hits += math.isclose(selected_actual, best_actual)
        regrets.append(best_actual - selected_actual)
    return (
        pairwise_credit / comparable_pairs if comparable_pairs else 0.0,
        top_hits / len(groups),
        sum(regrets) / len(regrets),
    )


def paired_ranking_bootstrap(
    records: list[dict[str, Any]],
    actual: list[float],
    reference: list[float],
    challenger: list[float],
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare ranking predictions while resampling whole source videos."""
    if len({len(records), len(actual), len(reference), len(challenger)}) != 1 or not records:
        raise ValueError("Expected equal-length, non-empty bootstrap inputs")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    by_video: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_video[record["video_id"]].append(index)
    groups = list(by_video.values())
    reference_observed = _ranking_summary(groups, actual, reference)
    challenger_observed = _ranking_summary(groups, actual, challenger)
    observed = [
        challenger_value - reference_value
        for reference_value, challenger_value in zip(
            reference_observed, challenger_observed, strict=True
        )
    ]

    sampled_deltas: list[list[float]] = [[] for _ in observed]
    generator = random.Random(seed)
    for _ in range(iterations):
        sampled_groups = [generator.choice(groups) for _ in groups]
        reference_sample = _ranking_summary(sampled_groups, actual, reference)
        challenger_sample = _ranking_summary(sampled_groups, actual, challenger)
        for index, (reference_value, challenger_value) in enumerate(
            zip(reference_sample, challenger_sample, strict=True)
        ):
            sampled_deltas[index].append(challenger_value - reference_value)

    names = ("pairwise_accuracy", "top_1_hit_rate", "mean_top_1_regret")
    comparisons: dict[str, Any] = {}
    for name, estimate, values in zip(names, observed, sampled_deltas, strict=True):
        values_array = np.asarray(values)
        improved = values_array < 0 if name == "mean_top_1_regret" else values_array > 0
        comparisons[f"{name}_delta"] = {
            "estimate": estimate,
            "confidence_interval_95": np.quantile(values_array, [0.025, 0.975]).tolist(),
            "bootstrap_probability_improved": float(improved.mean()),
        }
    return {
        "method": "paired cluster bootstrap resampling video_id",
        "iterations": iterations,
        "seed": seed,
        "challenger_minus_reference": comparisons,
    }


def _hybrid_feature_data(
    records: list[dict[str, Any]], artifact: dict[str, Any]
) -> tuple[np.ndarray, tuple[str, ...], str]:
    embeddings = embedding_matrix_for_records(records, artifact)
    embedding_fields = tuple(f"embedding_{index:03d}" for index in range(embeddings.shape[1]))
    handcrafted = np.asarray(
        [[record["features"][field] for field in MODEL_FEATURE_FIELDS] for record in records],
        dtype=float,
    )
    source_schemas = {record.get("feature_schema") for record in records}
    if len(source_schemas) != 1:
        raise ValueError(f"Expected one source feature schema, found {source_schemas}")
    source_schema = next(iter(source_schemas))
    if not isinstance(source_schema, str):
        raise ValueError("Training records require a feature schema")
    return (
        np.column_stack([handcrafted, embeddings]),
        (*MODEL_FEATURE_FIELDS, *embedding_fields),
        f"{source_schema}_plus_{EMBEDDING_SCHEMA}",
    )


def cross_validate_pairwise_ranker(
    records: list[dict[str, Any]],
    artifact: dict[str, Any],
    n_splits: int = 4,
    seed: int = 42,
    l2: float = 0.1,
    ridge_alpha: float = 10.0,
    bootstrap_iterations: int = 10_000,
) -> dict[str, Any]:
    """Compare pointwise and pairwise objectives on one representation and grouped folds."""
    features, feature_fields, feature_schema = _hybrid_feature_data(records, artifact)
    quality_scores = _quality_scores(records)
    folds = build_group_folds(records, n_splits=n_splits, seed=seed)
    pairwise_predictions = np.zeros(len(records), dtype=float)
    prediction_folds = [0] * len(records)
    fold_summaries: list[dict[str, Any]] = []

    for fold in folds:
        model = fit_pairwise_logistic(
            records,
            features,
            quality_scores,
            train_indices=fold["train_indices"],
            l2=l2,
        )
        test = np.asarray(fold["test_indices"], dtype=int)
        pairwise_predictions[test] = predict_pairwise_scores(features[test], model)
        for index in test:
            prediction_folds[int(index)] = fold["fold"]
        fold_summaries.append(
            {
                "fold": fold["fold"],
                "train_clip_count": len(fold["train_indices"]),
                "test_clip_count": len(fold["test_indices"]),
                "training_preferences": model["training_preferences"],
                "optimizer_iterations": model["iterations"],
                "optimizer_converged": model["converged"],
                "train_videos": fold["train_videos"],
                "test_videos": fold["test_videos"],
            }
        )

    hybrid_records = [
        {
            **record,
            "feature_schema": feature_schema,
            "features": dict(zip(feature_fields, row.tolist(), strict=True)),
        }
        for record, row in zip(records, features, strict=True)
    ]
    pointwise = cross_validate_ridge(
        hybrid_records,
        n_splits=n_splits,
        seed=seed,
        alpha=ridge_alpha,
        feature_fields=feature_fields,
        feature_schema=feature_schema,
        experiment="grouped_pointwise_hybrid_ridge_for_objective_ablation_v1",
    )
    pointwise_predictions = np.asarray(
        [prediction["ridge_quality_score"] for prediction in pointwise["predictions"]]
    )
    actual = quality_scores.tolist()
    pointwise_scores = pointwise_predictions.tolist()
    pairwise_scores = pairwise_predictions.tolist()
    final_model = fit_pairwise_logistic(records, features, quality_scores, l2=l2)
    serializable_model = {
        "model_schema": PAIRWISE_MODEL_SCHEMA,
        "feature_schema": feature_schema,
        "feature_fields": list(feature_fields),
        "feature_means": final_model["feature_means"].tolist(),
        "feature_scales": final_model["feature_scales"].tolist(),
        "standardized_weights": final_model["weights"].tolist(),
        "l2": l2,
        "training_preferences": final_model["training_preferences"],
        "optimizer": {
            "method": "Newton-Raphson with backtracking line search",
            "iterations": final_model["iterations"],
            "converged": final_model["converged"],
            "objective": final_model["objective"],
        },
    }

    return {
        "experiment": "pointwise_vs_pairwise_hybrid_objective_ablation_v1",
        "dataset": {
            "clips": len(records),
            "videos": len({record["video_id"] for record in records}),
            "comparable_preferences": ranking_metrics(records, actual, actual)[
                "comparable_pairs"
            ],
            "technically_exportable": sum(
                bool(record["technically_exportable"]) for record in records
            ),
            "technical_issue": sum(
                not bool(record["technically_exportable"]) for record in records
            ),
        },
        "representation": {
            "schema": feature_schema,
            "features": len(feature_fields),
            "handcrafted_features": len(MODEL_FEATURE_FIELDS),
            "semantic_features": artifact["metadata"]["dimension"],
            "sampler_proxy_included": False,
        },
        "target": {
            "quality_score": "unweighted mean of hook, completeness, payoff, and clarity",
            "preferences": "all non-tied within-video quality-score comparisons",
            "pair_weighting": "one equal-weight observation per comparison",
            "technically_exportable": "retained as a separate gate",
        },
        "cross_validation": {
            "strategy": "deterministic grouped folds by video_id",
            "n_splits": n_splits,
            "seed": seed,
            "folds": fold_summaries,
        },
        "models": {
            "pointwise_hybrid_ridge": {
                "objective": "squared error on four absolute ratings",
                "alpha": ridge_alpha,
                "ranking": pointwise["models"]["ridge"]["ranking"],
            },
            "pairwise_hybrid_logistic": {
                "objective": "Bradley-Terry logistic loss on within-video preferences",
                "l2": l2,
                "ranking": ranking_metrics(records, actual, pairwise_scores),
            },
        },
        "uncertainty": {
            "pairwise_vs_pointwise": paired_ranking_bootstrap(
                records,
                actual,
                pointwise_scores,
                pairwise_scores,
                iterations=bootstrap_iterations,
                seed=seed,
            )
        },
        "final_model": serializable_model,
        "predictions": [
            {
                "annotation_id": record["annotation_id"],
                "video_id": record["video_id"],
                "fold": prediction_folds[index],
                "actual_quality_score": actual[index],
                "pointwise_hybrid_ridge_score": pointwise_scores[index],
                "pairwise_hybrid_logistic_score": pairwise_scores[index],
            }
            for index, record in enumerate(records)
        ],
    }


def main() -> None:
    """Evaluate the pairwise ranker and write private model artifacts."""
    parser = argparse.ArgumentParser(description="Train CreatorCut's pairwise clip ranker")
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/training_clips.jsonl")
    )
    parser.add_argument(
        "--embeddings", type=Path, default=Path("data/processed/semantic_embeddings.json")
    )
    parser.add_argument(
        "--evaluation", type=Path, default=Path("data/processed/pairwise_evaluation.json")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("data/processed/pairwise_model.json")
    )
    parser.add_argument(
        "--predictions", type=Path, default=Path("data/processed/pairwise_predictions.jsonl")
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()

    evaluation = cross_validate_pairwise_ranker(
        load_jsonl(args.input),
        json.loads(args.embeddings.read_text(encoding="utf-8")),
        n_splits=args.folds,
        seed=args.seed,
        l2=args.l2,
        ridge_alpha=args.ridge_alpha,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    predictions = evaluation.pop("predictions")
    final_model = evaluation.pop("final_model")
    args.evaluation.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(final_model), encoding="utf-8")
    write_jsonl(predictions, args.predictions)

    print(json.dumps(evaluation["models"], indent=2))
    print(f"Wrote evaluation to {args.evaluation}")
    print(f"Wrote final model to {args.model}")
    print(f"Wrote out-of-fold predictions to {args.predictions}")


if __name__ == "__main__":
    main()
