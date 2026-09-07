"""Build supervised clip data and evaluate the first learned ranking baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from creatorcut.baseline import extract_transcript_features, spearman_correlation, tokenize
from creatorcut.candidates import CORE_SCORE_FIELDS, write_jsonl
from creatorcut.dataset import load_jsonl

FEATURE_SCHEMA = "handcrafted_transcript_v1"
QUEUE_FEATURE_SCHEMA = "queue_transcript_v1"
MODEL_FEATURE_FIELDS = (
    "duration_seconds",
    "word_count",
    "sentence_count",
    "clean_start",
    "complete_end",
    "hook_signal",
    "duration_fit",
    "words_per_second",
    "density_fit",
    "structure_fit",
    "lexical_diversity",
    "lexical_fit",
    "filler_ratio",
    "filler_control",
    "intro_outro",
)

SENTENCE_BOUNDARY_PATTERN = re.compile(r"[.!?][\"'”’)]*(?:\s|$)")


def _unique_index(
    records: list[dict[str, Any]], key: str, source_name: str
) -> dict[str, dict[str, Any]]:
    """Index records by a required unique string identifier."""
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{source_name} contains a record without a valid {key}")
        if value in indexed:
            raise ValueError(f"{source_name} contains duplicate {key}: {value}")
        indexed[value] = record
    return indexed


def _matching_value(
    annotation_id: str,
    field: str,
    first: dict[str, Any],
    second: dict[str, Any],
) -> None:
    """Require identifying queue and review values to agree."""
    first_value = first.get(field)
    second_value = second.get(field)
    if isinstance(first_value, int | float) and isinstance(second_value, int | float):
        matches = math.isclose(float(first_value), float(second_value), abs_tol=1e-6)
    else:
        matches = first_value == second_value
    if not matches:
        raise ValueError(
            f"{annotation_id}: queue and review disagree on {field}: "
            f"{first_value!r} != {second_value!r}"
        )


def candidate_model_features(candidate: dict[str, Any]) -> dict[str, float]:
    """Return the versioned, numeric feature vector used by the ridge model."""
    transcript_features = extract_transcript_features(candidate)
    return {
        "duration_seconds": float(candidate["duration_seconds"]),
        "word_count": float(candidate["word_count"]),
        "sentence_count": float(candidate["sentence_count"]),
        **transcript_features,
    }


def queue_model_features(queued: dict[str, Any]) -> dict[str, float]:
    """Reconstruct text features when a reviewed queue is the canonical local artifact."""
    text = queued.get("transcript_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{queued.get('annotation_id')}: transcript_text must be non-empty")
    tokens = tokenize(text)
    sentence_count = max(1, len(SENTENCE_BOUNDARY_PATTERN.findall(text)))
    candidate = {
        "text": text,
        "duration_seconds": float(queued["duration_seconds"]),
        "word_count": len(tokens),
        "sentence_count": sentence_count,
    }
    return candidate_model_features(candidate)


def _validated_targets(annotation_id: str, review: dict[str, Any]) -> dict[str, float]:
    targets: dict[str, float] = {}
    for field in CORE_SCORE_FIELDS:
        value = review.get(field)
        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{annotation_id}: {field} must be an integer from 1 to 5")
        targets[field] = float(value)
    targets["quality_score"] = sum(targets.values()) / len(CORE_SCORE_FIELDS)
    return targets


def _validated_exportability(annotation_id: str, review: dict[str, Any]) -> bool:
    technically_exportable = review.get("technically_exportable")
    if not isinstance(technically_exportable, bool):
        raise ValueError(f"{annotation_id}: technically_exportable must be boolean")
    return technically_exportable


def build_reviewed_training_records(
    queue: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join blind reviews to candidates without leaking sampler proxy scores."""
    queue_by_annotation = _unique_index(queue, "annotation_id", "annotation queue")
    candidates_by_id = _unique_index(candidates, "candidate_id", "candidate data")
    _unique_index(reviews, "annotation_id", "annotation reviews")

    output: list[dict[str, Any]] = []
    for review in reviews:
        annotation_id = review["annotation_id"]
        if annotation_id not in queue_by_annotation:
            raise ValueError(f"{annotation_id}: review is not present in the annotation queue")
        queued = queue_by_annotation[annotation_id]
        for field in (
            "candidate_id",
            "video_id",
            "start_seconds",
            "end_seconds",
            "duration_seconds",
        ):
            _matching_value(annotation_id, field, queued, review)

        candidate_id = queued["candidate_id"]
        if candidate_id not in candidates_by_id:
            raise ValueError(f"{annotation_id}: unknown candidate_id {candidate_id}")
        candidate = candidates_by_id[candidate_id]
        for field in ("video_id", "start_seconds", "end_seconds", "duration_seconds"):
            _matching_value(annotation_id, field, queued, candidate)
        if queued["transcript_text"] != candidate["text"]:
            raise ValueError(f"{annotation_id}: queued and candidate transcript text differ")

        targets = _validated_targets(annotation_id, review)
        technically_exportable = _validated_exportability(annotation_id, review)

        features = candidate_model_features(candidate)
        missing_features = set(MODEL_FEATURE_FIELDS) - features.keys()
        if missing_features:
            raise ValueError(f"{annotation_id}: missing model features {sorted(missing_features)}")
        output.append(
            {
                "annotation_id": annotation_id,
                "candidate_id": candidate_id,
                "video_id": review["video_id"],
                "start_seconds": float(review["start_seconds"]),
                "end_seconds": float(review["end_seconds"]),
                "duration_seconds": float(review["duration_seconds"]),
                "transcript_text": candidate["text"],
                "transcript_word_count": len(tokenize(candidate["text"])),
                "technically_exportable": technically_exportable,
                "feature_schema": FEATURE_SCHEMA,
                "features": {field: features[field] for field in MODEL_FEATURE_FIELDS},
                "targets": targets,
            }
        )
    return output


def build_reviewed_training_records_from_queue(
    queue: list[dict[str, Any]], reviews: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build model rows from an immutable review queue without sampler-score leakage."""
    queue_by_annotation = _unique_index(queue, "annotation_id", "annotation queue")
    _unique_index(reviews, "annotation_id", "annotation reviews")

    output: list[dict[str, Any]] = []
    for review in reviews:
        annotation_id = review["annotation_id"]
        if annotation_id not in queue_by_annotation:
            raise ValueError(f"{annotation_id}: review is not present in the annotation queue")
        queued = queue_by_annotation[annotation_id]
        for field in (
            "candidate_id",
            "video_id",
            "start_seconds",
            "end_seconds",
            "duration_seconds",
        ):
            _matching_value(annotation_id, field, queued, review)

        features = queue_model_features(queued)
        targets = _validated_targets(annotation_id, review)
        technically_exportable = _validated_exportability(annotation_id, review)
        output.append(
            {
                "annotation_id": annotation_id,
                "candidate_id": queued["candidate_id"],
                "video_id": review["video_id"],
                "start_seconds": float(review["start_seconds"]),
                "end_seconds": float(review["end_seconds"]),
                "duration_seconds": float(review["duration_seconds"]),
                "transcript_text": queued["transcript_text"],
                "transcript_word_count": len(tokenize(queued["transcript_text"])),
                "technically_exportable": technically_exportable,
                "feature_schema": QUEUE_FEATURE_SCHEMA,
                "features": {field: features[field] for field in MODEL_FEATURE_FIELDS},
                "targets": targets,
            }
        )
    return output


def build_group_folds(
    records: list[dict[str, Any]], n_splits: int = 4, seed: int = 42
) -> list[dict[str, Any]]:
    """Create deterministic folds in which a video occurs only in train or test."""
    video_ids = sorted({record["video_id"] for record in records})
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > len(video_ids):
        raise ValueError("n_splits cannot exceed the number of videos")

    random.Random(seed).shuffle(video_ids)
    fold_videos = [video_ids[index::n_splits] for index in range(n_splits)]
    folds: list[dict[str, Any]] = []
    for fold_index, test_videos in enumerate(fold_videos, start=1):
        test_video_set = set(test_videos)
        test_indices = [
            index for index, record in enumerate(records) if record["video_id"] in test_video_set
        ]
        train_indices = [
            index
            for index, record in enumerate(records)
            if record["video_id"] not in test_video_set
        ]
        folds.append(
            {
                "fold": fold_index,
                "train_indices": train_indices,
                "test_indices": test_indices,
                "train_videos": sorted(set(video_ids) - test_video_set),
                "test_videos": sorted(test_videos),
            }
        )
    return folds


def _fit_ridge(
    features: np.ndarray, targets: np.ndarray, alpha: float
) -> dict[str, np.ndarray]:
    """Fit standardized multi-output ridge regression in closed form."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = (features - means) / scales
    target_means = targets.mean(axis=0)
    centered_targets = targets - target_means
    feature_count = standardized.shape[1]
    if alpha == 0:
        weights = np.linalg.pinv(standardized) @ centered_targets
    elif feature_count <= len(standardized):
        weights = np.linalg.solve(
            standardized.T @ standardized + np.eye(feature_count) * alpha,
            standardized.T @ centered_targets,
        )
    else:
        weights = standardized.T @ np.linalg.solve(
            standardized @ standardized.T + np.eye(len(standardized)) * alpha,
            centered_targets,
        )
    coefficients = np.vstack([target_means, weights])
    return {"feature_means": means, "feature_scales": scales, "coefficients": coefficients}


def _predict_ridge(features: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    """Predict bounded 1–5 human scores from a fitted ridge model."""
    standardized = (features - model["feature_means"]) / model["feature_scales"]
    design = np.column_stack([np.ones(len(standardized)), standardized])
    return np.clip(design @ model["coefficients"], 1.0, 5.0)


def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float | None]:
    """Calculate error and rank-alignment metrics for continuous labels."""
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Expected two non-empty vectors with equal length")
    residuals = [truth - estimate for truth, estimate in zip(actual, predicted, strict=True)]
    return {
        "mae": sum(abs(value) for value in residuals) / len(residuals),
        "rmse": math.sqrt(sum(value**2 for value in residuals) / len(residuals)),
        "spearman": spearman_correlation(actual, predicted),
    }


def ranking_metrics(
    records: list[dict[str, Any]], actual: list[float], predicted: list[float]
) -> dict[str, float | int]:
    """Evaluate within-video ordering and top-clip selection."""
    by_video: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_video[record["video_id"]].append(index)

    pairwise_credit = 0.0
    comparable_pairs = 0
    top_hits = 0
    top_regrets: list[float] = []
    for indices in by_video.values():
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
        top_regrets.append(best_actual - selected_actual)

    return {
        "pairwise_accuracy": pairwise_credit / comparable_pairs if comparable_pairs else 0.0,
        "comparable_pairs": comparable_pairs,
        "top_1_hit_rate": top_hits / len(by_video),
        "mean_top_1_regret": sum(top_regrets) / len(top_regrets),
    }


def paired_video_bootstrap(
    records: list[dict[str, Any]],
    actual: list[float],
    reference: list[float],
    challenger: list[float],
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate paired metric differences by resampling whole videos."""
    vector_lengths = {len(records), len(actual), len(reference), len(challenger)}
    if len(vector_lengths) != 1 or not records:
        raise ValueError("Expected equal-length, non-empty bootstrap inputs")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    by_video: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_video[record["video_id"]].append(index)
    groups = list(by_video.values())

    def metrics(
        index_groups: list[list[int]], predicted: list[float]
    ) -> tuple[float, float, float, float]:
        pairwise_credit = 0.0
        comparable_pairs = 0
        top_hits = 0
        regrets: list[float] = []
        absolute_errors: list[float] = []
        for indices in index_groups:
            absolute_errors.extend(abs(actual[index] - predicted[index]) for index in indices)
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
            sum(absolute_errors) / len(absolute_errors),
            pairwise_credit / comparable_pairs if comparable_pairs else 0.0,
            top_hits / len(index_groups),
            sum(regrets) / len(regrets),
        )

    reference_observed = metrics(groups, reference)
    challenger_observed = metrics(groups, challenger)
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
        reference_sample = metrics(sampled_groups, reference)
        challenger_sample = metrics(sampled_groups, challenger)
        for index, (reference_value, challenger_value) in enumerate(
            zip(reference_sample, challenger_sample, strict=True)
        ):
            sampled_deltas[index].append(challenger_value - reference_value)

    names = ("quality_mae", "pairwise_accuracy", "top_1_hit_rate", "mean_top_1_regret")
    lower_is_better = {"quality_mae", "mean_top_1_regret"}
    comparisons: dict[str, Any] = {}
    for name, estimate, values in zip(names, observed, sampled_deltas, strict=True):
        values_array = np.asarray(values)
        improved = values_array < 0 if name in lower_is_better else values_array > 0
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


def _model_to_json(
    model: dict[str, np.ndarray],
    feature_fields: tuple[str, ...] = MODEL_FEATURE_FIELDS,
    feature_schema: str = FEATURE_SCHEMA,
) -> dict[str, Any]:
    """Serialize the final standardized ridge model without a pickle dependency."""
    coefficients = model["coefficients"]
    return {
        "feature_schema": feature_schema,
        "feature_fields": list(feature_fields),
        "target_fields": list(CORE_SCORE_FIELDS),
        "feature_means": model["feature_means"].tolist(),
        "feature_scales": model["feature_scales"].tolist(),
        "intercepts": {
            target: float(coefficients[0, index])
            for index, target in enumerate(CORE_SCORE_FIELDS)
        },
        "standardized_coefficients": {
            target: {
                feature: float(coefficients[feature_index + 1, target_index])
                for feature_index, feature in enumerate(feature_fields)
            }
            for target_index, target in enumerate(CORE_SCORE_FIELDS)
        },
    }


def cross_validate_ridge(
    records: list[dict[str, Any]],
    n_splits: int = 4,
    seed: int = 42,
    alpha: float = 10.0,
    feature_fields: tuple[str, ...] = MODEL_FEATURE_FIELDS,
    feature_schema: str = FEATURE_SCHEMA,
    experiment: str = "grouped_ridge_handcrafted_v1",
) -> dict[str, Any]:
    """Generate out-of-fold predictions and a final deployable ridge model."""
    if not records:
        raise ValueError("Training data is empty")
    schemas = {record.get("feature_schema") for record in records}
    if schemas != {feature_schema}:
        raise ValueError(f"Expected only feature schema {feature_schema}, found {schemas}")
    if not feature_fields:
        raise ValueError("At least one feature field is required")

    features = np.asarray(
        [[record["features"][field] for field in feature_fields] for record in records],
        dtype=float,
    )
    targets = np.asarray(
        [[record["targets"][field] for field in CORE_SCORE_FIELDS] for record in records],
        dtype=float,
    )
    ridge_predictions = np.zeros_like(targets)
    mean_predictions = np.zeros_like(targets)
    prediction_folds = [0] * len(records)
    folds = build_group_folds(records, n_splits=n_splits, seed=seed)

    fold_summaries: list[dict[str, Any]] = []
    for fold in folds:
        train = np.asarray(fold["train_indices"], dtype=int)
        test = np.asarray(fold["test_indices"], dtype=int)
        model = _fit_ridge(features[train], targets[train], alpha=alpha)
        ridge_predictions[test] = _predict_ridge(features[test], model)
        mean_predictions[test] = targets[train].mean(axis=0)
        for index in test:
            prediction_folds[int(index)] = fold["fold"]
        fold_summaries.append(
            {
                "fold": fold["fold"],
                "train_clip_count": len(train),
                "test_clip_count": len(test),
                "train_videos": fold["train_videos"],
                "test_videos": fold["test_videos"],
            }
        )

    def evaluate(predictions: np.ndarray) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for target_index, target in enumerate(CORE_SCORE_FIELDS):
            metrics[target] = regression_metrics(
                targets[:, target_index].tolist(), predictions[:, target_index].tolist()
            )
        actual_quality = targets.mean(axis=1).tolist()
        predicted_quality = predictions.mean(axis=1).tolist()
        metrics["quality_score"] = regression_metrics(actual_quality, predicted_quality)
        return {
            "regression": metrics,
            "ranking": ranking_metrics(records, actual_quality, predicted_quality),
        }

    final_model = _fit_ridge(features, targets, alpha=alpha)
    final_model_json = _model_to_json(
        final_model, feature_fields=feature_fields, feature_schema=feature_schema
    )
    final_model_json["alpha"] = alpha
    predictions = []
    for index, record in enumerate(records):
        predictions.append(
            {
                "annotation_id": record["annotation_id"],
                "video_id": record["video_id"],
                "fold": prediction_folds[index],
                "actual": {
                    target: float(targets[index, target_index])
                    for target_index, target in enumerate(CORE_SCORE_FIELDS)
                },
                "ridge_prediction": {
                    target: float(ridge_predictions[index, target_index])
                    for target_index, target in enumerate(CORE_SCORE_FIELDS)
                },
                "mean_prediction": {
                    target: float(mean_predictions[index, target_index])
                    for target_index, target in enumerate(CORE_SCORE_FIELDS)
                },
                "actual_quality_score": float(targets[index].mean()),
                "ridge_quality_score": float(ridge_predictions[index].mean()),
                "mean_quality_score": float(mean_predictions[index].mean()),
            }
        )

    return {
        "experiment": experiment,
        "dataset": {
            "clips": len(records),
            "videos": len({record["video_id"] for record in records}),
            "technically_exportable": sum(
                bool(record["technically_exportable"]) for record in records
            ),
            "technical_issue": sum(
                not bool(record["technically_exportable"]) for record in records
            ),
        },
        "features": {
            "schema": feature_schema,
            "count": len(feature_fields),
            "fields": list(feature_fields),
            "sampler_proxy_included": False,
        },
        "targets": {
            "fields": list(CORE_SCORE_FIELDS),
            "quality_score": "unweighted mean of the four content scores",
            "technically_exportable": "retained as a separate gate; not a regression target",
        },
        "cross_validation": {
            "strategy": "deterministic grouped folds by video_id",
            "n_splits": n_splits,
            "seed": seed,
            "folds": fold_summaries,
        },
        "models": {
            "fold_train_mean": evaluate(mean_predictions),
            "ridge": {"alpha": alpha, **evaluate(ridge_predictions)},
        },
        "final_model": final_model_json,
        "predictions": predictions,
    }


def build_main() -> None:
    """Build private model-ready rows from the completed blind review queue."""
    parser = argparse.ArgumentParser(description="Build supervised CreatorCut training rows")
    parser.add_argument(
        "--queue", type=Path, default=Path("data/processed/annotation_queue.jsonl")
    )
    parser.add_argument(
        "--reviews", type=Path, default=Path("data/processed/annotation_reviews.jsonl")
    )
    parser.add_argument(
        "--candidates", type=Path, default=Path("data/processed/candidates.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/training_clips.jsonl")
    )
    parser.add_argument(
        "--from-queue",
        action="store_true",
        help=(
            "Reconstruct transcript features from the immutable queue instead of loading the "
            "full candidate cache"
        ),
    )
    args = parser.parse_args()

    if args.from_queue:
        records = build_reviewed_training_records_from_queue(
            load_jsonl(args.queue), load_jsonl(args.reviews)
        )
    else:
        records = build_reviewed_training_records(
            load_jsonl(args.queue), load_jsonl(args.reviews), load_jsonl(args.candidates)
        )
    write_jsonl(records, args.output)
    print(
        f"Wrote {len(records)} reviewed clips from "
        f"{len({record['video_id'] for record in records})} videos to {args.output}"
    )


def train_main() -> None:
    """Run grouped cross-validation and save the evaluation and final ridge model."""
    parser = argparse.ArgumentParser(description="Train CreatorCut's first learned ranker")
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/training_clips.jsonl")
    )
    parser.add_argument(
        "--evaluation", type=Path, default=Path("data/processed/ridge_evaluation.json")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("data/processed/ridge_model.json")
    )
    parser.add_argument(
        "--predictions", type=Path, default=Path("data/processed/ridge_predictions.jsonl")
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=10.0)
    args = parser.parse_args()

    evaluation = cross_validate_ridge(
        load_jsonl(args.input), n_splits=args.folds, seed=args.seed, alpha=args.alpha
    )
    predictions = evaluation.pop("predictions")
    final_model = evaluation.pop("final_model")
    args.evaluation.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(final_model, indent=2), encoding="utf-8")
    write_jsonl(predictions, args.predictions)

    summary = {
        name: {
            "quality_score": result["regression"]["quality_score"],
            "ranking": result["ranking"],
        }
        for name, result in evaluation["models"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote evaluation to {args.evaluation}")
    print(f"Wrote final model to {args.model}")
    print(f"Wrote out-of-fold predictions to {args.predictions}")


if __name__ == "__main__":
    train_main()
