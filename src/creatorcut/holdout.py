"""Score and evaluate an external holdout without modifying the frozen ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from creatorcut.candidates import CORE_SCORE_FIELDS
from creatorcut.dataset import load_jsonl
from creatorcut.semantic import EMBEDDING_SCHEMA, embedding_matrix_for_records
from creatorcut.training import (
    MODEL_FEATURE_FIELDS,
    QUEUE_FEATURE_SCHEMA,
    queue_model_features,
    ranking_metrics,
    regression_metrics,
)

FREEZE_SCHEMA = "creatorcut_ranker_freeze_v1"
PREDICTION_SCHEMA = "creatorcut_frozen_predictions_v1"
COMMITMENT_SCHEMA = "creatorcut_prediction_commitment_v1"
EVALUATION_SCHEMA = "creatorcut_external_holdout_evaluation_v1"


def canonical_sha256(value: Any) -> str:
    """Hash JSON using a stable encoding independent of whitespace and key order."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _unique_index(
    records: list[dict[str, Any]], key: str, source: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        value = record.get(key)
        if not isinstance(value, str) or not value or value in indexed:
            raise ValueError(f"{source} requires unique non-empty {key} values")
        indexed[value] = record
    return indexed


def _validate_frozen_model(frozen: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if frozen.get("freeze_schema") != FREEZE_SCHEMA:
        raise ValueError(f"Expected freeze schema {FREEZE_SCHEMA}")
    if frozen.get("status") != "frozen_for_external_holdout_evaluation":
        raise ValueError("Model is not frozen for external holdout evaluation")
    policy = frozen.get("holdout_policy", {})
    if policy.get("new_video_ids_only") is not True:
        raise ValueError("Frozen model does not require new holdout videos")
    if policy.get("fit_or_tune_on_holdout") is not False:
        raise ValueError("Frozen model does not prohibit holdout tuning")

    model = frozen.get("model", {})
    semantic = frozen.get("semantic_feature_metadata", {})
    dimension = semantic.get("dimension")
    if semantic.get("embedding_schema") != EMBEDDING_SCHEMA:
        raise ValueError(f"Expected frozen semantic schema {EMBEDDING_SCHEMA}")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Frozen semantic dimension is invalid")
    embedding_fields = tuple(f"embedding_{index:03d}" for index in range(dimension))
    expected_fields = (*MODEL_FEATURE_FIELDS, *embedding_fields)
    if model.get("feature_fields") != list(expected_fields):
        raise ValueError("Frozen model feature fields are not the v1 text+semantic representation")
    expected_feature_schema = f"{QUEUE_FEATURE_SCHEMA}_plus_{EMBEDDING_SCHEMA}"
    if model.get("feature_schema") != expected_feature_schema:
        raise ValueError(f"Expected frozen feature schema {expected_feature_schema}")
    if model.get("target_fields") != list(CORE_SCORE_FIELDS):
        raise ValueError("Frozen target fields do not match the content scores")
    return model, dimension


def _validate_embedding_metadata(
    frozen: dict[str, Any], embedding_artifact: dict[str, Any]
) -> None:
    expected = frozen["semantic_feature_metadata"]
    actual = embedding_artifact.get("metadata", {})
    for field in (
        "embedding_schema",
        "model_id",
        "revision",
        "onnx_filename",
        "maximum_length",
        "dimension",
        "normalized",
        "pooling",
    ):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"Holdout embedding metadata differs on {field}")


def score_frozen_model(
    queue: list[dict[str, Any]],
    embedding_artifact: dict[str, Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    """Predict an unlabeled queue using exactly the serialized v1 parameters."""
    if not queue:
        raise ValueError("Holdout queue is empty")
    _unique_index(queue, "annotation_id", "Holdout queue")
    for record in queue:
        labels = record.get("labels", {})
        if not isinstance(labels, dict):
            raise ValueError(f"{record['annotation_id']}: labels placeholder must be an object")
        if any(value not in (None, "") for value in labels.values()):
            raise ValueError(f"{record['annotation_id']}: holdout queue already contains labels")
    model, dimension = _validate_frozen_model(frozen)
    _validate_embedding_metadata(frozen, embedding_artifact)
    embeddings = embedding_matrix_for_records(queue, embedding_artifact)
    if embeddings.shape[1] != dimension:
        raise ValueError("Holdout embedding dimension differs from frozen model")

    development_videos = set(
        frozen.get("evaluation_protocol", {})
        .get("folds", [{}])[0]
        .get("train_videos", [])
    )
    for fold in frozen.get("evaluation_protocol", {}).get("folds", []):
        development_videos.update(fold.get("test_videos", []))
    holdout_videos = {record["video_id"] for record in queue}
    overlap = sorted(development_videos & holdout_videos)
    if overlap:
        raise ValueError(f"Holdout contains development video IDs: {overlap}")

    handcrafted = np.asarray(
        [
            [queue_model_features(record)[field] for field in MODEL_FEATURE_FIELDS]
            for record in queue
        ],
        dtype=float,
    )
    feature_matrix = np.column_stack([handcrafted, embeddings])
    means = np.asarray(model.get("feature_means"), dtype=float)
    scales = np.asarray(model.get("feature_scales"), dtype=float)
    if means.shape != (feature_matrix.shape[1],) or scales.shape != means.shape:
        raise ValueError("Frozen feature scaling dimensions are invalid")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Frozen feature scaling values are invalid")
    standardized = (feature_matrix - means) / scales

    predicted_targets = np.empty((len(queue), len(CORE_SCORE_FIELDS)), dtype=float)
    coefficient_groups = model.get("standardized_coefficients", {})
    intercepts = model.get("intercepts", {})
    feature_fields = tuple(model["feature_fields"])
    for target_index, target in enumerate(CORE_SCORE_FIELDS):
        coefficients = coefficient_groups.get(target, {})
        if set(coefficients) != set(feature_fields):
            raise ValueError(f"Frozen coefficients are incomplete for {target}")
        intercept = intercepts.get(target)
        if not isinstance(intercept, int | float) or not math.isfinite(float(intercept)):
            raise ValueError(f"Frozen intercept is invalid for {target}")
        weights = np.asarray([coefficients[field] for field in feature_fields], dtype=float)
        if not np.isfinite(weights).all():
            raise ValueError(f"Frozen coefficients are non-finite for {target}")
        predicted_targets[:, target_index] = float(intercept) + standardized @ weights

    records: list[dict[str, Any]] = []
    for index, queued in enumerate(queue):
        targets = {
            target: float(predicted_targets[index, target_index])
            for target_index, target in enumerate(CORE_SCORE_FIELDS)
        }
        records.append(
            {
                "annotation_id": queued["annotation_id"],
                "candidate_id": queued["candidate_id"],
                "video_id": queued["video_id"],
                "start_seconds": float(queued["start_seconds"]),
                "end_seconds": float(queued["end_seconds"]),
                "predicted_targets": targets,
                "predicted_quality_score": float(np.mean(list(targets.values()))),
            }
        )
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_video[record["video_id"]].append(record)
    for video_records in by_video.values():
        ordered = sorted(
            video_records,
            key=lambda value: (-value["predicted_quality_score"], value["annotation_id"]),
        )
        for rank, record in enumerate(ordered, start=1):
            record["rank_within_video"] = rank
            record["model_selected"] = rank == 1

    return {
        "metadata": {
            "prediction_schema": PREDICTION_SCHEMA,
            "created_at": datetime.now(UTC).isoformat(),
            "sealed_before_human_review": True,
            "model_frozen_at": frozen["frozen_at"],
            "selected_model": frozen["selected_model"],
            "development_dataset_sha256": frozen["development_dataset"]["sha256"],
            "frozen_model_sha256": canonical_sha256(frozen),
            "queue_sha256": canonical_sha256(queue),
            "clips": len(records),
            "videos": len(by_video),
        },
        "records": records,
    }


def build_prediction_commitment(prediction_artifact: dict[str, Any]) -> dict[str, Any]:
    """Create a public, non-revealing hash commitment to sealed predictions."""
    metadata = prediction_artifact.get("metadata", {})
    records = prediction_artifact.get("records", [])
    if metadata.get("prediction_schema") != PREDICTION_SCHEMA or not records:
        raise ValueError("Cannot commit an invalid prediction artifact")
    return {
        "commitment_schema": COMMITMENT_SCHEMA,
        "created_at": metadata["created_at"],
        "canonical_json_sha256": canonical_sha256(prediction_artifact),
        "frozen_model_sha256": metadata["frozen_model_sha256"],
        "development_dataset_sha256": metadata["development_dataset_sha256"],
        "queue_sha256": metadata["queue_sha256"],
        "clips": metadata["clips"],
        "videos": metadata["videos"],
        "predictions_public": False,
        "human_labels_present_when_sealed": False,
    }


def evaluate_frozen_predictions(
    prediction_artifact: dict[str, Any], reviews: list[dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate sealed predictions after all corresponding blind reviews exist."""
    metadata = prediction_artifact.get("metadata", {})
    if metadata.get("prediction_schema") != PREDICTION_SCHEMA:
        raise ValueError(f"Expected prediction schema {PREDICTION_SCHEMA}")
    predictions = prediction_artifact.get("records", [])
    prediction_by_id = _unique_index(predictions, "annotation_id", "Predictions")
    review_by_id = _unique_index(reviews, "annotation_id", "Reviews")
    if set(prediction_by_id) != set(review_by_id):
        missing = sorted(prediction_by_id.keys() - review_by_id.keys())
        extra = sorted(review_by_id.keys() - prediction_by_id.keys())
        raise ValueError(f"Review IDs do not match predictions; missing={missing}, extra={extra}")

    aligned_records: list[dict[str, Any]] = []
    actual_by_target: dict[str, list[float]] = {field: [] for field in CORE_SCORE_FIELDS}
    predicted_by_target: dict[str, list[float]] = {field: [] for field in CORE_SCORE_FIELDS}
    actual_quality: list[float] = []
    predicted_quality: list[float] = []
    for prediction in predictions:
        review = review_by_id[prediction["annotation_id"]]
        for identity in ("candidate_id", "video_id"):
            if review.get(identity) != prediction.get(identity):
                raise ValueError(f"{prediction['annotation_id']}: {identity} does not match")
        for boundary in ("start_seconds", "end_seconds"):
            value = review.get(boundary)
            if not isinstance(value, int | float) or not math.isclose(
                float(value), float(prediction[boundary]), abs_tol=1e-6
            ):
                raise ValueError(f"{prediction['annotation_id']}: {boundary} does not match")
        actual_targets: dict[str, float] = {}
        for target in CORE_SCORE_FIELDS:
            value = review.get(target)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"{prediction['annotation_id']}: invalid {target} review")
            actual_targets[target] = float(value)
            actual_by_target[target].append(float(value))
            predicted_by_target[target].append(float(prediction["predicted_targets"][target]))
        actual_score = float(np.mean(list(actual_targets.values())))
        actual_quality.append(actual_score)
        predicted_quality.append(float(prediction["predicted_quality_score"]))
        technically_exportable = review.get("technically_exportable")
        if not isinstance(technically_exportable, bool):
            raise ValueError(
                f"{prediction['annotation_id']}: technically_exportable must be boolean"
            )
        aligned_records.append(
            {
                "annotation_id": prediction["annotation_id"],
                "candidate_id": prediction["candidate_id"],
                "video_id": prediction["video_id"],
                "actual_targets": actual_targets,
                "actual_quality_score": actual_score,
                "predicted_targets": prediction["predicted_targets"],
                "predicted_quality_score": prediction["predicted_quality_score"],
                "rank_within_video": prediction["rank_within_video"],
                "model_selected": prediction["model_selected"],
                "technically_exportable": technically_exportable,
            }
        )

    regression = {
        target: regression_metrics(actual_by_target[target], predicted_by_target[target])
        for target in CORE_SCORE_FIELDS
    }
    regression["quality_score"] = regression_metrics(actual_quality, predicted_quality)
    return {
        "evaluation_schema": EVALUATION_SCHEMA,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "prediction_commitment_sha256": canonical_sha256(prediction_artifact),
        "frozen_model_sha256": metadata["frozen_model_sha256"],
        "development_dataset_sha256": metadata["development_dataset_sha256"],
        "dataset": {
            "clips": len(aligned_records),
            "videos": len({record["video_id"] for record in aligned_records}),
            "technically_exportable": sum(
                record["technically_exportable"] is True for record in aligned_records
            ),
            "technical_issue": sum(
                record["technically_exportable"] is False for record in aligned_records
            ),
        },
        "regression": regression,
        "ranking": ranking_metrics(aligned_records, actual_quality, predicted_quality),
        "records": aligned_records,
    }


def score_main() -> None:
    parser = argparse.ArgumentParser(description="Seal frozen v1 predictions for a holdout queue")
    parser.add_argument(
        "--queue", type=Path, default=Path("data/processed/holdout_v1/annotation_queue.jsonl")
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/processed/holdout_v1/semantic_embeddings.json"),
    )
    parser.add_argument(
        "--frozen-model", type=Path, default=Path("data/processed/frozen_model_v1.json")
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/processed/holdout_v1/frozen_predictions.json"),
    )
    parser.add_argument(
        "--commitment",
        type=Path,
        default=Path("data/holdout_v1/prediction_commitment.json"),
    )
    args = parser.parse_args()

    queue = load_jsonl(args.queue)
    embeddings = json.loads(args.embeddings.read_text(encoding="utf-8"))
    frozen = json.loads(args.frozen_model.read_text(encoding="utf-8"))
    predictions = score_frozen_model(queue, embeddings, frozen)
    commitment = build_prediction_commitment(predictions)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    args.commitment.parent.mkdir(parents=True, exist_ok=True)
    args.commitment.write_text(json.dumps(commitment, indent=2), encoding="utf-8")
    print(
        f"Sealed {commitment['clips']} predictions across {commitment['videos']} videos; "
        f"commitment {commitment['canonical_json_sha256']}"
    )
    print(f"Private predictions: {args.predictions}")
    print(f"Public commitment: {args.commitment}")


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sealed v1 holdout predictions")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/processed/holdout_v1/frozen_predictions.json"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/processed/holdout_v1/annotation_reviews.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/holdout_v1/evaluation.json"),
    )
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    evaluation = evaluate_frozen_predictions(predictions, load_jsonl(args.reviews))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    summary = {
        "regression": evaluation["regression"],
        "ranking": evaluation["ranking"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote external holdout evaluation to {args.output}")
