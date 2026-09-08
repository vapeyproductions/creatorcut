import numpy as np
import pytest

from creatorcut.candidates import CORE_SCORE_FIELDS
from creatorcut.holdout import (
    build_prediction_commitment,
    canonical_sha256,
    evaluate_frozen_predictions,
    score_frozen_model,
)
from creatorcut.semantic import EMBEDDING_SCHEMA, build_embedding_artifact
from creatorcut.training import MODEL_FEATURE_FIELDS, QUEUE_FEATURE_SCHEMA


def holdout_queue() -> list[dict]:
    return [
        {
            "annotation_id": "video_021_sample_01",
            "candidate_id": "video_021_candidate_0001",
            "video_id": "video_021",
            "start_seconds": 0.0,
            "end_seconds": 20.0,
            "duration_seconds": 20.0,
            "transcript_text": "This first complete thought is useful.",
            "labels": {},
        },
        {
            "annotation_id": "video_021_sample_02",
            "candidate_id": "video_021_candidate_0002",
            "video_id": "video_021",
            "start_seconds": 30.0,
            "end_seconds": 55.0,
            "duration_seconds": 25.0,
            "transcript_text": "This second complete thought is even more useful.",
            "labels": {},
        },
    ]


def embedding_artifact(queue: list[dict]) -> dict:
    embeddings = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    return build_embedding_artifact(queue, embeddings)


def frozen_artifact(embedding_metadata: dict) -> dict:
    embedding_fields = tuple(
        f"embedding_{index:03d}" for index in range(embedding_metadata["dimension"])
    )
    fields = (*MODEL_FEATURE_FIELDS, *embedding_fields)
    coefficients = {
        target: {field: (1.0 if field == "embedding_000" else 0.0) for field in fields}
        for target in CORE_SCORE_FIELDS
    }
    return {
        "freeze_schema": "creatorcut_ranker_freeze_v1",
        "frozen_at": "2026-09-07T00:00:00+00:00",
        "status": "frozen_for_external_holdout_evaluation",
        "selected_model": "text_semantic_ridge",
        "development_dataset": {"sha256": "development-hash"},
        "evaluation_protocol": {
            "folds": [
                {
                    "train_videos": ["video_005"],
                    "test_videos": ["video_006"],
                }
            ]
        },
        "semantic_feature_metadata": embedding_metadata,
        "model": {
            "feature_schema": f"{QUEUE_FEATURE_SCHEMA}_plus_{EMBEDDING_SCHEMA}",
            "feature_fields": list(fields),
            "target_fields": list(CORE_SCORE_FIELDS),
            "feature_means": [0.0] * len(fields),
            "feature_scales": [1.0] * len(fields),
            "intercepts": {target: 1.0 for target in CORE_SCORE_FIELDS},
            "standardized_coefficients": coefficients,
            "alpha": 10.0,
        },
        "holdout_policy": {
            "new_video_ids_only": True,
            "fit_or_tune_on_holdout": False,
            "report_before_any_retraining": True,
        },
    }


def reviews() -> list[dict]:
    return [
        {
            "annotation_id": "video_021_sample_01",
            "candidate_id": "video_021_candidate_0001",
            "video_id": "video_021",
            "hook": 2,
            "completeness": 2,
            "payoff": 2,
            "clarity": 2,
            "technically_exportable": True,
            "start_seconds": 0.0,
            "end_seconds": 20.0,
        },
        {
            "annotation_id": "video_021_sample_02",
            "candidate_id": "video_021_candidate_0002",
            "video_id": "video_021",
            "hook": 4,
            "completeness": 4,
            "payoff": 4,
            "clarity": 4,
            "technically_exportable": True,
            "start_seconds": 30.0,
            "end_seconds": 55.0,
        },
    ]


def test_frozen_scoring_seals_unlabeled_predictions_and_commitment() -> None:
    queue = holdout_queue()
    embeddings = embedding_artifact(queue)
    frozen = frozen_artifact(embeddings["metadata"])

    predictions = score_frozen_model(queue, embeddings, frozen)
    commitment = build_prediction_commitment(predictions)

    assert predictions["metadata"]["sealed_before_human_review"] is True
    assert predictions["records"][1]["model_selected"] is True
    assert predictions["records"][1]["rank_within_video"] == 1
    assert commitment["canonical_json_sha256"] == canonical_sha256(predictions)
    assert commitment["human_labels_present_when_sealed"] is False


def test_frozen_scoring_rejects_development_video() -> None:
    queue = holdout_queue()
    queue[0]["video_id"] = "video_005"
    queue[1]["video_id"] = "video_005"
    embeddings = embedding_artifact(queue)
    frozen = frozen_artifact(embeddings["metadata"])

    with pytest.raises(ValueError, match="development video IDs"):
        score_frozen_model(queue, embeddings, frozen)


def test_external_evaluation_uses_sealed_predictions() -> None:
    queue = holdout_queue()
    embeddings = embedding_artifact(queue)
    predictions = score_frozen_model(
        queue,
        embeddings,
        frozen_artifact(embeddings["metadata"]),
    )

    evaluation = evaluate_frozen_predictions(predictions, reviews())

    assert evaluation["dataset"] == {
        "clips": 2,
        "videos": 1,
        "technically_exportable": 2,
        "technical_issue": 0,
    }
    assert evaluation["ranking"]["pairwise_accuracy"] == pytest.approx(1.0)
    assert evaluation["ranking"]["top_1_hit_rate"] == pytest.approx(1.0)
