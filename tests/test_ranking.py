import numpy as np
import pytest

from creatorcut.ranking import (
    build_preference_matrix,
    cross_validate_pairwise_ranker,
    fit_pairwise_logistic,
    paired_ranking_bootstrap,
    predict_pairwise_scores,
)
from creatorcut.semantic import build_embedding_artifact
from creatorcut.training import MODEL_FEATURE_FIELDS


def ranked_records(video_count: int = 8, clips_per_video: int = 3) -> list[dict]:
    records = []
    for video_index in range(video_count):
        for clip_index in range(clips_per_video):
            signal = float(clip_index + 1)
            features = {field: 0.0 for field in MODEL_FEATURE_FIELDS}
            features["duration_seconds"] = 20.0 + signal
            score = 1.0 + signal
            records.append(
                {
                    "annotation_id": f"annotation_{video_index}_{clip_index}",
                    "video_id": f"video_{video_index:03d}",
                    "technically_exportable": True,
                    "feature_schema": "queue_transcript_v1",
                    "features": features,
                    "targets": {
                        "hook": score,
                        "completeness": score,
                        "payoff": score,
                        "clarity": score,
                        "quality_score": score,
                    },
                }
            )
    return records


def test_preference_matrix_uses_only_within_video_non_ties() -> None:
    records = [
        {"video_id": "video_a"},
        {"video_id": "video_a"},
        {"video_id": "video_a"},
        {"video_id": "video_b"},
    ]
    features = np.asarray([[1.0], [3.0], [5.0], [100.0]])
    quality = np.asarray([1.0, 2.0, 2.0, 5.0])

    differences = build_preference_matrix(records, features, quality, list(range(4)))

    assert differences.shape == (2, 1)
    assert differences[:, 0].tolist() == pytest.approx([2.0, 4.0])


def test_pairwise_logistic_learns_ordering_signal() -> None:
    records = ranked_records(video_count=4)
    features = np.asarray(
        [[record["features"]["duration_seconds"]] for record in records], dtype=float
    )
    quality = np.asarray([record["targets"]["quality_score"] for record in records])

    model = fit_pairwise_logistic(records, features, quality, l2=0.1)
    predictions = predict_pairwise_scores(features, model)

    assert model["training_preferences"] == 12
    assert model["converged"] is True
    for start in range(0, len(predictions), 3):
        assert predictions[start : start + 3].tolist() == sorted(
            predictions[start : start + 3].tolist()
        )


def test_pairwise_objective_ablation_uses_grouped_out_of_fold_predictions() -> None:
    records = ranked_records()
    raw_embeddings = np.asarray(
        [[record["targets"]["quality_score"], 1.0] for record in records], dtype=float
    )
    normalized = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    artifact = build_embedding_artifact(records, normalized)

    result = cross_validate_pairwise_ranker(
        records, artifact, n_splits=4, l2=0.1, bootstrap_iterations=100
    )

    pairwise = result["models"]["pairwise_hybrid_logistic"]["ranking"]
    assert pairwise["pairwise_accuracy"] == pytest.approx(1.0)
    assert pairwise["top_1_hit_rate"] == pytest.approx(1.0)
    assert len(result["predictions"]) == len(records)
    for fold in result["cross_validation"]["folds"]:
        assert set(fold["train_videos"]).isdisjoint(fold["test_videos"])


def test_paired_ranking_bootstrap_resamples_whole_videos() -> None:
    records = [{"video_id": f"video_{index // 3}"} for index in range(12)]
    actual = [1.0, 2.0, 3.0] * 4
    reference = [3.0, 2.0, 1.0] * 4
    challenger = actual.copy()

    result = paired_ranking_bootstrap(
        records, actual, reference, challenger, iterations=100, seed=42
    )

    comparisons = result["challenger_minus_reference"]
    assert comparisons["pairwise_accuracy_delta"]["estimate"] == pytest.approx(1.0)
    assert comparisons["top_1_hit_rate_delta"]["bootstrap_probability_improved"] == 1.0
