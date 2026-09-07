import numpy as np
import pytest

from creatorcut.semantic import (
    EMBEDDING_SCHEMA,
    build_embedding_artifact,
    embedding_matrix_for_records,
    mean_pool_and_normalize,
    run_representation_ablation,
)
from creatorcut.training import FEATURE_SCHEMA, MODEL_FEATURE_FIELDS


def test_mean_pool_ignores_padding_and_normalizes() -> None:
    token_embeddings = np.asarray([[[3.0, 0.0], [0.0, 4.0], [100.0, 100.0]]])
    attention_mask = np.asarray([[1, 1, 0]])

    pooled = mean_pool_and_normalize(token_embeddings, attention_mask)

    assert pooled.shape == (1, 2)
    assert pooled[0].tolist() == pytest.approx([0.6, 0.8])


def training_records(video_count: int = 8, clips_per_video: int = 3) -> list[dict]:
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
                    "feature_schema": FEATURE_SCHEMA,
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


def test_embedding_artifact_joins_by_annotation_id() -> None:
    records = training_records(video_count=2, clips_per_video=2)
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=float
    )
    artifact = build_embedding_artifact(records, embeddings)
    artifact["records"].reverse()

    joined = embedding_matrix_for_records(records, artifact)

    assert artifact["metadata"]["embedding_schema"] == EMBEDDING_SCHEMA
    assert np.allclose(joined, embeddings)


def test_embedding_artifact_rejects_unnormalized_vectors() -> None:
    records = training_records(video_count=2, clips_per_video=1)

    with pytest.raises(ValueError, match="L2-normalized"):
        build_embedding_artifact(records, np.asarray([[2.0, 0.0], [0.0, 1.0]]))


def test_representation_ablation_uses_identical_grouped_folds() -> None:
    records = training_records()
    raw_embeddings = np.asarray(
        [[record["targets"]["quality_score"], 1.0] for record in records], dtype=float
    )
    normalized = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    artifact = build_embedding_artifact(records, normalized)

    result = run_representation_ablation(records, artifact, n_splits=4, alpha=1.0)

    assert set(result["models"]) == {
        "fold_train_mean",
        "handcrafted_ridge",
        "semantic_ridge",
        "hybrid_ridge",
    }
    assert result["models"]["semantic_ridge"]["ranking"]["pairwise_accuracy"] == pytest.approx(
        1.0
    )
    assert len(result["predictions"]) == len(records)
