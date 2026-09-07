import numpy as np
import pytest

from creatorcut.audio import (
    AUDIO_FEATURE_FIELDS,
    AUDIO_FEATURE_SCHEMA,
    audio_matrix_for_records,
    build_frozen_model_artifact,
    extract_audio_features,
    run_audio_ablation,
)
from creatorcut.semantic import build_embedding_artifact
from creatorcut.training import FEATURE_SCHEMA, MODEL_FEATURE_FIELDS


def training_records(video_count: int = 4, clips_per_video: int = 2) -> list[dict]:
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


def audio_artifact(records: list[dict]) -> dict:
    return {
        "metadata": {
            "audio_feature_schema": AUDIO_FEATURE_SCHEMA,
            "feature_fields": list(AUDIO_FEATURE_FIELDS),
        },
        "records": [
            {
                "annotation_id": record["annotation_id"],
                "video_id": record["video_id"],
                "features": {
                    field: float(index + record_index)
                    for index, field in enumerate(AUDIO_FEATURE_FIELDS)
                },
            }
            for record_index, record in enumerate(records)
        ],
    }


def test_extract_audio_features_returns_finite_declared_schema() -> None:
    sampling_rate = 16_000
    time = np.arange(sampling_rate, dtype=np.float32) / sampling_rate
    waveform = (0.25 * np.sin(2.0 * np.pi * 200.0 * time)).astype(np.float32)

    features = extract_audio_features(
        waveform,
        [{"start": 0, "end": sampling_rate}],
        sampling_rate=sampling_rate,
    )

    assert tuple(features) == AUDIO_FEATURE_FIELDS
    assert all(np.isfinite(value) for value in features.values())
    assert features["vad_speech_ratio"] == pytest.approx(1.0)
    assert features["pitch_median_hz"] == pytest.approx(200.0, rel=0.05)


def test_audio_artifact_joins_by_annotation_id() -> None:
    records = training_records(video_count=2)
    artifact = audio_artifact(records)
    artifact["records"].reverse()

    matrix = audio_matrix_for_records(records, artifact)

    assert matrix.shape == (4, len(AUDIO_FEATURE_FIELDS))
    assert matrix[0, 0] == pytest.approx(0.0)


def test_audio_artifact_rejects_identity_mismatch() -> None:
    records = training_records(video_count=2)
    artifact = audio_artifact(records)
    artifact["records"].pop()

    with pytest.raises(ValueError, match="Audio IDs do not match"):
        audio_matrix_for_records(records, artifact)


def test_audio_ablation_groups_videos_and_builds_freeze_contract() -> None:
    records = training_records()
    raw_embeddings = np.asarray(
        [[record["targets"]["quality_score"], 1.0] for record in records], dtype=float
    )
    normalized = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    embeddings = build_embedding_artifact(records, normalized)

    result = run_audio_ablation(
        records,
        embeddings,
        audio_artifact(records),
        n_splits=2,
        alpha=10.0,
        bootstrap_iterations=20,
    )
    frozen = build_frozen_model_artifact(result)

    assert set(result["models"]) == {
        "audio_ridge",
        "semantic_ridge",
        "text_semantic_ridge",
        "semantic_audio_ridge",
        "full_multimodal_ridge",
    }
    assert result["selected_model"] in {
        "text_semantic_ridge",
        "full_multimodal_ridge",
    }
    for fold in result["cross_validation"]["folds"]:
        assert set(fold["train_videos"]).isdisjoint(fold["test_videos"])
    assert frozen["status"] == "frozen_for_external_holdout_evaluation"
    assert frozen["development_dataset"]["sha256"] == result["development_dataset_sha256"]
    assert frozen["holdout_policy"] == {
        "new_video_ids_only": True,
        "fit_or_tune_on_holdout": False,
        "report_before_any_retraining": True,
    }
