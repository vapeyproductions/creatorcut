import numpy as np
import pytest

from creatorcut.boundary_learning import (
    BOUNDARY_DATA_SCHEMA,
    boundary_audio_features,
    build_boundary_query,
    cross_validate_boundary_ranker,
)


def transcript() -> dict:
    return {
        "media_duration_seconds": 90.0,
        "segments": [
            {
                "words": [
                    {"start": 8.0, "end": 8.4, "word": " Setup", "probability": 0.9},
                    {"start": 8.5, "end": 9.0, "word": " ends.", "probability": 0.9},
                    {"start": 10.0, "end": 10.3, "word": " Strong", "probability": 0.9},
                    {"start": 10.4, "end": 11.0, "word": " opening", "probability": 0.9},
                    {"start": 18.0, "end": 18.4, "word": " completes", "probability": 0.9},
                    {"start": 18.5, "end": 19.0, "word": " here.", "probability": 0.9},
                ]
            }
        ],
    }


def queued(annotation_id: str = "a1", video_id: str = "video_001") -> dict:
    return {
        "annotation_id": annotation_id,
        "video_id": video_id,
        "start_seconds": 10.0,
        "end_seconds": 18.0,
    }


def review(annotation_id: str = "a1", edit: bool = True) -> dict:
    return {
        "annotation_id": annotation_id,
        "boundary_edit": {"start_seconds": 8.0, "end_seconds": 19.0} if edit else None,
    }


def test_boundary_audio_features_detect_quiet_edit_point() -> None:
    audio = np.ones(32_000, dtype=np.float32) * 0.2
    audio[15_000:17_000] = 0.0

    features = boundary_audio_features(audio, 1.0, sample_rate=16_000)

    assert features["audio_low_energy_ratio"] > 0.0
    assert features["audio_min_frame_energy_db"] < features["audio_energy_before_db"]


def test_boundary_query_keeps_original_and_labels_edited_target() -> None:
    query = build_boundary_query(queued(), review(), transcript(), "start")

    assert query["target_time_seconds"] == pytest.approx(8.0)
    assert query["target_adjustment_seconds"] == pytest.approx(-2.0)
    assert query["materially_adjusted"] is True
    assert any(option["time_seconds"] == 10.0 for option in query["options"])
    assert min(option["target_distance_seconds"] for option in query["options"]) == 0.0


def test_grouped_boundary_ranker_produces_out_of_video_predictions() -> None:
    queries = []
    for video_index in range(4):
        for clip_index in range(2):
            annotation_id = f"a_{video_index}_{clip_index}"
            item = queued(annotation_id, f"video_{video_index}")
            item["start_seconds"] += clip_index
            item["end_seconds"] += clip_index
            item_review = review(annotation_id, edit=clip_index == 0)
            queries.extend(
                build_boundary_query(item, item_review, transcript(), kind)
                for kind in ("start", "end")
            )
    artifact = {
        "metadata": {"schema": BOUNDARY_DATA_SCHEMA},
        "queries": queries,
    }

    result = cross_validate_boundary_ranker(artifact, n_splits=2, seed=42, l2=0.3)

    assert result["dataset"]["videos"] == 4
    assert len(result["predictions"]) == 16
    for fold in result["cross_validation"]["folds"]:
        assert fold["test_videos"]
    assert result["metrics"]["start"]["candidate_oracle"]["mae_seconds"] == 0.0
