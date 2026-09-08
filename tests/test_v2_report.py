import pytest

from creatorcut.v2_report import build_v2_summary


def record(annotation_id: str, video_id: str, quality: float) -> dict:
    return {
        "annotation_id": annotation_id,
        "video_id": video_id,
        "targets": {"quality_score": quality},
    }


def prediction(annotation_id: str, video_id: str, pointwise: float, pairwise: float) -> dict:
    return {
        "annotation_id": annotation_id,
        "video_id": video_id,
        "pointwise_hybrid_ridge_score": pointwise,
        "pairwise_hybrid_logistic_score": pairwise,
    }


def test_v2_report_builds_equal_rank_ensemble_and_cohorts() -> None:
    records = [
        record("old_low", "old_video", 1.0),
        record("old_high", "old_video", 5.0),
        record("new_low", "new_video", 1.0),
        record("new_high", "new_video", 5.0),
    ]
    predictions = [
        prediction("old_low", "old_video", 1.0, 1.0),
        prediction("old_high", "old_video", 2.0, 2.0),
        prediction("new_low", "new_video", 1.0, 1.0),
        prediction("new_high", "new_video", 2.0, 2.0),
    ]
    boundary = {
        "cross_validation": {"strategy": "grouped by video"},
        "metrics": {"start": {}, "end": {}},
    }

    result = build_v2_summary(records, predictions, {"new_low", "new_high"}, boundary)

    ensemble = result["moment_ranking"]["all"]["equal_rank_ensemble"]
    assert ensemble["pairwise_accuracy"] == 1.0
    assert ensemble["top_1_hit_rate"] == 1.0
    assert result["moment_ranking"]["original_cohort"]["videos"] == 1
    assert result["moment_ranking"]["promoted_cohort"]["clips"] == 2


def test_v2_report_rejects_prediction_identity_mismatch() -> None:
    boundary = {
        "cross_validation": {"strategy": "grouped by video"},
        "metrics": {},
    }
    with pytest.raises(ValueError, match="identities"):
        build_v2_summary(
            [record("old", "old_video", 1.0), record("new", "new_video", 2.0)],
            [prediction("old", "old_video", 1.0, 1.0)],
            {"new"},
            boundary,
        )
