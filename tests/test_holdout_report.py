import pytest

from creatorcut.holdout_report import build_public_holdout_summary


def review(annotation_id, video_id, score, boundary_edit=None):
    return {
        "annotation_id": annotation_id,
        "video_id": video_id,
        "hook": score,
        "completeness": score,
        "payoff": score,
        "clarity": score,
        "boundary_edit": boundary_edit,
    }


def evaluation_record(annotation_id, video_id, actual, predicted):
    return {
        "annotation_id": annotation_id,
        "video_id": video_id,
        "actual_quality_score": actual,
        "predicted_quality_score": predicted,
        "predicted_targets": {
            "hook": predicted,
            "completeness": predicted,
            "payoff": predicted,
            "clarity": predicted,
        },
    }


def test_public_summary_aggregates_without_clip_level_records():
    records = [
        evaluation_record("a", "video_a", 5.0, 4.0),
        evaluation_record("b", "video_a", 2.0, 3.0),
        evaluation_record("c", "video_b", 1.0, 2.0),
        evaluation_record("d", "video_b", 4.0, 5.0),
    ]
    edit = {
        "scores": {"hook": 5, "completeness": 5, "payoff": 5, "clarity": 5},
        "start_adjustment_seconds": -2.0,
        "end_adjustment_seconds": 1.0,
    }
    reviews = [
        review("a", "video_a", 5),
        review("b", "video_a", 2),
        review("c", "video_b", 1, edit),
        review("d", "video_b", 4),
    ]
    evaluation = {
        "evaluation_schema": "test",
        "evaluated_at": "2026-01-01T00:00:00+00:00",
        "prediction_commitment_sha256": "prediction",
        "frozen_model_sha256": "model",
        "development_dataset_sha256": "development",
        "dataset": {"clips": 4, "videos": 2},
        "regression": {"quality_score": {"mae": 1.0}},
        "ranking": {"top_1_hit_rate": 1.0},
        "records": records,
    }

    summary = build_public_holdout_summary(
        evaluation, reviews, bootstrap_iterations=100, seed=1
    )

    assert "records" not in summary
    assert summary["exploratory_product_metrics"]["top_3_hit_rate"] == 1.0
    assert summary["conditional_boundary_edit_diagnostics"]["edited_clips"] == 1
    assert summary["conditional_boundary_edit_diagnostics"][
        "mean_quality_change_when_an_edit_was_chosen"
    ] == 4.0


def test_public_summary_rejects_mismatched_reviews():
    evaluation = {
        "records": [evaluation_record("a", "video_a", 5.0, 4.0)],
    }

    with pytest.raises(ValueError, match="different annotation IDs"):
        build_public_holdout_summary(evaluation, [], bootstrap_iterations=10)
