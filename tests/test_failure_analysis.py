import pytest

from creatorcut.failure_analysis import build_failure_cases, summarize_failures
from creatorcut.training import MODEL_FEATURE_FIELDS


def clip(annotation_id: str, video_id: str, quality: float) -> dict:
    features = {field: 0.0 for field in MODEL_FEATURE_FIELDS}
    features["complete_end"] = 1.0
    return {
        "annotation_id": annotation_id,
        "candidate_id": f"candidate_{annotation_id}",
        "video_id": video_id,
        "start_seconds": 0.0,
        "end_seconds": 30.0,
        "duration_seconds": 30.0,
        "transcript_text": "A complete thought.",
        "features": features,
        "targets": {
            "hook": quality,
            "completeness": quality,
            "payoff": quality,
            "clarity": quality,
            "quality_score": quality,
        },
    }


def prediction(annotation_id: str, video_id: str, score: float) -> dict:
    return {
        "annotation_id": annotation_id,
        "video_id": video_id,
        "pointwise_hybrid_ridge_score": score,
    }


def test_build_failure_cases_returns_only_mistaken_top_selections() -> None:
    records = [
        clip("a_low", "video_a", 2.0),
        clip("a_high", "video_a", 5.0),
        clip("b_low", "video_b", 2.0),
        clip("b_high", "video_b", 5.0),
    ]
    predictions = [
        prediction("a_low", "video_a", 0.9),
        prediction("a_high", "video_a", 0.1),
        prediction("b_low", "video_b", 0.1),
        prediction("b_high", "video_b", 0.9),
    ]

    cases = build_failure_cases(records, predictions)

    assert len(cases) == 1
    assert cases[0]["analysis_id"] == "video_a_top1_failure"
    assert cases[0]["model_selected"]["annotation_id"] == "a_low"
    assert cases[0]["human_best"]["annotation_id"] == "a_high"
    assert cases[0]["regret"] == pytest.approx(3.0)
    assert cases[0]["target_gaps"]["hook"] == pytest.approx(3.0)


def test_build_failure_cases_requires_matching_prediction_ids() -> None:
    records = [clip("a", "video_a", 2.0)]

    with pytest.raises(ValueError, match="Prediction IDs do not match"):
        build_failure_cases(records, [])


def test_summarize_failures_reports_target_gaps_and_hypothesis_counts() -> None:
    records = [clip("low", "video_a", 2.0), clip("high", "video_a", 5.0)]
    records[0]["features"]["clean_start"] = 0.0
    cases = build_failure_cases(
        records,
        [prediction("low", "video_a", 0.9), prediction("high", "video_a", 0.1)],
    )

    summary = summarize_failures(cases, total_videos=2)

    assert summary["top_1_hits"] == 1
    assert summary["top_1_failures"] == 1
    assert summary["top_1_hit_rate"] == pytest.approx(0.5)
    assert summary["mean_human_best_minus_selected_target"]["hook"] == pytest.approx(3.0)
    assert summary["selected_clip_automatic_signal_counts"][
        "context_dependent_first_word"
    ] == 1
