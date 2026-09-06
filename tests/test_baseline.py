import pytest

from creatorcut.baseline import (
    average_ranks,
    evaluate_baseline,
    extract_transcript_features,
    score_candidates,
    select_diverse_candidates,
    spearman_correlation,
)


def candidate(
    candidate_id: str,
    start: float,
    end: float,
    text: str = "Why does this matter? Here is the answer.",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "video_id": "video_001",
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": end - start,
        "sentence_count": 3,
        "word_count": 75,
        "text": text,
        "start_unit_index": 0,
        "end_unit_index": 2,
    }


def test_extract_transcript_features_detects_clean_hook() -> None:
    features = extract_transcript_features(candidate("candidate_001", 0.0, 30.0))

    assert features["clean_start"] == 1.0
    assert features["complete_end"] == 1.0
    assert features["hook_signal"] > 0.0
    assert features["words_per_second"] == pytest.approx(2.5)


def test_select_diverse_candidates_suppresses_overlapping_results() -> None:
    candidates = [
        {**candidate("first", 0.0, 40.0), "score": 100.0},
        {**candidate("overlap", 2.0, 42.0), "score": 99.0},
        {**candidate("different", 50.0, 90.0), "score": 80.0},
    ]

    selected = select_diverse_candidates(candidates, "score", 2)

    assert [item["candidate_id"] for item in selected] == ["first", "different"]


def test_average_ranks_handles_ties() -> None:
    assert average_ranks([10.0, 20.0, 20.0, 40.0]) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_correlation_recognizes_same_order() -> None:
    assert spearman_correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)


def test_evaluate_baseline_measures_known_positive_recall() -> None:
    annotations = [
        {
            "annotation_id": "clip_001",
            "video_id": "video_001",
            "start_seconds": 0.0,
            "end_seconds": 30.0,
            "hook": 5,
            "completeness": 5,
            "payoff": 5,
            "clarity": 5,
        }
    ]
    candidates = score_candidates(
        [
            candidate("matching", 0.0, 30.0),
            candidate("different", 60.0, 90.0, text="Ordinary words continue here."),
        ]
    )

    result = evaluate_baseline(
        annotations,
        candidates,
        "transcript_heuristic_score",
        top_ks=(1,),
    )

    assert result["known_positive_recall"]["known_strong_recall_at_1"] == 1.0
