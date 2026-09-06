import pytest

from creatorcut.candidates import (
    SentenceUnit,
    evaluate_candidate_recall,
    generate_candidates,
    interval_iou,
    words_to_sentence_units,
)


def test_words_to_sentence_units_uses_punctuation_and_pause() -> None:
    words = [
        {"start": 0.0, "end": 0.4, "word": " Hello"},
        {"start": 0.4, "end": 0.8, "word": " world."},
        {"start": 1.0, "end": 1.4, "word": " New"},
        {"start": 1.4, "end": 3.5, "word": " thought"},
        {"start": 5.0, "end": 5.4, "word": " After"},
        {"start": 5.4, "end": 5.8, "word": " pause"},
    ]

    units = words_to_sentence_units(words)

    assert [unit.text for unit in units] == ["Hello world.", "New thought", "After pause"]


def test_generate_candidates_selects_sentence_aligned_target_durations() -> None:
    units = [
        SentenceUnit(
            start=float(index * 10),
            end=float((index + 1) * 10),
            text=str(index),
            word_count=1,
        )
        for index in range(7)
    ]

    candidates = generate_candidates(
        "video_001",
        units,
        minimum_duration=20.0,
        maximum_duration=60.0,
        target_durations=(20.0, 45.0, 60.0),
    )

    first_start = [item for item in candidates if item["start_seconds"] == 0.0]
    assert [item["duration_seconds"] for item in first_start] == [20.0, 40.0, 60.0]
    assert candidates[0]["candidate_id"] == "video_001_candidate_0001"


def test_interval_iou() -> None:
    assert interval_iou((0.0, 10.0), (5.0, 15.0)) == pytest.approx(1 / 3)
    assert interval_iou((0.0, 10.0), (10.0, 20.0)) == 0.0


def test_evaluate_candidate_recall_finds_best_match() -> None:
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
    candidates = [
        {
            "candidate_id": "candidate_001",
            "video_id": "video_001",
            "start_seconds": 2.0,
            "end_seconds": 32.0,
        },
        {
            "candidate_id": "candidate_002",
            "video_id": "video_001",
            "start_seconds": 20.0,
            "end_seconds": 50.0,
        },
    ]

    result = evaluate_candidate_recall(annotations, candidates)

    assert result["recall"]["strong_at_iou_0.50"] == 1.0
    assert result["matches"][0]["best_candidate_id"] == "candidate_001"
