import pytest

from creatorcut.sampling import (
    SCORE_BANDS,
    build_annotation_queue,
    partition_score_bands,
    sample_video_candidates,
    summarize_queue,
)


def scored_candidate(index: int, video_id: str = "video_004") -> dict:
    start = float(index * 70)
    return {
        "candidate_id": f"{video_id}_candidate_{index:04d}",
        "video_id": video_id,
        "start_seconds": start,
        "end_seconds": start + 20.0 + index,
        "duration_seconds": 20.0 + index,
        "sentence_count": 3,
        "word_count": 60,
        "text": "Why does this matter? Here is one complete answer.",
        "transcript_heuristic_score": float(index),
    }


def test_partition_score_bands_balances_candidates():
    bands = partition_score_bands([scored_candidate(index) for index in range(12)])

    assert {band: len(items) for band, items in bands.items()} == {
        "low": 4,
        "medium": 4,
        "high": 4,
    }
    assert max(item["transcript_heuristic_score"] for item in bands["low"]) < min(
        item["transcript_heuristic_score"] for item in bands["medium"]
    )


def test_sample_video_candidates_is_balanced_and_deterministic():
    candidates = [scored_candidate(index) for index in range(18)]

    first = sample_video_candidates(candidates, samples_per_video=6, seed=7)
    second = sample_video_candidates(candidates, samples_per_video=6, seed=7)

    assert [item["candidate_id"] for item in first] == [item["candidate_id"] for item in second]
    assert {
        band: sum(item["sampling_score_band"] == band for item in first) for band in SCORE_BANDS
    } == {"low": 2, "medium": 2, "high": 2}


def test_sample_video_candidates_requires_balanced_count():
    with pytest.raises(ValueError, match="multiple of 3"):
        sample_video_candidates(
            [scored_candidate(index) for index in range(9)], samples_per_video=4
        )


def test_build_annotation_queue_excludes_previously_labeled_videos():
    candidates = [
        *[scored_candidate(index, "video_001") for index in range(12)],
        *[scored_candidate(index, "video_004") for index in range(12)],
    ]
    for candidate in candidates:
        candidate.pop("transcript_heuristic_score")
    annotations = [{"video_id": "video_001"}]

    queue = build_annotation_queue(candidates, annotations, samples_per_video=6)
    summary = summarize_queue(queue)

    assert {item["video_id"] for item in queue} == {"video_004"}
    assert len({item["candidate_id"] for item in queue}) == 6
    assert summary["annotation_tasks"] == 6
    assert summary["score_bands"] == {"low": 2, "medium": 2, "high": 2}


def test_build_annotation_queue_excludes_quality_warning_videos():
    candidates = [
        *[scored_candidate(index, "video_004") for index in range(12)],
        *[scored_candidate(index, "video_005") for index in range(12)],
    ]
    for candidate in candidates:
        candidate.pop("transcript_heuristic_score")

    queue = build_annotation_queue(
        candidates,
        annotations=[],
        samples_per_video=6,
        excluded_video_ids={"video_004"},
    )

    assert {item["video_id"] for item in queue} == {"video_005"}
