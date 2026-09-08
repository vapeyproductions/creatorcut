import pytest

from creatorcut.product_pipeline import (
    export_dimensions,
    prefilter_candidates,
    select_diverse_top_clips,
    validate_export_interval,
)


def candidate(candidate_id, start, end, score=3.0):
    return {
        "candidate_id": candidate_id,
        "video_id": "upload_test",
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": end - start,
        "sentence_count": 3,
        "word_count": 60,
        "text": "Here is a complete and useful idea that you need to know.",
        "personalized_score": score,
        "global_score": score,
    }


def test_select_diverse_top_clips_avoids_repeating_the_same_moment():
    candidates = [
        candidate("first", 0.0, 40.0, 5.0),
        candidate("overlap", 2.0, 42.0, 4.9),
        candidate("second", 60.0, 100.0, 4.5),
        candidate("third", 120.0, 160.0, 4.0),
    ]

    selected = select_diverse_top_clips(candidates)

    assert [item["candidate_id"] for item in selected] == ["first", "second", "third"]


def test_prefilter_candidates_is_deterministic_and_limited():
    candidates = [
        candidate(f"candidate_{index:03d}", index * 5.0, index * 5.0 + 30.0)
        for index in range(80)
    ]

    first = prefilter_candidates(candidates, limit=20)
    second = prefilter_candidates(candidates, limit=20)

    assert len(first) == 20
    assert [item["candidate_id"] for item in first] == [item["candidate_id"] for item in second]
    assert min(item["start_seconds"] for item in first) < 40
    assert max(item["start_seconds"] for item in first) > 300


def test_validate_export_interval_accepts_a_small_edit():
    clip = candidate("clip", 30.0, 70.0)

    assert validate_export_interval(clip, 31.2, 67.8, 120.0) == (31.2, 67.8)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [(3840, 2160, (1920, 1080)), (1080, 1920, (1080, 1920)), (1280, 720, (1280, 720))],
)
def test_export_dimensions_preserve_aspect_ratio_within_social_video_limits(
    width, height, expected
):
    assert export_dimensions(width, height) == expected


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (14.9, 70.0, "edited start"),
        (30.0, 85.1, "edited end"),
        (30.0, 125.0, "outside the uploaded video"),
    ],
)
def test_validate_export_interval_rejects_unsafe_edits(start, end, message):
    clip = candidate("clip", 30.0, 70.0)

    with pytest.raises(ValueError, match=message):
        validate_export_interval(clip, start, end, 120.0)
