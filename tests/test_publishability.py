import pytest

from creatorcut.publishability import (
    assess_publishability,
    attach_publishability,
    evaluate_publishability_rules,
    summarize_publishability_batch,
)


def candidate(text, duration=30.0):
    return {
        "text": text,
        "duration_seconds": duration,
        "word_count": len(text.split()),
        "sentence_count": max(1, text.count(".")),
    }


def test_sponsor_read_is_blocked_by_high_precision_rule():
    result = assess_publishability(
        candidate("This episode is sponsored by Example. Use promo code CREATOR20.")
    )

    assert result["eligible"] is False
    assert result["score"] == 0.0
    assert "advertisement_language" in {reason["code"] for reason in result["reasons"]}


def test_music_marker_without_speech_is_blocked():
    result = assess_publishability(candidate("[Music] ♪"))

    assert result["eligible"] is False
    assert result["signals"]["music_marker_count"] == 2


def test_boundary_risks_are_reviewable_instead_of_hard_blocked():
    result = assess_publishability(
        candidate("And this point needs the previous context and does not finish")
    )

    assert result["eligible"] is True
    assert result["score"] < 1.0
    assert {reason["code"] for reason in result["reasons"]} >= {
        "context_dependent_start",
        "incomplete_ending",
    }


def test_clean_candidate_has_no_publishability_adjustment():
    result = attach_publishability(
        candidate(
            "Here is a complete example with enough spoken detail "
            "to explain the central idea clearly.",
            duration=12.0,
        )
    )

    assert result["publishability"]["eligible"] is True
    assert result["publishability"]["score"] == 1.0
    assert result["publishability_adjustment"] == pytest.approx(0.0)


def test_batch_summary_preserves_blocked_candidate_counts():
    assessed = [
        attach_publishability(candidate("This episode is brought to you by Example.")),
        attach_publishability(candidate("Here is a complete useful idea.", duration=5.0)),
    ]

    summary = summarize_publishability_batch(assessed)

    assert summary["candidate_count"] == 2
    assert summary["eligible_count"] == 1
    assert summary["blocked_count"] == 1
    assert summary["blocked_reason_counts"] == {"advertisement_language": 1}


def test_shadow_evaluation_keeps_human_best_safe():
    records = [
        {
            "annotation_id": "video_a_ad",
            "video_id": "video_a",
            "transcript_text": "This episode is brought to you by Example.",
            "duration_seconds": 20.0,
            "features": {"word_count": 8, "sentence_count": 1},
            "targets": {"quality_score": 1.0},
        },
        {
            "annotation_id": "video_a_best",
            "video_id": "video_a",
            "transcript_text": (
                "Here is a complete useful explanation with enough detail for the audience."
            ),
            "duration_seconds": 10.0,
            "features": {"word_count": 12, "sentence_count": 1},
            "targets": {"quality_score": 5.0},
        },
    ]

    summary = evaluate_publishability_rules(records)

    assert summary["hard_blocked"] == 1
    assert summary["mean_blocked_human_quality"] == 1.0
    assert summary["videos_with_every_human_best_clip_blocked"] == 0
