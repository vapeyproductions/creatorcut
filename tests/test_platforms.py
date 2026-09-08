import pytest

from creatorcut.delivery import normalize_delivery_features
from creatorcut.platforms import (
    attach_platform_scores,
    plan_platform_clips,
    validate_clip_plan,
)
from creatorcut.product_pipeline import select_multimodal_candidates


def candidate(candidate_id, start, duration, score=4.0, urgency=0.5, excitement=0.5):
    return {
        "candidate_id": candidate_id,
        "start_seconds": start,
        "end_seconds": start + duration,
        "duration_seconds": duration,
        "global_score": score,
        "personalized_score": score,
        "predicted_targets": {
            "hook": 4.0,
            "completeness": 4.0,
            "payoff": 4.0,
            "clarity": 4.0,
        },
        "multimodal_features": {
            "audio_urgency_score": urgency,
            "visual_excitement_score": excitement,
        },
    }


def test_validate_clip_plan_accepts_auto_and_requested_counts():
    plan = validate_clip_plan({"youtube": "", "instagram": "3"})

    assert plan["platforms"]["youtube"]["requested_count"] is None
    assert plan["platforms"]["instagram"]["requested_count"] == 3
    with pytest.raises(ValueError, match="between 1 and 8"):
        validate_clip_plan({"tiktok": 9})


def test_platform_scoring_prefers_different_delivery_profiles():
    slower = candidate("slower", 0.0, 45.0, urgency=0.2, excitement=0.2)
    faster = candidate("faster", 90.0, 25.0, urgency=0.9, excitement=0.9)

    youtube = attach_platform_scores([slower, faster], "youtube")
    tiktok = attach_platform_scores([slower, faster], "tiktok")

    assert youtube[0]["platform_adjustment"] > youtube[1]["platform_adjustment"]
    assert tiktok[1]["platform_adjustment"] > tiktok[0]["platform_adjustment"]


def test_auto_clip_plan_caps_output_at_distinct_non_overlapping_moments():
    candidates = [
        candidate("one", 0.0, 30.0, 5.0),
        candidate("overlap", 2.0, 30.0, 4.9),
        candidate("two", 80.0, 30.0, 4.8),
        candidate("three", 160.0, 30.0, 4.7),
    ]

    selected, plan = plan_platform_clips(candidates, "tiktok", 900.0, 8)

    assert [item["candidate_id"] for item in selected] == ["one", "two", "three"]
    assert plan["available_non_overlapping_count"] == 3
    assert plan["delivered_count"] == 3
    assert plan["recommended_count"] == 3


def test_delivery_normalization_produces_relative_audio_and_visual_scores():
    records = [
        {
            "audio_energy_db": -30.0,
            "audio_variation_db": 2.0,
            "audio_opening_delta_db": -1.0,
            "audio_silence_ratio": 0.4,
            "audio_clipping_ratio": 0.0,
            "speech_words_per_second": 1.5,
            "visual_colorfulness": 0.1,
            "visual_saturation": 0.2,
            "visual_motion": 0.05,
            "visual_luminance_variation": 0.1,
            "visual_scene_change_rate": 0.0,
        },
        {
            "audio_energy_db": -15.0,
            "audio_variation_db": 8.0,
            "audio_opening_delta_db": 4.0,
            "audio_silence_ratio": 0.05,
            "audio_clipping_ratio": 0.0,
            "speech_words_per_second": 3.2,
            "visual_colorfulness": 0.6,
            "visual_saturation": 0.7,
            "visual_motion": 0.3,
            "visual_luminance_variation": 0.3,
            "visual_scene_change_rate": 0.4,
        },
    ]

    normalized = normalize_delivery_features(records)

    assert normalized[1]["audio_urgency_score"] > normalized[0]["audio_urgency_score"]
    assert normalized[1]["visual_excitement_score"] > normalized[0][
        "visual_excitement_score"
    ]


def test_multimodal_candidate_budget_handles_one_slot():
    candidates = [
        {
            "candidate_id": f"candidate_{index}",
            "global_score": float(index),
            "start_seconds": float(index * 30),
        }
        for index in range(3)
    ]

    selected = select_multimodal_candidates(candidates, limit=1)

    assert [candidate["candidate_id"] for candidate in selected] == ["candidate_2"]
