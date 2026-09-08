"""Platform-aware clip planning and bounded ranking adjustments."""

from __future__ import annotations

import math
from typing import Any

from creatorcut.candidates import interval_iou

SUPPORTED_PLATFORMS = ("youtube", "instagram", "tiktok")
MAXIMUM_CLIPS_PER_PLATFORM = 8
PLATFORM_PROFILE_VERSION = "creatorcut_platform_profiles_v1"

PLATFORM_PROFILES: dict[str, dict[str, Any]] = {
    "youtube": {
        "label": "YouTube Shorts",
        "aspect_ratio": "9:16",
        "export_width": 720,
        "export_height": 1280,
        "preferred_duration_seconds": 45.0,
        "generation_range_seconds": [20.0, 60.0],
        "default_export_format": "vertical_captions",
    },
    "instagram": {
        "label": "Instagram Reels",
        "aspect_ratio": "9:16",
        "export_width": 720,
        "export_height": 1280,
        "preferred_duration_seconds": 32.0,
        "generation_range_seconds": [20.0, 60.0],
        "default_export_format": "vertical_captions",
    },
    "tiktok": {
        "label": "TikTok",
        "aspect_ratio": "9:16",
        "export_width": 720,
        "export_height": 1280,
        "preferred_duration_seconds": 25.0,
        "generation_range_seconds": [20.0, 60.0],
        "default_export_format": "vertical_captions",
    },
}


def validate_clip_plan(value: Any) -> dict[str, Any]:
    """Validate requested platforms and optional per-platform clip counts."""
    if not isinstance(value, dict):
        raise ValueError("Choose at least one output platform")
    platforms: dict[str, dict[str, int | None]] = {}
    for platform, count in value.items():
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError("Choose a supported output platform")
        if count in (None, ""):
            checked_count = None
        else:
            if isinstance(count, bool):
                raise ValueError("Clip counts must be whole numbers")
            try:
                checked_count = int(count)
            except (TypeError, ValueError) as error:
                raise ValueError("Clip counts must be whole numbers") from error
            if str(checked_count) != str(count).strip() and not isinstance(count, int):
                raise ValueError("Clip counts must be whole numbers")
            if not 1 <= checked_count <= MAXIMUM_CLIPS_PER_PLATFORM:
                raise ValueError(
                    f"Choose between 1 and {MAXIMUM_CLIPS_PER_PLATFORM} clips per platform"
                )
        platforms[platform] = {"requested_count": checked_count}
    if not platforms:
        raise ValueError("Choose at least one output platform")
    return {
        "schema": "creatorcut_clip_plan_v1",
        "profile_version": PLATFORM_PROFILE_VERSION,
        "platforms": platforms,
    }


def _duration_fit(duration_seconds: float, target_seconds: float) -> float:
    return max(-1.0, 1.0 - abs(duration_seconds - target_seconds) / 25.0)


def platform_adjustment(candidate: dict[str, Any], platform: str) -> float:
    """Return a small, inspectable platform prior without replacing the frozen ranker."""
    if platform not in PLATFORM_PROFILES:
        raise ValueError("Choose a supported output platform")
    profile = PLATFORM_PROFILES[platform]
    targets = candidate.get("predicted_targets", {})
    delivery = candidate.get("multimodal_features", {})
    normalized = {
        field: (float(targets.get(field, 3.0)) - 3.0) / 2.0
        for field in ("hook", "completeness", "payoff", "clarity")
    }
    duration = _duration_fit(
        float(candidate["duration_seconds"]),
        float(profile["preferred_duration_seconds"]),
    )
    urgency = 2.0 * float(delivery.get("audio_urgency_score", 0.5)) - 1.0
    excitement = 2.0 * float(delivery.get("visual_excitement_score", 0.5)) - 1.0
    weights = {
        "youtube": {
            "duration": 0.050,
            "hook": 0.025,
            "completeness": 0.030,
            "payoff": 0.035,
            "clarity": 0.020,
            "urgency": 0.010,
            "excitement": 0.010,
        },
        "instagram": {
            "duration": 0.050,
            "hook": 0.035,
            "completeness": 0.015,
            "payoff": 0.020,
            "clarity": 0.020,
            "urgency": 0.025,
            "excitement": 0.045,
        },
        "tiktok": {
            "duration": 0.060,
            "hook": 0.050,
            "completeness": 0.010,
            "payoff": 0.020,
            "clarity": 0.010,
            "urgency": 0.045,
            "excitement": 0.045,
        },
    }[platform]
    raw = (
        weights["duration"] * duration
        + weights["hook"] * normalized["hook"]
        + weights["completeness"] * normalized["completeness"]
        + weights["payoff"] * normalized["payoff"]
        + weights["clarity"] * normalized["clarity"]
        + weights["urgency"] * urgency
        + weights["excitement"] * excitement
    )
    return max(-0.18, min(0.18, raw))


def attach_platform_scores(
    candidates: list[dict[str, Any]], platform: str
) -> list[dict[str, Any]]:
    """Attach the bounded platform prior to each already personalized candidate."""
    return [
        {
            **candidate,
            "platform": platform,
            "platform_adjustment": platform_adjustment(candidate, platform),
            "platform_score": float(candidate["personalized_score"])
            + platform_adjustment(candidate, platform),
        }
        for candidate in candidates
    ]


def _strict_diverse(
    candidates: list[dict[str, Any]], count: int, maximum_iou: float = 0.10
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate["platform_score"]),
            -float(candidate["global_score"]),
            candidate["candidate_id"],
        ),
    )
    selected: list[dict[str, Any]] = []
    for candidate in ordered:
        interval = (float(candidate["start_seconds"]), float(candidate["end_seconds"]))
        if all(
            interval_iou(
                interval,
                (float(existing["start_seconds"]), float(existing["end_seconds"])),
            )
            <= maximum_iou
            for existing in selected
        ):
            selected.append(candidate)
        if len(selected) == count:
            break
    return selected


def plan_platform_clips(
    candidates: list[dict[str, Any]],
    platform: str,
    media_duration_seconds: float,
    requested_count: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose non-repetitive clips and explain the recommended count range."""
    scored = attach_platform_scores(candidates, platform)
    if not scored:
        return [], {
            "requested_count": requested_count,
            "recommended_count": 0,
            "suggested_range": [0, 0],
            "available_non_overlapping_count": 0,
            "delivered_count": 0,
        }
    best_score = max(float(candidate["platform_score"]) for candidate in scored)
    quality_pool = [
        candidate
        for candidate in scored
        if float(candidate["platform_score"]) >= best_score - 0.65
    ]
    diverse_quality = _strict_diverse(quality_pool, MAXIMUM_CLIPS_PER_PLATFORM)
    diverse_all = _strict_diverse(scored, MAXIMUM_CLIPS_PER_PLATFORM)
    duration_budget = max(1, min(6, math.ceil(media_duration_seconds / 300.0)))
    recommended = min(len(diverse_quality), duration_budget)
    if recommended == 0 and diverse_all:
        recommended = 1
    range_low = max(1, recommended - 1) if recommended else 0
    range_high = min(len(diverse_all), MAXIMUM_CLIPS_PER_PLATFORM, recommended + 1)
    delivery_count = recommended if requested_count is None else requested_count
    selected = _strict_diverse(scored, min(delivery_count, len(diverse_all)))
    return selected, {
        "requested_count": requested_count,
        "recommended_count": recommended,
        "suggested_range": [range_low, range_high],
        "available_non_overlapping_count": len(diverse_all),
        "delivered_count": len(selected),
        "profile": PLATFORM_PROFILES[platform],
        "rationale": (
            "Count balances source duration, candidate quality, and temporal overlap. "
            "Requested counts are capped when the source has fewer distinct usable moments."
        ),
    }


def explain_platform_fit(candidate: dict[str, Any]) -> str:
    """Summarize the strongest platform-specific delivery reason."""
    platform = candidate["platform"]
    delivery = candidate.get("multimodal_features", {})
    reasons = [
        (
            abs(
                float(candidate["duration_seconds"])
                - PLATFORM_PROFILES[platform]["preferred_duration_seconds"]
            ),
            "duration fit",
        ),
        (-float(delivery.get("audio_urgency_score", 0.5)), "audio urgency"),
        (-float(delivery.get("visual_excitement_score", 0.5)), "visual activity"),
    ]
    reasons.sort(key=lambda item: item[0])
    return f"{PLATFORM_PROFILES[platform]['label']} fit emphasizes {reasons[0][1]}."
