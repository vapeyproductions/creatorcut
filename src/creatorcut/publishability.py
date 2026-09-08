"""High-precision publishability rules kept separate from learned clip quality."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from creatorcut.baseline import extract_transcript_features, tokenize
from creatorcut.dataset import load_jsonl
from creatorcut.training import SENTENCE_BOUNDARY_PATTERN

PUBLISHABILITY_RULE_VERSION = "creatorcut_publishability_rules_v1"
MAXIMUM_PUBLISHABILITY_ADJUSTMENT = 0.35

ADVERTISEMENT_PATTERNS = (
    re.compile(r"\b(?:this|today'?s) (?:video|episode) is sponsored by\b", re.I),
    re.compile(r"\bbrought to you by\b", re.I),
    re.compile(r"\buse (?:the )?(?:promo )?code\b", re.I),
    re.compile(r"\bthanks? to .{1,80} for sponsoring\b", re.I),
)
MUSIC_MARKER = re.compile(r"\[(?:intro )?music\]|\((?:intro )?music\)|[♪♫]", re.I)


def _reason(code: str, severity: str, description: str, penalty: float) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "description": description,
        "penalty": penalty,
    }


def assess_publishability(candidate: dict[str, Any]) -> dict[str, Any]:
    """Assess obvious publishing risks without claiming a learned probability."""
    text = str(candidate.get("text") or candidate.get("transcript_text") or "").strip()
    duration = float(candidate["duration_seconds"])
    tokens = tokenize(text)
    sentence_count = max(1, len(SENTENCE_BOUNDARY_PATTERN.findall(text)))
    feature_input = {
        "text": text,
        "duration_seconds": duration,
        "word_count": int(candidate.get("word_count", len(tokens))),
        "sentence_count": int(candidate.get("sentence_count", sentence_count)),
    }
    features = extract_transcript_features(feature_input)
    reasons: list[dict[str, Any]] = []

    ad_match = next(
        (pattern.search(text) for pattern in ADVERTISEMENT_PATTERNS if pattern.search(text)),
        None,
    )
    if ad_match:
        reasons.append(
            _reason(
                "advertisement_language",
                "block",
                "The transcript contains a high-confidence sponsor or promo phrase.",
                1.0,
            )
        )

    music_markers = MUSIC_MARKER.findall(text)
    spoken_text = MUSIC_MARKER.sub(" ", text)
    spoken_word_count = len(tokenize(spoken_text))
    if music_markers and spoken_word_count < 8:
        reasons.append(
            _reason(
                "music_only",
                "block",
                "The interval appears to contain a music marker without enough speech.",
                1.0,
            )
        )

    if features["clean_start"] < 0.5:
        reasons.append(
            _reason(
                "context_dependent_start",
                "review",
                "The first word may depend on the preceding sentence.",
                0.18,
            )
        )
    if features["complete_end"] < 0.5:
        reasons.append(
            _reason(
                "incomplete_ending",
                "review",
                "The transcript does not end with sentence-completing punctuation.",
                0.20,
            )
        )
    if features["intro_outro"] >= 0.5:
        reasons.append(
            _reason(
                "intro_or_outro_language",
                "review",
                "The interval contains language commonly used in intros or outros.",
                0.28,
            )
        )
    if duration > 55:
        reasons.append(
            _reason(
                "long_interval",
                "review",
                "The interval is near the long end of the supported Shorts range.",
                min(0.15, 0.05 + (duration - 55) / 100),
            )
        )
    if features["words_per_second"] < 1.0:
        reasons.append(
            _reason(
                "sparse_speech",
                "review",
                "The transcript has unusually little speech for its duration.",
                0.15,
            )
        )
    elif features["words_per_second"] > 4.2:
        reasons.append(
            _reason(
                "dense_speech",
                "review",
                "The transcript may be difficult to follow at this speech density.",
                0.10,
            )
        )
    if features["filler_ratio"] > 0.10:
        reasons.append(
            _reason(
                "high_filler_ratio",
                "review",
                "The transcript contains an unusually high proportion of filler words.",
                0.10,
            )
        )

    eligible = not any(reason["severity"] == "block" for reason in reasons)
    review_penalty = sum(
        float(reason["penalty"]) for reason in reasons if reason["severity"] == "review"
    )
    score = 0.0 if not eligible else max(0.0, 1.0 - review_penalty)
    return {
        "rule_version": PUBLISHABILITY_RULE_VERSION,
        "eligible": eligible,
        "score": round(score, 4),
        "reasons": reasons,
        "signals": {
            "clean_start": bool(features["clean_start"]),
            "complete_end": bool(features["complete_end"]),
            "intro_or_outro_phrase": bool(features["intro_outro"]),
            "words_per_second": round(float(features["words_per_second"]), 4),
            "filler_ratio": round(float(features["filler_ratio"]), 4),
            "music_marker_count": len(music_markers),
            "spoken_word_count": spoken_word_count,
            "advertisement_phrase": ad_match.group(0) if ad_match else None,
        },
    }


def attach_publishability(candidate: dict[str, Any]) -> dict[str, Any]:
    """Attach a versioned assessment and bounded deterministic score adjustment."""
    assessment = assess_publishability(candidate)
    adjustment = (
        -MAXIMUM_PUBLISHABILITY_ADJUSTMENT * (1.0 - float(assessment["score"]))
        if assessment["eligible"]
        else -MAXIMUM_PUBLISHABILITY_ADJUSTMENT
    )
    return {
        **candidate,
        "publishability": assessment,
        "publishability_adjustment": round(adjustment, 6),
    }


def summarize_publishability_batch(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a complete candidate gate pass for operational auditing."""
    reason_counts: Counter[str] = Counter()
    blocked_reason_counts: Counter[str] = Counter()
    eligible_count = 0
    for candidate in candidates:
        assessment = candidate["publishability"]
        eligible_count += int(bool(assessment["eligible"]))
        for reason in assessment["reasons"]:
            reason_counts[reason["code"]] += 1
            if reason["severity"] == "block":
                blocked_reason_counts[reason["code"]] += 1
    return {
        "rule_version": PUBLISHABILITY_RULE_VERSION,
        "candidate_count": len(candidates),
        "eligible_count": eligible_count,
        "blocked_count": len(candidates) - eligible_count,
        "reason_counts": dict(sorted(reason_counts.items())),
        "blocked_reason_counts": dict(sorted(blocked_reason_counts.items())),
    }


def publishability_summary(assessment: dict[str, Any]) -> str:
    """Return restrained display copy for a persisted gate result."""
    if not assessment.get("eligible", True):
        return "Blocked by the publishability gate."
    review_reasons = [
        reason["description"]
        for reason in assessment.get("reasons", [])
        if reason.get("severity") == "review"
    ]
    if not review_reasons:
        return "No rule-based publishing risks detected."
    return "Review suggested: " + " ".join(review_reasons)


def evaluate_publishability_rules(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Shadow-evaluate rules on reviewed development clips without fitting thresholds."""
    if not records:
        raise ValueError("Publishability evaluation records are empty")
    assessed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    by_video: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    reason_counts: Counter[str] = Counter()
    for record in records:
        features = record.get("features", {})
        assessment = assess_publishability(
            {
                "transcript_text": record["transcript_text"],
                "duration_seconds": record["duration_seconds"],
                "word_count": features.get("word_count"),
                "sentence_count": features.get("sentence_count"),
            }
        )
        assessed.append((record, assessment))
        by_video.setdefault(record["video_id"], []).append((record, assessment))
        reason_counts.update(reason["code"] for reason in assessment["reasons"])
    blocked = [(record, result) for record, result in assessed if not result["eligible"]]
    all_best_blocked = 0
    for rows in by_video.values():
        best_quality = max(float(row[0]["targets"]["quality_score"]) for row in rows)
        best = [
            row
            for row in rows
            if float(row[0]["targets"]["quality_score"]) == best_quality
        ]
        all_best_blocked += int(all(not result["eligible"] for _, result in best))
    blocked_quality = [float(row["targets"]["quality_score"]) for row, _ in blocked]
    return {
        "schema": "creatorcut_publishability_gate_shadow_evaluation_v1",
        "rule_version": PUBLISHABILITY_RULE_VERSION,
        "evaluation_scope": "reviewed development clips; not an unseen production estimate",
        "records": len(records),
        "videos": len(by_video),
        "hard_blocked": len(blocked),
        "hard_block_rate": len(blocked) / len(records),
        "mean_blocked_human_quality": (
            sum(blocked_quality) / len(blocked_quality) if blocked_quality else None
        ),
        "videos_with_every_human_best_clip_blocked": all_best_blocked,
        "review_reason_counts": dict(sorted(reason_counts.items())),
        "mean_rules_score": sum(result["score"] for _, result in assessed) / len(assessed),
        "interpretation": (
            "This is a shadow check for obvious rule regressions. It does not estimate recall "
            "for unobserved ad, music, or context failures."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow-evaluate the publishability rules")
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("data/processed/v2/training_clips_attainable_quality.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/v2_development/publishability_gate_summary.json"),
    )
    args = parser.parse_args()
    summary = evaluate_publishability_rules(load_jsonl(args.records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
