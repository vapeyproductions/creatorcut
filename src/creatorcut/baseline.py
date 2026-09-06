"""Transparent transcript baselines for clip ranking."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from creatorcut.candidates import core_relevance, interval_iou, write_jsonl
from creatorcut.dataset import load_jsonl

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")
COMPLETE_END = re.compile(r"[.!?][\"'”’)]*$")

CONTEXT_OPENERS = {
    "also",
    "and",
    "because",
    "but",
    "he",
    "her",
    "him",
    "it",
    "its",
    "she",
    "so",
    "that",
    "their",
    "them",
    "they",
    "this",
    "those",
    "we",
    "which",
}

HOOK_CUES = {
    "biggest",
    "danger",
    "how",
    "imagine",
    "important",
    "mistake",
    "never",
    "problem",
    "reason",
    "secret",
    "surprising",
    "truth",
    "want",
    "why",
}

FILLERS = {"actually", "basically", "honestly", "like", "literally", "um", "uh"}

INTRO_OUTRO_PHRASES = (
    "good morning",
    "in today's episode",
    "subscribe",
    "thanks for listening",
    "thanks for watching",
    "welcome back",
)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Constrain a numeric value to a closed interval."""
    return max(minimum, min(maximum, value))


def tokenize(text: str) -> list[str]:
    """Return lowercase word tokens for lightweight lexical features."""
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def extract_transcript_features(candidate: dict[str, Any]) -> dict[str, float]:
    """Extract inexpensive, interpretable features from one candidate."""
    text = candidate["text"].strip()
    lowered = text.lower()
    tokens = tokenize(text)
    first_tokens = tokens[:12]
    duration = float(candidate["duration_seconds"])
    word_count = int(candidate["word_count"])
    sentence_count = int(candidate["sentence_count"])
    words_per_second = word_count / duration if duration else 0.0

    first_token = tokens[0] if tokens else ""
    clean_start = float(first_token not in CONTEXT_OPENERS)
    complete_end = float(bool(COMPLETE_END.search(text)))
    hook_cue_count = sum(token in HOOK_CUES for token in first_tokens)
    hook_signal = clamp(
        0.50 * min(hook_cue_count, 2) / 2
        + 0.30 * float("?" in text[:160])
        + 0.20 * float(any(token.isdigit() for token in first_tokens))
    )
    duration_fit = clamp(1.0 - abs(duration - 40.0) / 30.0)
    density_fit = clamp(1.0 - abs(words_per_second - 2.5) / 1.5)
    structure_fit = clamp(1.0 - abs(sentence_count - 5) / 7.0)
    lexical_diversity = len(set(tokens)) / len(tokens) if tokens else 0.0
    lexical_fit = clamp((lexical_diversity - 0.35) / 0.40)
    filler_ratio = sum(token in FILLERS for token in tokens) / len(tokens) if tokens else 0.0
    filler_control = clamp(1.0 - filler_ratio / 0.08)
    intro_outro = float(any(phrase in lowered for phrase in INTRO_OUTRO_PHRASES))

    return {
        "clean_start": clean_start,
        "complete_end": complete_end,
        "hook_signal": hook_signal,
        "duration_fit": duration_fit,
        "words_per_second": words_per_second,
        "density_fit": density_fit,
        "structure_fit": structure_fit,
        "lexical_diversity": lexical_diversity,
        "lexical_fit": lexical_fit,
        "filler_ratio": filler_ratio,
        "filler_control": filler_control,
        "intro_outro": intro_outro,
    }


def transcript_heuristic_score(features: dict[str, float]) -> float:
    """Combine features using fixed, pre-evaluation weights."""
    standalone = 0.50 * features["clean_start"] + 0.50 * features["complete_end"]
    weighted_score = (
        0.25 * standalone
        + 0.20 * features["hook_signal"]
        + 0.15 * features["density_fit"]
        + 0.15 * features["structure_fit"]
        + 0.10 * features["duration_fit"]
        + 0.10 * features["lexical_fit"]
        + 0.05 * features["filler_control"]
    )
    penalty = 0.15 * features["intro_outro"]
    return round(100.0 * clamp(weighted_score - penalty), 4)


def score_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach duration-only and transcript-heuristic scores."""
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        features = extract_transcript_features(candidate)
        scored.append(
            {
                **candidate,
                "features": features,
                "duration_only_score": round(100.0 * features["duration_fit"], 4),
                "transcript_heuristic_score": transcript_heuristic_score(features),
            }
        )
    return scored


def select_diverse_candidates(
    candidates: list[dict[str, Any]],
    score_field: str,
    limit: int,
    maximum_overlap_iou: float = 0.50,
) -> list[dict[str, Any]]:
    """Greedily select high-scoring candidates while suppressing near duplicates."""
    ranked = sorted(
        candidates,
        key=lambda candidate: (-candidate[score_field], candidate["start_seconds"]),
    )
    selected: list[dict[str, Any]] = []
    for candidate in ranked:
        interval = (candidate["start_seconds"], candidate["end_seconds"])
        if all(
            interval_iou(
                interval,
                (existing["start_seconds"], existing["end_seconds"]),
            )
            <= maximum_overlap_iou
            for existing in selected
        ):
            selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def average_ranks(values: list[float]) -> list[float]:
    """Assign ascending average ranks, including ties."""
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        average_rank = ((position + 1) + end) / 2
        for index in range(position, end):
            ranks[ordered[index][0]] = average_rank
        position = end
    return ranks


def pearson_correlation(first: list[float], second: list[float]) -> float | None:
    """Calculate Pearson correlation, returning None for constant inputs."""
    if len(first) != len(second) or not first:
        raise ValueError("Expected two non-empty vectors with equal length")
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    numerator = sum(
        (first_value - first_mean) * (second_value - second_mean)
        for first_value, second_value in zip(first, second, strict=True)
    )
    first_scale = math.sqrt(sum((value - first_mean) ** 2 for value in first))
    second_scale = math.sqrt(sum((value - second_mean) ** 2 for value in second))
    if first_scale == 0 or second_scale == 0:
        return None
    return numerator / (first_scale * second_scale)


def spearman_correlation(first: list[float], second: list[float]) -> float | None:
    """Calculate Spearman rank correlation with average ranks for ties."""
    return pearson_correlation(average_ranks(first), average_ranks(second))


def match_annotations_to_candidate_scores(
    annotations: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    score_field: str,
) -> list[dict[str, Any]]:
    """Associate each annotation with the most-overlapping candidate and its score."""
    candidates_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_video[candidate["video_id"]].append(candidate)

    matches: list[dict[str, Any]] = []
    for annotation in annotations:
        relevant_candidates = candidates_by_video[annotation["video_id"]]
        best_candidate = max(
            relevant_candidates,
            key=lambda candidate: interval_iou(
                (annotation["start_seconds"], annotation["end_seconds"]),
                (candidate["start_seconds"], candidate["end_seconds"]),
            ),
        )
        matches.append(
            {
                "annotation_id": annotation["annotation_id"],
                "human_relevance": core_relevance(annotation),
                "candidate_id": best_candidate["candidate_id"],
                "candidate_score": best_candidate[score_field],
                "interval_iou": interval_iou(
                    (annotation["start_seconds"], annotation["end_seconds"]),
                    (best_candidate["start_seconds"], best_candidate["end_seconds"]),
                ),
            }
        )
    return matches


def evaluate_baseline(
    annotations: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    score_field: str,
    strong_threshold: float = 4.0,
    match_iou: float = 0.50,
    top_ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Evaluate known-positive recall and human-score rank correlation."""
    candidates_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_video[candidate["video_id"]].append(candidate)

    strong_annotations = [
        annotation for annotation in annotations if core_relevance(annotation) >= strong_threshold
    ]
    recalls: dict[str, float] = {}
    selections: dict[str, list[dict[str, Any]]] = {}

    for top_k in top_ks:
        selected_by_video = {
            video_id: select_diverse_candidates(video_candidates, score_field, top_k)
            for video_id, video_candidates in candidates_by_video.items()
        }
        hits = 0
        for annotation in strong_annotations:
            human_interval = (annotation["start_seconds"], annotation["end_seconds"])
            hit = any(
                interval_iou(
                    human_interval,
                    (candidate["start_seconds"], candidate["end_seconds"]),
                )
                >= match_iou
                for candidate in selected_by_video.get(annotation["video_id"], [])
            )
            hits += hit
        recalls[f"known_strong_recall_at_{top_k}"] = (
            hits / len(strong_annotations) if strong_annotations else 0.0
        )
        if top_k == 3:
            selections = {
                video_id: [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "start_seconds": candidate["start_seconds"],
                        "end_seconds": candidate["end_seconds"],
                        "score": candidate[score_field],
                        "text_preview": candidate["text"][:160],
                    }
                    for candidate in selected
                ]
                for video_id, selected in selected_by_video.items()
            }

    annotation_matches = match_annotations_to_candidate_scores(
        annotations, candidates, score_field
    )
    correlation = spearman_correlation(
        [match["human_relevance"] for match in annotation_matches],
        [match["candidate_score"] for match in annotation_matches],
    )

    return {
        "score_field": score_field,
        "strong_threshold": strong_threshold,
        "match_iou": match_iou,
        "strong_annotation_count": len(strong_annotations),
        "known_positive_recall": recalls,
        "human_score_spearman": correlation,
        "top_3_by_video": selections,
        "annotation_matches": annotation_matches,
    }


def score_main() -> None:
    """Score every generated candidate from the command line."""
    parser = argparse.ArgumentParser(description="Score candidates with transparent baselines")
    parser.add_argument("--input", type=Path, default=Path("data/processed/candidates.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/scored_candidates.jsonl")
    )
    args = parser.parse_args()

    scored = score_candidates(load_jsonl(args.input))
    write_jsonl(scored, args.output)
    print(f"Scored {len(scored)} candidates and wrote {args.output}")


def evaluate_main() -> None:
    """Evaluate the baseline rankings from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate CreatorCut ranking baselines")
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/annotations/seed_labels.jsonl")
    )
    parser.add_argument(
        "--candidates", type=Path, default=Path("data/processed/scored_candidates.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/baseline_evaluation.json")
    )
    args = parser.parse_args()

    annotations = load_jsonl(args.annotations)
    candidates = load_jsonl(args.candidates)
    evaluation = {
        "evaluation_note": (
            "Recall is measured only against known human positives. Unlabeled candidates "
            "must not be treated as negatives."
        ),
        "baselines": {
            score_field: evaluate_baseline(annotations, candidates, score_field)
            for score_field in ("duration_only_score", "transcript_heuristic_score")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    summary = {
        name: {
            "known_positive_recall": result["known_positive_recall"],
            "human_score_spearman": result["human_score_spearman"],
        }
        for name, result in evaluation["baselines"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote detailed baseline evaluation to {args.output}")
