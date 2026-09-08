"""Learn clean clip starts and ends from human boundary corrections."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from creatorcut.boundary_refinement import CONTEXT_DEPENDENT_OPENINGS, INTRO_OR_AD_PHRASES
from creatorcut.dataset import load_jsonl

BOUNDARY_DATA_SCHEMA = "boundary_candidates_multimodal_v1"
BOUNDARY_MODEL_SCHEMA = "boundary_choice_ranker_v1"
SEARCH_RADIUS_SECONDS = 35.0
MAXIMUM_PAIRS_PER_QUERY = 64
MATERIAL_ADJUSTMENT_SECONDS = 0.5
TOKEN = re.compile(r"[A-Za-z']+")
SENTENCE_END = re.compile(r"[.!?][\"'”’)]*$")
DANGLING_ENDINGS = {
    "and",
    "because",
    "but",
    "if",
    "or",
    "so",
    "that",
    "then",
    "which",
    "while",
}

BOUNDARY_FEATURE_FIELDS = (
    "offset_seconds",
    "absolute_offset_seconds",
    "original_duration_seconds",
    "candidate_duration_seconds",
    "absolute_duration_change_seconds",
    "duration_fit",
    "exact_original_boundary",
    "word_gap_before_seconds",
    "word_gap_after_seconds",
    "word_confidence",
    "previous_token_ends_sentence",
    "current_token_ends_sentence",
    "context_dependent_opening",
    "dangling_ending",
    "intro_or_ad_context",
    "position_in_video",
    "audio_energy_before_db",
    "audio_energy_after_db",
    "audio_energy_change_db",
    "audio_low_energy_ratio",
    "audio_min_frame_energy_db",
)


def _words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words = [
        word
        for segment in transcript.get("segments", [])
        for word in segment.get("words", [])
        if isinstance(word.get("start"), int | float)
        and isinstance(word.get("end"), int | float)
        and float(word["end"]) >= float(word["start"])
    ]
    return sorted(words, key=lambda word: (float(word["start"]), float(word["end"])))


def _token(text: str) -> str:
    matches = TOKEN.findall(text.lower())
    return matches[-1] if matches else ""


def _duration_fit(duration: float) -> float:
    if 20.0 <= duration <= 60.0:
        return 1.0
    return max(0.0, 1.0 - min(abs(duration - 40.0), 40.0) / 40.0)


def _rms_db(samples: np.ndarray) -> float:
    if not len(samples):
        return -80.0
    rms = math.sqrt(float(np.mean(np.asarray(samples, dtype=np.float64) ** 2)) + 1e-12)
    return max(-80.0, 20.0 * math.log10(rms + 1e-8))


def boundary_audio_features(
    audio: np.ndarray | None,
    time_seconds: float,
    sample_rate: int = 16_000,
) -> dict[str, float]:
    """Summarize energy immediately around a proposed edit point."""
    if audio is None:
        return {
            "audio_energy_before_db": 0.0,
            "audio_energy_after_db": 0.0,
            "audio_energy_change_db": 0.0,
            "audio_low_energy_ratio": 0.0,
            "audio_min_frame_energy_db": 0.0,
        }
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1 or not len(waveform):
        raise ValueError("audio must be a non-empty mono waveform")
    center = min(max(round(time_seconds * sample_rate), 0), len(waveform))
    half_window = round(0.50 * sample_rate)
    before = waveform[max(0, center - half_window) : center]
    after = waveform[center : min(len(waveform), center + half_window)]
    local = waveform[max(0, center - half_window) : min(len(waveform), center + half_window)]
    before_db = _rms_db(before)
    after_db = _rms_db(after)

    frame_length = max(1, round(0.04 * sample_rate))
    hop_length = max(1, round(0.02 * sample_rate))
    if len(local) < frame_length:
        local = np.pad(local, (0, frame_length - len(local)))
    frame_count = 1 + (len(local) - frame_length) // hop_length
    energies = np.asarray(
        [
            _rms_db(local[index * hop_length : index * hop_length + frame_length])
            for index in range(frame_count)
        ],
        dtype=float,
    )
    threshold = max(-55.0, float(np.percentile(energies, 90)) - 25.0)
    return {
        "audio_energy_before_db": before_db,
        "audio_energy_after_db": after_db,
        "audio_energy_change_db": after_db - before_db,
        "audio_low_energy_ratio": float(np.mean(energies < threshold)),
        "audio_min_frame_energy_db": float(np.min(energies)),
    }


def _nearest_word_index(words: list[dict[str, Any]], time_seconds: float, kind: str) -> int:
    field = "start" if kind == "start" else "end"
    return min(range(len(words)), key=lambda index: abs(float(words[index][field]) - time_seconds))


def _candidate_features(
    words: list[dict[str, Any]],
    word_index: int,
    kind: str,
    time_seconds: float,
    original_start: float,
    original_end: float,
    media_duration: float,
    audio: np.ndarray | None,
) -> dict[str, float]:
    word = words[word_index]
    previous_text = str(words[word_index - 1].get("word", "")).strip() if word_index else ""
    current_text = str(word.get("word", "")).strip()
    next_start = (
        float(words[word_index + 1]["start"])
        if word_index + 1 < len(words)
        else float(word["end"])
    )
    previous_end = float(words[word_index - 1]["end"]) if word_index else float(word["start"])
    gap_before = max(0.0, float(word["start"]) - previous_end)
    gap_after = max(0.0, next_start - float(word["end"]))
    context = " ".join(
        str(value.get("word", "")).strip()
        for value in words[max(0, word_index - 5) : word_index + 7]
    ).lower()
    candidate_duration = (
        original_end - time_seconds if kind == "start" else time_seconds - original_start
    )
    original_duration = original_end - original_start
    offset = time_seconds - (original_start if kind == "start" else original_end)
    current_token = _token(current_text)
    features = {
        "offset_seconds": offset,
        "absolute_offset_seconds": abs(offset),
        "original_duration_seconds": original_duration,
        "candidate_duration_seconds": candidate_duration,
        "absolute_duration_change_seconds": abs(candidate_duration - original_duration),
        "duration_fit": _duration_fit(candidate_duration),
        "exact_original_boundary": float(abs(offset) < 1e-6),
        "word_gap_before_seconds": min(gap_before, 5.0),
        "word_gap_after_seconds": min(gap_after, 5.0),
        "word_confidence": float(word.get("probability") or 0.0),
        "previous_token_ends_sentence": float(bool(SENTENCE_END.search(previous_text))),
        "current_token_ends_sentence": float(bool(SENTENCE_END.search(current_text))),
        "context_dependent_opening": float(
            kind == "start" and current_token in CONTEXT_DEPENDENT_OPENINGS
        ),
        "dangling_ending": float(kind == "end" and current_token in DANGLING_ENDINGS),
        "intro_or_ad_context": float(any(phrase in context for phrase in INTRO_OR_AD_PHRASES)),
        "position_in_video": time_seconds / max(media_duration, 1.0),
        **boundary_audio_features(audio, time_seconds),
    }
    if set(features) != set(BOUNDARY_FEATURE_FIELDS):
        raise RuntimeError("Boundary features do not match the versioned schema")
    return features


def build_boundary_query(
    queued: dict[str, Any],
    review: dict[str, Any],
    transcript: dict[str, Any],
    kind: str,
    audio: np.ndarray | None = None,
    search_radius: float = SEARCH_RADIUS_SECONDS,
) -> dict[str, Any]:
    """Build the inference-time choices and supervised target for one boundary."""
    if kind not in {"start", "end"}:
        raise ValueError("kind must be start or end")
    if queued["annotation_id"] != review["annotation_id"]:
        raise ValueError("queue and review annotation IDs differ")
    words = _words(transcript)
    if not words:
        raise ValueError(f"{queued['annotation_id']}: transcript has no timestamped words")
    original_start = float(queued["start_seconds"])
    original_end = float(queued["end_seconds"])
    original_time = original_start if kind == "start" else original_end
    boundary_edit = review.get("boundary_edit")
    target_time = float(boundary_edit[f"{kind}_seconds"]) if boundary_edit else original_time
    field = "start" if kind == "start" else "end"
    option_times = [original_time]
    option_times.extend(
        float(word[field])
        for word in words
        if abs(float(word[field]) - original_time) <= search_radius
    )
    option_times = sorted({round(value, 3) for value in option_times})
    media_duration = float(transcript.get("media_duration_seconds") or words[-1]["end"])
    options = []
    for time_seconds in option_times:
        word_index = _nearest_word_index(words, time_seconds, kind)
        options.append(
            {
                "time_seconds": time_seconds,
                "target_distance_seconds": abs(time_seconds - target_time),
                "features": _candidate_features(
                    words,
                    word_index,
                    kind,
                    time_seconds,
                    original_start,
                    original_end,
                    media_duration,
                    audio,
                ),
            }
        )
    return {
        "query_id": f"{queued['annotation_id']}__{kind}",
        "annotation_id": queued["annotation_id"],
        "video_id": queued["video_id"],
        "kind": kind,
        "edited": boundary_edit is not None,
        "materially_adjusted": abs(target_time - original_time) > MATERIAL_ADJUSTMENT_SECONDS,
        "original_time_seconds": original_time,
        "target_time_seconds": target_time,
        "target_adjustment_seconds": target_time - original_time,
        "options": options,
    }


def build_boundary_artifact(
    queue: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    transcripts_dir: Path,
    media_dir: Path | None = None,
) -> dict[str, Any]:
    """Create start/end choice sets, decoding each video's audio at most once."""
    review_by_id = {review["annotation_id"]: review for review in reviews}
    if len(review_by_id) != len(reviews):
        raise ValueError("reviews contain duplicate annotation IDs")
    if {row["annotation_id"] for row in queue} != set(review_by_id):
        raise ValueError("queue and reviews must contain exactly the same annotation IDs")

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for queued in queue:
        by_video[queued["video_id"]].append(queued)
    queries: list[dict[str, Any]] = []
    for video_index, (video_id, rows) in enumerate(sorted(by_video.items()), start=1):
        transcript_path = transcripts_dir / f"{video_id}.json"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        audio = None
        if media_dir is not None:
            from faster_whisper.audio import decode_audio

            media_path = media_dir / f"{video_id}.mp4"
            if not media_path.is_file():
                raise FileNotFoundError(media_path)
            audio = decode_audio(str(media_path), sampling_rate=16_000)
        for queued in rows:
            review = review_by_id[queued["annotation_id"]]
            for kind in ("start", "end"):
                queries.append(build_boundary_query(queued, review, transcript, kind, audio=audio))
        print(
            f"[{video_index}/{len(by_video)}] {video_id}: built {len(rows) * 2} queries",
            flush=True,
        )
    return {
        "metadata": {
            "schema": BOUNDARY_DATA_SCHEMA,
            "feature_fields": list(BOUNDARY_FEATURE_FIELDS),
            "search_radius_seconds": SEARCH_RADIUS_SECONDS,
            "audio_features_included": media_dir is not None,
            "label_definition": (
                "edited start/end when a boundary_edit exists; otherwise the accepted original"
            ),
        },
        "queries": queries,
    }


def _fit_choice_ranker(
    queries: list[dict[str, Any]], l2: float, seed: int = 42
) -> dict[str, Any]:
    if not queries:
        raise ValueError("training queries are empty")
    if l2 <= 0:
        raise ValueError("l2 must be positive")
    feature_rows = np.asarray(
        [
            [option["features"][field] for field in BOUNDARY_FEATURE_FIELDS]
            for query in queries
            for option in query["options"]
        ],
        dtype=float,
    )
    means = feature_rows.mean(axis=0)
    scales = feature_rows.std(axis=0)
    scales[scales < 1e-12] = 1.0
    differences: list[np.ndarray] = []
    generator = random.Random(seed)
    for query in queries:
        rows = np.asarray(
            [
                [option["features"][field] for field in BOUNDARY_FEATURE_FIELDS]
                for option in query["options"]
            ],
            dtype=float,
        )
        rows = (rows - means) / scales
        distances = np.asarray(
            [float(option["target_distance_seconds"]) for option in query["options"]]
        )
        best = int(np.argmin(distances))
        alternatives = [
            index for index, value in enumerate(distances) if value > distances[best] + 0.10
        ]
        if len(alternatives) > MAXIMUM_PAIRS_PER_QUERY:
            alternatives = sorted(generator.sample(alternatives, MAXIMUM_PAIRS_PER_QUERY))
        differences.extend(rows[best] - rows[index] for index in alternatives)
    if not differences:
        raise ValueError("training queries contain no boundary preferences")
    matrix = np.asarray(differences, dtype=float)
    weights = np.zeros(matrix.shape[1], dtype=float)
    identity = np.eye(matrix.shape[1])
    converged = False
    completed_iterations = 0
    for iteration in range(1, 101):
        completed_iterations = iteration
        margins = matrix @ weights
        mistakes = 1.0 / (1.0 + np.exp(np.clip(margins, -40.0, 40.0)))
        gradient = -(matrix.T @ mistakes) / len(matrix) + l2 * weights
        curvature = mistakes * (1.0 - mistakes)
        hessian = (matrix.T * curvature) @ matrix / len(matrix) + l2 * identity
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.linalg.norm(step)) <= 1e-8 * (1.0 + float(np.linalg.norm(weights))):
            converged = True
            break
    return {
        "feature_means": means,
        "feature_scales": scales,
        "weights": weights,
        "l2": l2,
        "training_preferences": len(matrix),
        "iterations": completed_iterations,
        "converged": converged,
    }


def _score_options(query: dict[str, Any], model: dict[str, Any]) -> np.ndarray:
    rows = np.asarray(
        [
            [option["features"][field] for field in BOUNDARY_FEATURE_FIELDS]
            for option in query["options"]
        ],
        dtype=float,
    )
    standardized = (rows - model["feature_means"]) / model["feature_scales"]
    return standardized @ model["weights"]


def _original_option_features(query: dict[str, Any]) -> np.ndarray:
    option = min(
        query["options"],
        key=lambda value: abs(
            float(value["time_seconds"]) - float(query["original_time_seconds"])
        ),
    )
    return np.asarray([option["features"][field] for field in BOUNDARY_FEATURE_FIELDS])


def _fit_adjustment_gate(queries: list[dict[str, Any]], l2: float) -> dict[str, Any]:
    """Fit a class-balanced logistic gate for whether a boundary needs movement."""
    features = np.asarray([_original_option_features(query) for query in queries], dtype=float)
    targets = np.asarray([float(query["materially_adjusted"]) for query in queries])
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if not positives or not negatives:
        raise ValueError("adjustment-gate training requires both classes")
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = (features - means) / scales
    design = np.column_stack([np.ones(len(standardized)), standardized])
    sample_weights = np.where(
        targets == 1.0,
        len(targets) / (2.0 * positives),
        len(targets) / (2.0 * negatives),
    )
    weights = np.zeros(design.shape[1], dtype=float)
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0.0
    completed_iterations = 0
    for iteration in range(1, 101):
        completed_iterations = iteration
        margins = np.clip(design @ weights, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-margins))
        gradient = design.T @ (sample_weights * (probabilities - targets)) / len(targets)
        gradient += penalty @ weights
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design / len(targets) + penalty
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.linalg.norm(step)) <= 1e-8 * (1.0 + float(np.linalg.norm(weights))):
            break
    return {
        "feature_means": means,
        "feature_scales": scales,
        "intercept": float(weights[0]),
        "weights": weights[1:],
        "l2": l2,
        "training_positive": positives,
        "training_negative": negatives,
        "iterations": completed_iterations,
    }


def _predict_adjustment_probability(query: dict[str, Any], model: dict[str, Any]) -> float:
    features = _original_option_features(query)
    standardized = (features - model["feature_means"]) / model["feature_scales"]
    margin = float(model["intercept"] + standardized @ model["weights"])
    return 1.0 / (1.0 + math.exp(-min(max(margin, -40.0), 40.0)))


def _heuristic_scores(query: dict[str, Any]) -> list[float]:
    scores = []
    for option in query["options"]:
        features = option["features"]
        pause = (
            features["word_gap_before_seconds"]
            if query["kind"] == "start"
            else features["word_gap_after_seconds"]
        )
        sentence = (
            features["previous_token_ends_sentence"]
            if query["kind"] == "start"
            else features["current_token_ends_sentence"]
        )
        penalty = (
            features["context_dependent_opening"]
            if query["kind"] == "start"
            else features["dangling_ending"]
        )
        scores.append(
            0.9 * min(pause / 1.5, 1.0)
            + 1.2 * sentence
            + 0.35 * features["audio_low_energy_ratio"]
            - 0.8 * penalty
            - 0.035 * features["absolute_offset_seconds"]
            - 1.2 * features["intro_or_ad_context"]
        )
    return scores


def _metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    errors = [abs(float(row[key]) - float(row["target_time_seconds"])) for row in rows]
    adjusted = [
        abs(float(row[key]) - float(row["target_time_seconds"]))
        for row in rows
        if row["materially_adjusted"]
    ]
    return {
        "mae_seconds": float(np.mean(errors)),
        "median_absolute_error_seconds": float(np.median(errors)),
        "within_1_second_rate": float(np.mean(np.asarray(errors) <= 1.0)),
        "within_2_seconds_rate": float(np.mean(np.asarray(errors) <= 2.0)),
        "adjusted_only_mae_seconds": float(np.mean(adjusted)) if adjusted else None,
    }


def _gate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual = [bool(row["materially_adjusted"]) for row in rows]
    probabilities = [float(row["adjustment_probability"]) for row in rows]
    predicted = [value >= 0.5 for value in probabilities]
    true_positive = sum(a and p for a, p in zip(actual, predicted, strict=True))
    false_positive = sum(not a and p for a, p in zip(actual, predicted, strict=True))
    false_negative = sum(a and not p for a, p in zip(actual, predicted, strict=True))
    positive_scores = [score for value, score in zip(actual, probabilities, strict=True) if value]
    negative_scores = [
        score for value, score in zip(actual, probabilities, strict=True) if not value
    ]
    auc_credit = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positive_scores
        for negative in negative_scores
    )
    return {
        "threshold": 0.5,
        "accuracy": sum(a == p for a, p in zip(actual, predicted, strict=True)) / len(rows),
        "precision": true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0,
        "recall": true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0,
        "roc_auc": auc_credit / (len(positive_scores) * len(negative_scores)),
    }


def cross_validate_boundary_ranker(
    artifact: dict[str, Any], n_splits: int = 4, seed: int = 42, l2: float = 0.3
) -> dict[str, Any]:
    """Evaluate start/end choice rankers while holding out complete source videos."""
    if artifact.get("metadata", {}).get("schema") != BOUNDARY_DATA_SCHEMA:
        raise ValueError(f"Expected {BOUNDARY_DATA_SCHEMA} artifact")
    queries = artifact.get("queries", [])
    video_ids = sorted({query["video_id"] for query in queries})
    if not 2 <= n_splits <= len(video_ids):
        raise ValueError("n_splits must be between 2 and the number of videos")
    random.Random(seed).shuffle(video_ids)
    fold_videos = [video_ids[index::n_splits] for index in range(n_splits)]
    predictions: list[dict[str, Any]] = []
    folds = []
    for fold_number, test_videos in enumerate(fold_videos, start=1):
        test_set = set(test_videos)
        fold_info = {"fold": fold_number, "test_videos": sorted(test_videos), "models": {}}
        for kind in ("start", "end"):
            train = [q for q in queries if q["kind"] == kind and q["video_id"] not in test_set]
            test = [q for q in queries if q["kind"] == kind and q["video_id"] in test_set]
            adjusted_train = [query for query in train if query["materially_adjusted"]]
            model = _fit_choice_ranker(adjusted_train, l2=l2, seed=seed + fold_number)
            gate = _fit_adjustment_gate(train, l2=l2)
            fold_info["models"][kind] = {
                "training_queries": len(train),
                "training_adjusted_queries": len(adjusted_train),
                "training_preferences": model["training_preferences"],
                "ranker_converged": model["converged"],
                "gate_positive": gate["training_positive"],
                "gate_negative": gate["training_negative"],
            }
            for query in test:
                learned_scores = _score_options(query, model)
                heuristic_scores = _heuristic_scores(query)
                selector = query["options"][int(np.argmax(learned_scores))]["time_seconds"]
                adjustment_probability = _predict_adjustment_probability(query, gate)
                learned = (
                    selector
                    if adjustment_probability >= 0.5
                    else query["original_time_seconds"]
                )
                heuristic = query["options"][int(np.argmax(heuristic_scores))]["time_seconds"]
                oracle = min(
                    query["options"], key=lambda option: option["target_distance_seconds"]
                )["time_seconds"]
                oracle_gate = (
                    selector if query["materially_adjusted"] else query["original_time_seconds"]
                )
                predictions.append(
                    {
                        "query_id": query["query_id"],
                        "annotation_id": query["annotation_id"],
                        "video_id": query["video_id"],
                        "kind": kind,
                        "fold": fold_number,
                        "edited": query["edited"],
                        "materially_adjusted": query["materially_adjusted"],
                        "adjustment_probability": adjustment_probability,
                        "target_time_seconds": query["target_time_seconds"],
                        "no_change_time_seconds": query["original_time_seconds"],
                        "heuristic_time_seconds": heuristic,
                        "selector_time_seconds": selector,
                        "oracle_gate_selector_time_seconds": oracle_gate,
                        "learned_time_seconds": learned,
                        "oracle_time_seconds": oracle,
                    }
                )
        folds.append(fold_info)

    final_models = {}
    for kind in ("start", "end"):
        kind_queries = [query for query in queries if query["kind"] == kind]
        model = _fit_choice_ranker(
            [query for query in kind_queries if query["materially_adjusted"]],
            l2=l2,
            seed=seed,
        )
        gate = _fit_adjustment_gate(kind_queries, l2=l2)
        final_models[kind] = {
            "model_schema": BOUNDARY_MODEL_SCHEMA,
            "feature_schema": BOUNDARY_DATA_SCHEMA,
            "feature_fields": list(BOUNDARY_FEATURE_FIELDS),
            "feature_means": model["feature_means"].tolist(),
            "feature_scales": model["feature_scales"].tolist(),
            "standardized_weights": model["weights"].tolist(),
            "l2": l2,
            "training_preferences": model["training_preferences"],
            "adjustment_gate": {
                "intercept": gate["intercept"],
                "standardized_weights": gate["weights"].tolist(),
                "training_positive": gate["training_positive"],
                "training_negative": gate["training_negative"],
            },
        }
    by_kind = {}
    for kind in ("start", "end"):
        rows = [row for row in predictions if row["kind"] == kind]
        by_kind[kind] = {
            "queries": len(rows),
            "edited_queries": sum(bool(row["edited"]) for row in rows),
            "materially_adjusted_queries": sum(
                bool(row["materially_adjusted"]) for row in rows
            ),
            "adjustment_gate": _gate_metrics(rows),
            "no_change": _metrics(rows, "no_change_time_seconds"),
            "fixed_heuristic": _metrics(rows, "heuristic_time_seconds"),
            "selector_with_oracle_gate": _metrics(
                rows, "oracle_gate_selector_time_seconds"
            ),
            "learned_ranker": _metrics(rows, "learned_time_seconds"),
            "candidate_oracle": _metrics(rows, "oracle_time_seconds"),
        }
    return {
        "experiment": "grouped_multimodal_boundary_choice_ranking_v1",
        "dataset": {
            "queries": len(queries),
            "clips": len({query["annotation_id"] for query in queries}),
            "videos": len(video_ids),
            "edited_clips": len(
                {query["annotation_id"] for query in queries if query["edited"]}
            ),
            "candidate_options": sum(len(query["options"]) for query in queries),
        },
        "cross_validation": {
            "strategy": "deterministic grouped folds by source video",
            "n_splits": n_splits,
            "seed": seed,
            "folds": folds,
            "leakage_note": "No candidate from a test video appears in its training fold.",
        },
        "training": {
            "objective": (
                "class-balanced edit gate, then pairwise candidate ranking trained only on "
                "materially adjusted boundaries"
            ),
            "l2": l2,
            "maximum_pairs_per_query": MAXIMUM_PAIRS_PER_QUERY,
            "material_adjustment_seconds": MATERIAL_ADJUSTMENT_SECONDS,
        },
        "metrics": by_kind,
        "final_models": final_models,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CreatorCut boundary choice rankers")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--transcripts-dir", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l2", type=float, default=0.3)
    args = parser.parse_args()

    artifact = build_boundary_artifact(
        load_jsonl(args.queue),
        load_jsonl(args.reviews),
        args.transcripts_dir,
        media_dir=args.media_dir,
    )
    result = cross_validate_boundary_ranker(
        artifact, n_splits=args.folds, seed=args.seed, l2=args.l2
    )
    predictions = result.pop("predictions")
    models = result.pop("final_models")
    for path, value in (
        (args.artifact, artifact),
        (args.evaluation, {**result, "predictions": predictions}),
        (args.model, {"schema": BOUNDARY_MODEL_SCHEMA, "models": models}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
