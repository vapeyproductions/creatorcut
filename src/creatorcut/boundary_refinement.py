"""Propose and evaluate transcript- and pause-aligned clip boundaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from creatorcut.candidates import SENTENCE_END, SentenceUnit, words_to_sentence_units, write_jsonl
from creatorcut.dataset import load_jsonl, load_manifest

SAMPLE_RATE = 16_000
CONTEXT_DEPENDENT_OPENINGS = {
    "also",
    "and",
    "because",
    "but",
    "he",
    "however",
    "it",
    "she",
    "so",
    "that",
    "then",
    "they",
    "this",
    "those",
    "we",
    "which",
    "while",
    "who",
}
INTRO_OR_AD_PHRASES = (
    "before we get into",
    "brought to you by",
    "don't forget to subscribe",
    "in today's episode",
    "promo code",
    "sponsor",
    "welcome back",
)
TOKEN = re.compile(r"[A-Za-z']+")


def _first_token(text: str) -> str:
    match = TOKEN.search(text.lower())
    return match.group(0) if match else ""


def _last_token(text: str) -> str:
    matches = TOKEN.findall(text.lower())
    return matches[-1] if matches else ""


def _contains_intro_or_ad(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in INTRO_OR_AD_PHRASES)


def _word_gap_before(units: list[SentenceUnit], index: int) -> float:
    return max(0.0, units[index].start - units[index - 1].end) if index else 0.0


def _word_gap_after(units: list[SentenceUnit], index: int) -> float:
    if index + 1 >= len(units):
        return 0.0
    return max(0.0, units[index + 1].start - units[index].end)


def _vad_gap_before(time_seconds: float, speech_intervals: list[dict[str, float]]) -> float:
    previous_ends = [
        interval["end_seconds"]
        for interval in speech_intervals
        if interval["end_seconds"] <= time_seconds + 0.05
    ]
    return max(0.0, time_seconds - max(previous_ends)) if previous_ends else 0.0


def _vad_gap_after(time_seconds: float, speech_intervals: list[dict[str, float]]) -> float:
    next_starts = [
        interval["start_seconds"]
        for interval in speech_intervals
        if interval["start_seconds"] >= time_seconds - 0.05
    ]
    return max(0.0, min(next_starts) - time_seconds) if next_starts else 0.0


def _bounded_pause(value: float) -> float:
    return min(max(value, 0.0) / 1.5, 1.0)


def _start_option(
    units: list[SentenceUnit],
    index: int,
    original_start: float,
    search_radius: float,
    speech_intervals: list[dict[str, float]],
) -> dict[str, Any]:
    unit = units[index]
    first_word = _first_token(unit.text)
    clean_opening = bool(first_word) and first_word not in CONTEXT_DEPENDENT_OPENINGS
    previous_complete = index == 0 or bool(SENTENCE_END.search(units[index - 1].text.strip()))
    pause_seconds = max(
        _word_gap_before(units, index),
        _vad_gap_before(unit.start, speech_intervals),
    )
    distance_ratio = min(abs(unit.start - original_start) / search_radius, 1.0)
    intro_or_ad = _contains_intro_or_ad(unit.text)
    score = (
        1.2 * float(clean_opening)
        + 0.7 * _bounded_pause(pause_seconds)
        + 0.6 * float(previous_complete)
        - 0.55 * distance_ratio
        - 1.5 * float(intro_or_ad)
    )
    return {
        "time_seconds": unit.start,
        "score": score,
        "clean_opening": clean_opening,
        "pause_seconds": pause_seconds,
        "previous_unit_complete": previous_complete,
        "intro_or_ad_phrase": intro_or_ad,
        "unit_text": unit.text,
    }


def _end_option(
    units: list[SentenceUnit],
    index: int,
    original_end: float,
    search_radius: float,
    speech_intervals: list[dict[str, float]],
) -> dict[str, Any]:
    unit = units[index]
    complete_sentence = bool(SENTENCE_END.search(unit.text.strip()))
    dangling_ending = _last_token(unit.text) in {
        "and",
        "because",
        "but",
        "or",
        "so",
        "that",
        "then",
        "which",
        "while",
    }
    pause_seconds = max(
        _word_gap_after(units, index),
        _vad_gap_after(unit.end, speech_intervals),
    )
    distance_ratio = min(abs(unit.end - original_end) / search_radius, 1.0)
    score = (
        1.4 * float(complete_sentence)
        + 0.8 * _bounded_pause(pause_seconds)
        - 0.7 * float(dangling_ending)
        - 0.55 * distance_ratio
    )
    return {
        "time_seconds": unit.end,
        "score": score,
        "complete_sentence": complete_sentence,
        "pause_seconds": pause_seconds,
        "dangling_ending": dangling_ending,
        "unit_text": unit.text,
    }


def propose_boundary_refinement(
    case: dict[str, Any],
    window: dict[str, Any],
    search_radius: float = 30.0,
    minimum_duration: float = 8.0,
    maximum_duration: float = 90.0,
) -> dict[str, Any]:
    """Choose a nearby pair of transcript and neural-VAD boundaries without using labels."""
    if search_radius <= 0:
        raise ValueError("search_radius must be positive")
    selected = case["model_selected"]
    original_start = float(selected["start_seconds"])
    original_end = float(selected["end_seconds"])
    words = sorted(window.get("words", []), key=lambda word: (word["start"], word["end"]))
    units = words_to_sentence_units(words, pause_split_seconds=0.65, maximum_unit_seconds=20.0)
    if not units:
        raise ValueError(f"{case['analysis_id']}: boundary window has no timestamped words")
    speech_intervals = window.get("speech_intervals", [])

    start_options = [
        _start_option(units, index, original_start, search_radius, speech_intervals)
        for index in range(len(units))
        if abs(units[index].start - original_start) <= search_radius
    ]
    end_options = [
        _end_option(units, index, original_end, search_radius, speech_intervals)
        for index in range(len(units))
        if abs(units[index].end - original_end) <= search_radius
    ]
    original_duration = original_end - original_start
    pair_options: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
    for start in start_options:
        for end in end_options:
            duration = end["time_seconds"] - start["time_seconds"]
            overlap = max(
                0.0,
                min(end["time_seconds"], original_end)
                - max(start["time_seconds"], original_start),
            )
            if not minimum_duration <= duration <= maximum_duration:
                continue
            if overlap < min(5.0, original_duration * 0.25):
                continue
            duration_change = abs(duration - original_duration) / max(original_duration, 1.0)
            score = start["score"] + end["score"] - 0.35 * duration_change
            movement = abs(start["time_seconds"] - original_start) + abs(
                end["time_seconds"] - original_end
            )
            pair_options.append((score, movement, start, end))
    if not pair_options:
        raise ValueError(f"{case['analysis_id']}: no valid refined boundary pair")

    score, _, start, end = max(pair_options, key=lambda item: (item[0], -item[1]))
    refined_start = float(start["time_seconds"])
    refined_end = float(end["time_seconds"])
    return {
        "analysis_id": case["analysis_id"],
        "video_id": case["video_id"],
        "method": "whisper_sentence_units_plus_silero_vad_v1",
        "original_start_seconds": original_start,
        "original_end_seconds": original_end,
        "refined_start_seconds": refined_start,
        "refined_end_seconds": refined_end,
        "start_adjustment_seconds": refined_start - original_start,
        "end_adjustment_seconds": refined_end - original_end,
        "original_duration_seconds": original_duration,
        "refined_duration_seconds": refined_end - refined_start,
        "boundary_score": score,
        "start_evidence": start,
        "end_evidence": end,
    }


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def evaluate_boundary_refinements(
    proposals: list[dict[str, Any]], reviews: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare frozen proposals with human deltas; review labels never affect proposals."""
    review_by_id = {review["analysis_id"]: review for review in reviews}
    if len(review_by_id) != len(reviews):
        raise ValueError("Failure reviews contain duplicate analysis IDs")
    examples: list[dict[str, Any]] = []
    errors: dict[str, list[float]] = {"start": [], "end": []}
    baselines: dict[str, list[float]] = {"start": [], "end": []}
    within_two: dict[str, list[bool]] = {"start": [], "end": []}

    for proposal in proposals:
        review = review_by_id.get(proposal["analysis_id"])
        if review is None:
            continue
        comparison: dict[str, Any] = {
            "analysis_id": proposal["analysis_id"],
            "video_id": proposal["video_id"],
        }
        for boundary in ("start", "end"):
            human_delta = review.get(f"{boundary}_adjustment_seconds")
            if human_delta is None:
                continue
            proposed_delta = float(proposal[f"{boundary}_adjustment_seconds"])
            error = abs(proposed_delta - float(human_delta))
            baseline_error = abs(float(human_delta))
            errors[boundary].append(error)
            baselines[boundary].append(baseline_error)
            within_two[boundary].append(error <= 2.0)
            comparison[boundary] = {
                "human_adjustment_seconds": float(human_delta),
                "proposed_adjustment_seconds": proposed_delta,
                "absolute_error_seconds": error,
                "no_change_absolute_error_seconds": baseline_error,
            }
        if len(comparison) > 2:
            examples.append(comparison)

    metrics = {}
    for boundary in ("start", "end"):
        model_mae = _mean_or_none(errors[boundary])
        baseline_mae = _mean_or_none(baselines[boundary])
        metrics[boundary] = {
            "labeled_count": len(errors[boundary]),
            "mean_absolute_error_seconds": model_mae,
            "no_change_mean_absolute_error_seconds": baseline_mae,
            "mae_improvement_seconds": (
                baseline_mae - model_mae
                if model_mae is not None and baseline_mae is not None
                else None
            ),
            "within_2_seconds_rate": (
                sum(within_two[boundary]) / len(within_two[boundary])
                if within_two[boundary]
                else None
            ),
        }
    return {
        "method": "whisper_sentence_units_plus_silero_vad_v1",
        "proposal_count": len(proposals),
        "evaluation_examples": len(examples),
        "metrics": metrics,
        "examples": examples,
        "interpretation": (
            "This is a fixed, untrained baseline. Human corrections are used only after proposal "
            "generation, so the evaluation does not tune on its five start labels and one end "
            "label."
        ),
    }


def transcribe_boundary_windows(
    cases: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    media_dir: Path,
    model_name: str,
    compute_type: str,
    model_cache: Path,
    context_seconds: float = 35.0,
) -> list[dict[str, Any]]:
    """Transcribe only local windows needed for boundary refinement and add neural VAD spans."""
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    if context_seconds <= 0:
        raise ValueError("context_seconds must be positive")
    paths = {
        item["video_id"]: media_dir / Path(item["local_filename"]).name for item in manifest
    }
    model_cache.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type=compute_type,
        download_root=str(model_cache),
    )
    windows: list[dict[str, Any]] = []
    vad_options = VadOptions(
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
        speech_pad_ms=0,
    )

    for case in cases:
        path = paths.get(case["video_id"])
        if path is None or not path.is_file():
            raise FileNotFoundError(path or case["video_id"])
        selected = case["model_selected"]
        window_start = max(0.0, float(selected["start_seconds"]) - context_seconds)
        audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
        media_duration = len(audio) / SAMPLE_RATE
        window_end = min(media_duration, float(selected["end_seconds"]) + context_seconds)
        start_sample = round(window_start * SAMPLE_RATE)
        end_sample = round(window_end * SAMPLE_RATE)
        window_audio = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
        segments, _ = model.transcribe(
            window_audio,
            language="en",
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        words = [
            {
                "start": float(word.start) + window_start,
                "end": float(word.end) + window_start,
                "word": word.word,
                "probability": float(word.probability),
            }
            for segment in segments
            for word in (segment.words or [])
            if word.start is not None and word.end is not None
        ]
        speech_intervals = [
            {
                "start_seconds": value["start"] / SAMPLE_RATE + window_start,
                "end_seconds": value["end"] / SAMPLE_RATE + window_start,
            }
            for value in get_speech_timestamps(
                window_audio,
                vad_options=vad_options,
                sampling_rate=SAMPLE_RATE,
            )
        ]
        windows.append(
            {
                "analysis_id": case["analysis_id"],
                "video_id": case["video_id"],
                "window_start_seconds": window_start,
                "window_end_seconds": window_end,
                "model_name": model_name,
                "words": words,
                "speech_intervals": speech_intervals,
            }
        )
        print(
            f"{case['analysis_id']}: {window_end - window_start:.1f}s window, "
            f"{len(words)} words, {len(speech_intervals)} speech spans",
            flush=True,
        )
    return windows


def main() -> None:
    """Transcribe review windows, freeze boundary proposals, then evaluate human corrections."""
    parser = argparse.ArgumentParser(description="Evaluate CreatorCut boundary refinement")
    parser.add_argument(
        "--cases", type=Path, default=Path("data/processed/failure_cases.jsonl")
    )
    parser.add_argument(
        "--reviews", type=Path, default=Path("data/processed/failure_reviews.jsonl")
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/videos.json"))
    parser.add_argument("--media-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--windows", type=Path, default=Path("data/processed/boundary_windows.jsonl")
    )
    parser.add_argument(
        "--proposals", type=Path, default=Path("data/processed/boundary_proposals.jsonl")
    )
    parser.add_argument(
        "--evaluation", type=Path, default=Path("data/processed/boundary_evaluation.json")
    )
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--model-cache", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--context-seconds", type=float, default=35.0)
    parser.add_argument("--force-transcription", action="store_true")
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    if args.windows.exists() and not args.force_transcription:
        windows = load_jsonl(args.windows)
        print(f"Reusing {len(windows)} cached boundary windows from {args.windows}")
    else:
        windows = transcribe_boundary_windows(
            cases,
            load_manifest(args.manifest),
            args.media_dir,
            args.model,
            args.compute_type,
            args.model_cache,
            context_seconds=args.context_seconds,
        )
        write_jsonl(windows, args.windows)
        print(f"Wrote {len(windows)} private boundary windows to {args.windows}")

    window_by_id = {window["analysis_id"]: window for window in windows}
    if set(window_by_id) != {case["analysis_id"] for case in cases}:
        raise ValueError("Boundary windows do not match failure cases")
    proposals = [
        propose_boundary_refinement(case, window_by_id[case["analysis_id"]])
        for case in cases
    ]
    write_jsonl(proposals, args.proposals)
    evaluation = evaluate_boundary_refinements(proposals, load_jsonl(args.reviews))
    args.evaluation.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(json.dumps(evaluation["metrics"], indent=2))
    print(f"Wrote {len(proposals)} private proposals to {args.proposals}")
    print(f"Wrote aggregate evaluation to {args.evaluation}")


if __name__ == "__main__":
    main()
