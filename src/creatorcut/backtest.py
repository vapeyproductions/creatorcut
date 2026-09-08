"""Leakage-safe historical recommendation tests and media alignment."""

from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from creatorcut.analytics import MAXIMUM_ANALYTICS_BYTES, _csv_files, _decode_csv
from creatorcut.candidates import interval_iou

BACKTEST_SCHEMA = "creatorcut_historical_backtest_v1"
ALIGNMENT_SCHEMA = "creatorcut_audio_alignment_v1"
EVALUATION_SCHEMA = "creatorcut_backtest_evaluation_v1"
MAXIMUM_BACKTEST_SHORTS = 30
ALIGNMENT_RATE_HZ = 50
ALIGNMENT_SAMPLE_RATE = 8_000
MINIMUM_ALIGNMENT_SCORE = 0.55


def _compact_number(value: Any) -> int | None:
    """Parse ordinary counts plus the compact K/M notation in public trackers."""
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kKmM]?)", text)
    if match is None:
        raise ValueError(f"Could not read analytics count: {text}")
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000}[match.group(2).casefold()]
    return int(round(float(match.group(1)) * multiplier))


def parse_backtest_tracker(
    filename: str,
    payload: bytes,
    reference_keys: list[str],
    holdout_key: str,
) -> dict[str, Any]:
    """Parse a compact video/Short tracker and partition reference from holdout rows."""
    if not payload:
        raise ValueError("The stats tracker is empty")
    if len(payload) > MAXIMUM_ANALYTICS_BYTES:
        raise ValueError("The stats tracker must be 20 MB or smaller")
    keys = [str(value).strip() for value in reference_keys]
    holdout = str(holdout_key).strip()
    if len(keys) != 2 or len(set(keys)) != 2 or not all(keys):
        raise ValueError("Enter two different reference video IDs")
    if not holdout or holdout in keys:
        raise ValueError("The test video ID must differ from the reference video IDs")

    normalized_rows: list[dict[str, Any]] = []
    for csv_name, csv_bytes in _csv_files(filename, payload):
        text = _decode_csv(csv_bytes)
        delimiter = "\t" if Path(csv_name).suffix.casefold() == ".tsv" else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        if not reader.fieldnames:
            continue
        headers = {
            re.sub(r"[^a-z0-9]+", " ", header.casefold()).strip(): header
            for header in reader.fieldnames
            if header
        }
        id_header = headers.get("video id") or headers.get("video") or headers.get("id")
        if id_header is None:
            continue
        for source in reader:
            clip_key = str(source.get(id_header) or "").strip()
            if not clip_key:
                continue
            link_header = headers.get("link") or headers.get("url")
            normalized_rows.append(
                {
                    "content_id": clip_key,
                    "url": str(source.get(link_header) or "").strip()
                    if link_header
                    else "",
                    "views": _compact_number(source.get(headers.get("views", ""))),
                    "likes": _compact_number(source.get(headers.get("likes", ""))),
                    "comments": _compact_number(source.get(headers.get("comments", ""))),
                }
            )
    if not normalized_rows:
        raise ValueError("The tracker needs Link, Views, Likes, Comments, and video ID columns")

    all_keys = [*keys, holdout]
    source_rows: dict[str, dict[str, Any]] = {}
    clip_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in normalized_rows:
        content_id = row["content_id"]
        if content_id in seen_ids:
            raise ValueError(f"The tracker contains duplicate video ID {content_id}")
        seen_ids.add(content_id)
        if content_id in all_keys:
            source_rows[content_id] = row
            continue
        matches = [key for key in all_keys if content_id.startswith(f"{key}_clip")]
        if len(matches) == 1:
            source_key = matches[0]
            if row["views"] is None:
                raise ValueError(f"{content_id} is missing its view count")
            clip_rows.append(
                {
                    **row,
                    "source_key": source_key,
                    "role": "holdout" if source_key == holdout else "reference",
                }
            )

    missing_sources = [key for key in all_keys if key not in source_rows]
    if missing_sources:
        raise ValueError(
            "The tracker is missing long-video rows for " + ", ".join(missing_sources)
        )
    counts = {key: sum(row["source_key"] == key for row in clip_rows) for key in all_keys}
    missing_clips = [key for key, count in counts.items() if count == 0]
    if missing_clips:
        raise ValueError(
            "The tracker is missing corresponding Shorts for " + ", ".join(missing_clips)
        )
    if len(clip_rows) > MAXIMUM_BACKTEST_SHORTS:
        raise ValueError(f"A historical test can contain at most {MAXIMUM_BACKTEST_SHORTS} Shorts")
    return {
        "schema": BACKTEST_SCHEMA,
        "reference_keys": keys,
        "holdout_key": holdout,
        "source_rows": source_rows,
        "clip_rows": sorted(clip_rows, key=lambda row: row["content_id"]),
        "clip_counts": counts,
    }


def audio_envelope(path: Path) -> np.ndarray:
    """Decode a compact, re-encoding-resistant log-energy fingerprint."""
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(path), sampling_rate=ALIGNMENT_SAMPLE_RATE)
    hop = ALIGNMENT_SAMPLE_RATE // ALIGNMENT_RATE_HZ
    usable = len(audio) // hop * hop
    if usable < hop:
        raise ValueError(f"{path.name} does not contain usable audio")
    frames = np.asarray(audio[:usable], dtype=np.float32).reshape(-1, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return np.log(rms + 1e-6)


def _match_envelope(long_audio: np.ndarray, template: np.ndarray) -> tuple[int, float, float]:
    """Return the best normalized FFT correlation and a separated competing peak."""
    if len(template) >= len(long_audio):
        raise ValueError("The Short audio must be shorter than its source video")
    centered = template.astype(np.float64) - float(np.mean(template))
    template_norm = float(np.linalg.norm(centered))
    if template_norm <= 1e-9:
        raise ValueError("The Short audio does not contain enough variation to align")
    output_length = len(long_audio) + len(centered) - 1
    fft_length = 1 << (output_length - 1).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(long_audio, fft_length)
        * np.fft.rfft(centered[::-1], fft_length),
        fft_length,
    )[len(centered) - 1 : len(long_audio)]
    cumulative = np.concatenate(([0.0], np.cumsum(long_audio, dtype=np.float64)))
    cumulative_squared = np.concatenate(
        ([0.0], np.cumsum(long_audio.astype(np.float64) ** 2))
    )
    window_sum = cumulative[len(centered) :] - cumulative[: -len(centered)]
    window_squared = (
        cumulative_squared[len(centered) :] - cumulative_squared[: -len(centered)]
    )
    variance = np.maximum(
        window_squared - window_sum * window_sum / len(centered), 1e-12
    )
    scores = correlation / (np.sqrt(variance) * template_norm)
    best_index = int(np.argmax(scores))
    separated = scores.copy()
    radius = max(ALIGNMENT_RATE_HZ * 5, len(centered) // 2)
    separated[max(0, best_index - radius) : best_index + radius + 1] = -1.0
    competing_score = float(np.max(separated)) if len(separated) else -1.0
    return best_index, float(scores[best_index]), competing_score


def align_short_to_source(
    source_path: Path,
    short_path: Path,
    source_envelope: np.ndarray | None = None,
) -> dict[str, Any]:
    """Map a published Short to one or more source intervals using audio fingerprints."""
    source = source_envelope if source_envelope is not None else audio_envelope(source_path)
    short = audio_envelope(short_path)
    short_duration = len(short) / ALIGNMENT_RATE_HZ
    window_seconds = min(15.0, max(8.0, short_duration / 3.0))
    window_frames = min(len(short), max(200, round(window_seconds * ALIGNMENT_RATE_HZ)))
    available = max(0, len(short) - window_frames)
    fractions = np.linspace(0.08, 0.92, min(7, max(3, math.ceil(short_duration / 12))))
    starts = sorted({round(available * float(fraction)) for fraction in fractions})
    matches: list[dict[str, float]] = []
    for short_start in starts:
        long_start, score, competing = _match_envelope(
            source, short[short_start : short_start + window_frames]
        )
        matches.append(
            {
                "short_start": short_start / ALIGNMENT_RATE_HZ,
                "short_end": (short_start + window_frames) / ALIGNMENT_RATE_HZ,
                "source_start": long_start / ALIGNMENT_RATE_HZ,
                "offset": (long_start - short_start) / ALIGNMENT_RATE_HZ,
                "score": score,
                "peak_margin": score - competing,
            }
        )

    reliable = [match for match in matches if match["score"] >= 0.48]
    clusters: list[list[dict[str, float]]] = []
    for match in sorted(reliable, key=lambda value: value["offset"]):
        nearest = next(
            (
                cluster
                for cluster in clusters
                if abs(float(np.median([item["offset"] for item in cluster])) - match["offset"])
                <= 3.0
            ),
            None,
        )
        if nearest is None:
            clusters.append([match])
        else:
            nearest.append(match)

    clusters.sort(
        key=lambda cluster: (
            -len(cluster),
            -float(np.mean([item["score"] for item in cluster])),
        )
    )
    segments: list[dict[str, float]] = []
    significant_clusters = [
        cluster for index, cluster in enumerate(clusters) if index == 0 or len(cluster) >= 2
    ]
    for cluster in significant_clusters:
        offset = float(np.median([item["offset"] for item in cluster]))
        short_start = min(item["short_start"] for item in cluster)
        short_end = max(item["short_end"] for item in cluster)
        segments.append(
            {
                "short_start_seconds": round(short_start, 3),
                "short_end_seconds": round(short_end, 3),
                "source_start_seconds": round(max(0.0, short_start + offset), 3),
                "source_end_seconds": round(
                    min(len(source) / ALIGNMENT_RATE_HZ, short_end + offset), 3
                ),
                "confidence": round(float(np.mean([item["score"] for item in cluster])), 4),
                "window_count": len(cluster),
            }
        )
    primary = segments[0] if segments else None
    coverage = (
        primary["window_count"] / len(starts) if primary is not None and starts else 0.0
    )
    confidence = (
        float(primary["confidence"]) * (0.65 + 0.35 * coverage)
        if primary is not None
        else 0.0
    )
    if primary is not None and primary["window_count"] >= 2:
        offset = float(
            np.median(
                [
                    match["offset"]
                    for match in reliable
                    if abs(
                        match["offset"]
                        - (
                            primary["source_start_seconds"]
                            - primary["short_start_seconds"]
                        )
                    )
                    <= 3.0
                ]
            )
        )
        start = max(0.0, offset)
        end = min(len(source) / ALIGNMENT_RATE_HZ, start + short_duration)
    elif primary is not None:
        start = primary["source_start_seconds"] - primary["short_start_seconds"]
        start = max(0.0, start)
        end = min(len(source) / ALIGNMENT_RATE_HZ, start + short_duration)
    else:
        start = end = None
    compound_edit = len(significant_clusters) >= 2
    return {
        "schema": ALIGNMENT_SCHEMA,
        "status": (
            "aligned"
            if confidence >= MINIMUM_ALIGNMENT_SCORE and not compound_edit
            else "needs_review"
        ),
        "source_start_seconds": round(start, 3) if start is not None else None,
        "source_end_seconds": round(end, 3) if end is not None else None,
        "short_duration_seconds": round(short_duration, 3),
        "confidence": round(confidence, 4),
        "method": "audio_log_energy_fft",
        "compound_edit_detected": compound_edit,
        "segments": segments,
        "matched_window_count": len(reliable),
        "window_count": len(starts),
    }


def _centered_percentile(value: float, values: list[float]) -> float:
    if len(values) < 2 or max(values) == min(values):
        return 0.5
    below = sum(candidate < value for candidate in values)
    equal = sum(candidate == value for candidate in values)
    return (below + (equal - 1) / 2) / (len(values) - 1)


def actual_performance_scores(actuals: list[dict[str, Any]]) -> dict[str, float]:
    """Build a within-test outcome index while retaining raw metrics for inspection."""
    views = [float(item["views"]) for item in actuals if item.get("views") is not None]
    like_rates = [
        float(item.get("likes") or 0) / float(item["views"])
        for item in actuals
        if item.get("views")
    ]
    comment_rates = [
        float(item.get("comments") or 0) / float(item["views"])
        for item in actuals
        if item.get("views")
    ]
    result = {}
    for item in actuals:
        if not item.get("views"):
            continue
        view_score = _centered_percentile(float(item["views"]), views)
        like_score = _centered_percentile(
            float(item.get("likes") or 0) / float(item["views"]), like_rates
        )
        comment_score = _centered_percentile(
            float(item.get("comments") or 0) / float(item["views"]), comment_rates
        )
        result[item["content_id"]] = round(
            100.0 * (0.5 * view_score + 0.3 * like_score + 0.2 * comment_score), 2
        )
    return result


def _rank_correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    def ranks(values: list[float]) -> np.ndarray:
        ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
        result = np.empty(len(values), dtype=float)
        position = 0
        while position < len(ordered):
            end = position + 1
            while end < len(ordered) and values[ordered[end]] == values[ordered[position]]:
                end += 1
            average = (position + end - 1) / 2.0
            for index in ordered[position:end]:
                result[index] = average
            position = end
        return result

    predicted = ranks([pair[0] for pair in pairs])
    actual = ranks([pair[1] for pair in pairs])
    if float(np.std(predicted)) <= 1e-9 or float(np.std(actual)) <= 1e-9:
        return None
    return float(np.corrcoef(predicted, actual)[0, 1])


def evaluate_backtest(
    predictions: list[dict[str, Any]], actuals: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare frozen recommendations with later-revealed organization choices."""
    usable_actuals = [
        item
        for item in actuals
        if item.get("source_start_seconds") is not None
        and item.get("source_end_seconds") is not None
    ]
    outcomes = actual_performance_scores(usable_actuals)
    ordered_predictions = sorted(predictions, key=lambda item: item["rank"])
    pair_scores = []
    for prediction_index, prediction in enumerate(ordered_predictions):
        for actual_index, actual in enumerate(usable_actuals):
            segments = actual.get("segments") or [
                {
                    "source_start_seconds": actual["source_start_seconds"],
                    "source_end_seconds": actual["source_end_seconds"],
                }
            ]
            iou = max(
                interval_iou(
                    (prediction["start_seconds"], prediction["end_seconds"]),
                    (segment["source_start_seconds"], segment["source_end_seconds"]),
                )
                for segment in segments
            )
            pair_scores.append((iou, prediction_index, actual_index))
    assignments: dict[int, tuple[int, float]] = {}
    assigned_actuals: set[int] = set()
    for iou, prediction_index, actual_index in sorted(pair_scores, reverse=True):
        if iou < 0.10:
            break
        if prediction_index in assignments or actual_index in assigned_actuals:
            continue
        assignments[prediction_index] = (actual_index, iou)
        assigned_actuals.add(actual_index)

    matches = []
    for prediction_index, prediction in enumerate(ordered_predictions):
        assigned = assignments.get(prediction_index)
        actual = usable_actuals[assigned[0]] if assigned is not None else None
        best_iou = assigned[1] if assigned is not None else 0.0
        matches.append(
            {
                "prediction_rank": prediction["rank"],
                "prediction_clip_id": prediction["clip_id"],
                "prediction_start_seconds": prediction["start_seconds"],
                "prediction_end_seconds": prediction["end_seconds"],
                "estimated_relative_performance": prediction[
                    "estimated_relative_performance"
                ],
                "actual_content_id": actual["content_id"] if actual else None,
                "actual_start_seconds": actual["source_start_seconds"] if actual else None,
                "actual_end_seconds": actual["source_end_seconds"] if actual else None,
                "temporal_iou": round(best_iou, 4) if actual else 0.0,
                "recovered": bool(actual and best_iou >= 0.50),
                "start_error_seconds": round(
                    prediction["start_seconds"] - actual["source_start_seconds"], 3
                )
                if actual
                else None,
                "end_error_seconds": round(
                    prediction["end_seconds"] - actual["source_end_seconds"], 3
                )
                if actual
                else None,
                "actual_performance_score": outcomes.get(actual["content_id"])
                if actual
                else None,
                "views": actual.get("views") if actual else None,
                "likes": actual.get("likes") if actual else None,
                "comments": actual.get("comments") if actual else None,
            }
        )
    recovered_actual_ids = {
        match["actual_content_id"] for match in matches if match["recovered"]
    }
    comparable = [
        (
            float(match["estimated_relative_performance"]),
            float(match["actual_performance_score"]),
        )
        for match in matches
        if match["actual_performance_score"] is not None and match["temporal_iou"] >= 0.10
    ]
    return {
        "schema": EVALUATION_SCHEMA,
        "prediction_count": len(predictions),
        "organization_clip_count": len(actuals),
        "aligned_organization_clip_count": len(usable_actuals),
        "top_k_recall_at_iou_50": round(
            len(recovered_actual_ids) / len(usable_actuals), 4
        )
        if usable_actuals
        else None,
        "mean_best_iou": round(
            sum(match["temporal_iou"] for match in matches) / len(matches), 4
        )
        if matches
        else None,
        "performance_rank_correlation": (
            round(value, 4) if (value := _rank_correlation(comparable)) is not None else None
        ),
        "compound_organization_clip_count": sum(
            bool(item.get("compound_edit_detected")) for item in usable_actuals
        ),
        "matches": matches,
        "notes": [
            "Recommendations were frozen before holdout Short files were accepted.",
            "Temporal recovery uses one-to-one matching and an IoU threshold of 0.50.",
            "The performance score is a within-test composite of views, likes per view, "
            "and comments per view; it is not a view forecast.",
        ],
    }
