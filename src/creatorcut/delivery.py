"""Compact audio and visual delivery features for production clip ranking."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

DELIVERY_FEATURE_SCHEMA = "creatorcut_delivery_features_v1"
VISUAL_SAMPLE_SECONDS = 1.0


def _unit_percentiles(values: list[float]) -> list[float]:
    """Convert one within-video feature into tie-aware percentile scores."""
    if not values:
        return []
    if max(values) == min(values):
        return [0.5] * len(values)
    result = []
    for value in values:
        below = sum(candidate < value for candidate in values)
        equal = sum(candidate == value for candidate in values)
        result.append((below + (equal - 1) / 2) / (len(values) - 1))
    return result


def normalize_delivery_features(records: list[dict[str, float]]) -> list[dict[str, float]]:
    """Add relative urgency and visual-excitement scores to raw clip descriptors."""
    if not records:
        return []
    fields = {
        field: _unit_percentiles([float(record[field]) for record in records])
        for field in (
            "audio_energy_db",
            "audio_variation_db",
            "audio_opening_delta_db",
            "speech_words_per_second",
            "visual_colorfulness",
            "visual_saturation",
            "visual_motion",
            "visual_scene_change_rate",
        )
    }
    silence = _unit_percentiles([float(record["audio_silence_ratio"]) for record in records])
    normalized = []
    for index, record in enumerate(records):
        urgency = (
            fields["audio_energy_db"][index]
            + fields["audio_variation_db"][index]
            + fields["audio_opening_delta_db"][index]
            + fields["speech_words_per_second"][index]
            + (1.0 - silence[index])
        ) / 5.0
        excitement = (
            fields["visual_colorfulness"][index]
            + fields["visual_saturation"][index]
            + fields["visual_motion"][index]
            + fields["visual_scene_change_rate"][index]
        ) / 4.0
        normalized.append(
            {
                **record,
                "audio_urgency_score": round(urgency, 6),
                "visual_excitement_score": round(excitement, 6),
            }
        )
    return normalized


def _audio_features(audio: np.ndarray, candidate: dict[str, Any]) -> dict[str, float]:
    sample_rate = 16_000
    start = max(0, round(float(candidate["start_seconds"]) * sample_rate))
    end = min(len(audio), round(float(candidate["end_seconds"]) * sample_rate))
    clip = np.asarray(audio[start:end], dtype=np.float32)
    if not len(clip):
        return {
            "audio_energy_db": -80.0,
            "audio_variation_db": 0.0,
            "audio_opening_delta_db": 0.0,
            "audio_silence_ratio": 1.0,
            "audio_clipping_ratio": 0.0,
        }
    frame_length = 1600
    usable = len(clip) // frame_length * frame_length
    frames = clip[:usable].reshape(-1, frame_length) if usable else clip.reshape(1, -1)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    energy = 20.0 * np.log10(rms + 1e-8)
    active_threshold = max(-55.0, float(np.percentile(energy, 90)) - 25.0)
    active = energy[energy >= active_threshold]
    active = active if len(active) else energy
    edge_frames = max(1, min(len(energy), 20))
    mean_energy = float(np.mean(active))
    return {
        "audio_energy_db": mean_energy,
        "audio_variation_db": float(np.std(active)),
        "audio_opening_delta_db": float(np.mean(energy[:edge_frames])) - mean_energy,
        "audio_silence_ratio": float(np.mean(energy < active_threshold)),
        "audio_clipping_ratio": float(np.mean(np.abs(clip) >= 0.999)),
    }


def _visual_samples(path: Path) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    previous_gray: np.ndarray | None = None
    next_time = 0.0
    with av.open(str(path)) as container:
        stream = next(iter(container.streams.video))
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            frame_time = float(frame.pts * frame.time_base)
            if frame_time + 1e-6 < next_time:
                continue
            pixels = frame.to_ndarray(format="rgb24")
            small = cv2.resize(pixels, (64, 36), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            red, green, blue = [small[:, :, index].astype(np.float32) for index in range(3)]
            red_green = red - green
            yellow_blue = (red + green) / 2.0 - blue
            colorfulness = math.sqrt(float(red_green.var() + yellow_blue.var()))
            colorfulness += 0.3 * math.sqrt(
                float(red_green.mean() ** 2 + yellow_blue.mean() ** 2)
            )
            motion = (
                float(np.mean(np.abs(gray.astype(np.float32) - previous_gray))) / 255.0
                if previous_gray is not None
                else 0.0
            )
            samples.append(
                {
                    "time": frame_time,
                    "visual_colorfulness": colorfulness / 255.0,
                    "visual_saturation": float(np.mean(hsv[:, :, 1])) / 255.0,
                    "visual_motion": motion,
                    "visual_luminance_variation": float(np.std(gray)) / 255.0,
                }
            )
            previous_gray = gray.astype(np.float32)
            next_time = frame_time + VISUAL_SAMPLE_SECONDS
    return samples


def _visual_features(
    samples: list[dict[str, float]], candidate: dict[str, Any]
) -> dict[str, float]:
    start = float(candidate["start_seconds"])
    end = float(candidate["end_seconds"])
    selected = [sample for sample in samples if start <= sample["time"] <= end]
    if not selected and samples:
        midpoint = (start + end) / 2.0
        selected = [min(samples, key=lambda sample: abs(sample["time"] - midpoint))]
    if not selected:
        return {
            "visual_colorfulness": 0.0,
            "visual_saturation": 0.0,
            "visual_motion": 0.0,
            "visual_luminance_variation": 0.0,
            "visual_scene_change_rate": 0.0,
        }
    return {
        field: float(np.mean([sample[field] for sample in selected]))
        for field in (
            "visual_colorfulness",
            "visual_saturation",
            "visual_motion",
            "visual_luminance_variation",
        )
    } | {
        "visual_scene_change_rate": float(
            np.mean([sample["visual_motion"] >= 0.12 for sample in selected])
        )
    }


def extract_candidate_delivery_features(
    path: Path,
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, float | str]]:
    """Decode a source once and attach compact multimodal descriptors per candidate."""
    if not candidates:
        return {}
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(path), sampling_rate=16_000)
    visual = _visual_samples(path)
    records: list[dict[str, float]] = []
    for candidate in candidates:
        duration = max(float(candidate["duration_seconds"]), 1e-6)
        records.append(
            _audio_features(audio, candidate)
            | _visual_features(visual, candidate)
            | {
                "speech_words_per_second": float(candidate.get("word_count", 0)) / duration,
            }
        )
    normalized = normalize_delivery_features(records)
    return {
        candidate["candidate_id"]: {
            "schema": DELIVERY_FEATURE_SCHEMA,
            **features,
        }
        for candidate, features in zip(candidates, normalized, strict=True)
    }
