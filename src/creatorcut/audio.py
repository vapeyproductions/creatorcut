"""Audio-delivery features and leakage-safe multimodal ranking ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from creatorcut.candidates import CORE_SCORE_FIELDS, write_jsonl
from creatorcut.dataset import load_jsonl, load_manifest
from creatorcut.semantic import EMBEDDING_SCHEMA, embedding_matrix_for_records
from creatorcut.training import MODEL_FEATURE_FIELDS, cross_validate_ridge, paired_video_bootstrap

AUDIO_FEATURE_SCHEMA = "audio_delivery_v1"
AUDIO_FEATURE_FIELDS = (
    "active_log_energy_mean_db",
    "energy_std_db",
    "energy_dynamic_range_db",
    "opening_energy_delta_db",
    "closing_energy_delta_db",
    "energy_slope_db",
    "energy_silence_ratio",
    "energy_pause_rate_per_minute",
    "energy_mean_pause_seconds",
    "energy_max_pause_seconds",
    "vad_speech_ratio",
    "vad_segments_per_minute",
    "vad_mean_segment_seconds",
    "vad_max_silence_seconds",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
    "pitch_median_hz",
    "pitch_std_semitones",
    "pitch_range_semitones",
    "pitch_voiced_frame_ratio",
    "clipping_ratio",
)
SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.04
HOP_SECONDS = 0.02
MINIMUM_PAUSE_SECONDS = 0.25


def _frame_audio(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if audio.ndim != 1:
        raise ValueError("audio must be a one-dimensional mono waveform")
    if not len(audio):
        raise ValueError("audio must not be empty")
    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))
    frame_count = 1 + (len(audio) - frame_length) // hop_length
    indices = np.arange(frame_length)[None, :] + hop_length * np.arange(frame_count)[:, None]
    return audio[indices]


def _true_run_seconds(mask: np.ndarray, hop_seconds: float) -> list[float]:
    runs: list[float] = []
    current = 0
    for value in mask.tolist():
        if value:
            current += 1
        elif current:
            runs.append(current * hop_seconds)
            current = 0
    if current:
        runs.append(current * hop_seconds)
    return runs


def _pitch_statistics(
    frames: np.ndarray,
    active_mask: np.ndarray,
    sampling_rate: int,
) -> tuple[float, float, float, float]:
    active_indices = np.flatnonzero(active_mask)
    if not len(active_indices):
        return 0.0, 0.0, 0.0, 0.0
    if len(active_indices) > 800:
        positions = np.linspace(0, len(active_indices) - 1, 800, dtype=int)
        active_indices = active_indices[positions]
    selected = frames[active_indices].astype(np.float64)
    selected -= selected.mean(axis=1, keepdims=True)
    selected *= np.hanning(selected.shape[1])[None, :]
    fft_size = 1 << (2 * selected.shape[1] - 1).bit_length()
    spectrum = np.fft.rfft(selected, n=fft_size, axis=1)
    autocorrelation = np.fft.irfft(spectrum * spectrum.conjugate(), n=fft_size, axis=1)
    autocorrelation = autocorrelation[:, : selected.shape[1]]
    autocorrelation /= np.clip(autocorrelation[:, :1], 1e-12, None)
    minimum_lag = max(1, math.floor(sampling_rate / 400.0))
    maximum_lag = min(selected.shape[1] - 1, math.ceil(sampling_rate / 60.0))
    search = autocorrelation[:, minimum_lag : maximum_lag + 1]
    best_offsets = np.argmax(search, axis=1)
    peaks = search[np.arange(len(search)), best_offsets]
    voiced = peaks >= 0.30
    if not voiced.any():
        return 0.0, 0.0, 0.0, 0.0
    pitches = sampling_rate / (best_offsets[voiced] + minimum_lag)
    semitones = 69.0 + 12.0 * np.log2(pitches / 440.0)
    return (
        float(np.median(pitches)),
        float(np.std(semitones)),
        float(np.percentile(semitones, 90) - np.percentile(semitones, 10)),
        float(voiced.mean()),
    )


def _vad_statistics(
    speech_timestamps: list[dict[str, int]],
    sample_count: int,
    sampling_rate: int,
) -> tuple[float, float, float, float]:
    duration = sample_count / sampling_rate
    speech_durations = [
        max(0.0, (value["end"] - value["start"]) / sampling_rate)
        for value in speech_timestamps
    ]
    speech_seconds = sum(speech_durations)
    silence_gaps: list[float] = []
    previous_end = 0
    for value in speech_timestamps:
        silence_gaps.append(max(0.0, (value["start"] - previous_end) / sampling_rate))
        previous_end = value["end"]
    silence_gaps.append(max(0.0, (sample_count - previous_end) / sampling_rate))
    return (
        speech_seconds / duration if duration else 0.0,
        len(speech_timestamps) / duration * 60.0 if duration else 0.0,
        float(np.mean(speech_durations)) if speech_durations else 0.0,
        max(silence_gaps, default=0.0),
    )


def extract_audio_features(
    audio: np.ndarray,
    speech_timestamps: list[dict[str, int]],
    sampling_rate: int = SAMPLE_RATE,
) -> dict[str, float]:
    """Extract fixed delivery, pause, energy, and pitch descriptors from one clip."""
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1 or not len(waveform):
        raise ValueError("audio must be a non-empty one-dimensional waveform")
    if not np.isfinite(waveform).all():
        raise ValueError("audio contains non-finite samples")

    frame_length = max(2, round(FRAME_SECONDS * sampling_rate))
    hop_length = max(1, round(HOP_SECONDS * sampling_rate))
    frames = _frame_audio(waveform, frame_length, hop_length)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    log_energy = 20.0 * np.log10(rms + 1e-8)
    reference_energy = float(np.percentile(log_energy, 90))
    silence_threshold = max(-55.0, reference_energy - 25.0)
    silent = log_energy < silence_threshold
    active = ~silent
    active_energy = log_energy[active] if active.any() else log_energy
    pause_runs = [
        value
        for value in _true_run_seconds(silent, HOP_SECONDS)
        if value >= MINIMUM_PAUSE_SECONDS
    ]
    duration = len(waveform) / sampling_rate
    edge_frames = max(1, round(2.0 / HOP_SECONDS))
    overall_energy = float(np.mean(active_energy))
    opening_energy = float(np.mean(log_energy[:edge_frames]))
    closing_energy = float(np.mean(log_energy[-edge_frames:]))
    normalized_time = np.linspace(-1.0, 1.0, len(log_energy))
    centered_time = normalized_time - normalized_time.mean()
    energy_slope = float(
        np.dot(centered_time, log_energy - log_energy.mean())
        / np.clip(np.dot(centered_time, centered_time), 1e-12, None)
    )
    signs = np.signbit(frames)
    zero_crossing = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    pitch_median, pitch_std, pitch_range, voiced_ratio = _pitch_statistics(
        frames, active, sampling_rate
    )
    vad_speech_ratio, vad_rate, vad_mean, vad_max_silence = _vad_statistics(
        speech_timestamps, len(waveform), sampling_rate
    )
    features = {
        "active_log_energy_mean_db": overall_energy,
        "energy_std_db": float(np.std(active_energy)),
        "energy_dynamic_range_db": float(
            np.percentile(active_energy, 90) - np.percentile(active_energy, 10)
        ),
        "opening_energy_delta_db": opening_energy - overall_energy,
        "closing_energy_delta_db": closing_energy - overall_energy,
        "energy_slope_db": energy_slope,
        "energy_silence_ratio": float(silent.mean()),
        "energy_pause_rate_per_minute": len(pause_runs) / duration * 60.0,
        "energy_mean_pause_seconds": float(np.mean(pause_runs)) if pause_runs else 0.0,
        "energy_max_pause_seconds": max(pause_runs, default=0.0),
        "vad_speech_ratio": vad_speech_ratio,
        "vad_segments_per_minute": vad_rate,
        "vad_mean_segment_seconds": vad_mean,
        "vad_max_silence_seconds": vad_max_silence,
        "zero_crossing_rate_mean": float(np.mean(zero_crossing)),
        "zero_crossing_rate_std": float(np.std(zero_crossing)),
        "pitch_median_hz": pitch_median,
        "pitch_std_semitones": pitch_std,
        "pitch_range_semitones": pitch_range,
        "pitch_voiced_frame_ratio": voiced_ratio,
        "clipping_ratio": float(np.mean(np.abs(waveform) >= 0.999)),
    }
    if set(features) != set(AUDIO_FEATURE_FIELDS):
        raise RuntimeError("Audio feature implementation does not match its schema")
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("audio features contain non-finite values")
    return features


def extract_audio_artifact(
    records: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    media_dir: Path,
) -> dict[str, Any]:
    """Decode every source once and cache compact features for its reviewed clips."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    if not records:
        raise ValueError("Training records are empty")
    paths = {
        item["video_id"]: media_dir / Path(item["local_filename"]).name for item in manifest
    }
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_video[record["video_id"]].append(record)
    vad_options = VadOptions(
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
    )
    extracted: list[dict[str, Any]] = []
    for video_index, (video_id, video_records) in enumerate(sorted(by_video.items()), start=1):
        path = paths.get(video_id)
        if path is None or not path.is_file():
            raise FileNotFoundError(path or video_id)
        audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
        for record in video_records:
            start_sample = max(0, round(float(record["start_seconds"]) * SAMPLE_RATE))
            end_sample = min(len(audio), round(float(record["end_seconds"]) * SAMPLE_RATE))
            clip = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
            if not len(clip):
                raise ValueError(f"{record['annotation_id']}: decoded clip is empty")
            speech = get_speech_timestamps(
                clip,
                vad_options=vad_options,
                sampling_rate=SAMPLE_RATE,
            )
            extracted.append(
                {
                    "annotation_id": record["annotation_id"],
                    "video_id": video_id,
                    "features": extract_audio_features(clip, speech, SAMPLE_RATE),
                }
            )
        print(
            f"[{video_index}/{len(by_video)}] {video_id}: "
            f"extracted {len(video_records)} reviewed clips",
            flush=True,
        )
    order = {record["annotation_id"]: index for index, record in enumerate(records)}
    extracted.sort(key=lambda value: order[value["annotation_id"]])
    return {
        "metadata": {
            "audio_feature_schema": AUDIO_FEATURE_SCHEMA,
            "feature_fields": list(AUDIO_FEATURE_FIELDS),
            "sample_rate": SAMPLE_RATE,
            "frame_seconds": FRAME_SECONDS,
            "hop_seconds": HOP_SECONDS,
            "pitch_method": "windowed autocorrelation via FFT, 60-400 Hz",
            "voice_activity_model": "Silero VAD bundled with faster-whisper 1.2.1",
            "label_dependent_extraction": False,
        },
        "records": extracted,
    }


def audio_matrix_for_records(
    records: list[dict[str, Any]], artifact: dict[str, Any]
) -> np.ndarray:
    """Join a validated audio artifact to training rows by annotation identity."""
    metadata = artifact.get("metadata", {})
    if metadata.get("audio_feature_schema") != AUDIO_FEATURE_SCHEMA:
        raise ValueError(f"Expected audio schema {AUDIO_FEATURE_SCHEMA}")
    if metadata.get("feature_fields") != list(AUDIO_FEATURE_FIELDS):
        raise ValueError("Audio feature fields do not match the declared schema")
    indexed: dict[str, dict[str, Any]] = {}
    for value in artifact.get("records", []):
        annotation_id = value.get("annotation_id")
        if not isinstance(annotation_id, str) or annotation_id in indexed:
            raise ValueError("Audio records require unique annotation IDs")
        indexed[annotation_id] = value
    expected_ids = {record["annotation_id"] for record in records}
    if set(indexed) != expected_ids:
        missing = sorted(expected_ids - indexed.keys())
        extra = sorted(indexed.keys() - expected_ids)
        raise ValueError(f"Audio IDs do not match training data; missing={missing}, extra={extra}")

    rows: list[list[float]] = []
    for record in records:
        value = indexed[record["annotation_id"]]
        if value.get("video_id") != record["video_id"]:
            raise ValueError(f"{record['annotation_id']}: audio video_id does not match")
        features = value.get("features", {})
        if set(features) != set(AUDIO_FEATURE_FIELDS):
            raise ValueError(f"{record['annotation_id']}: invalid audio feature fields")
        rows.append([float(features[field]) for field in AUDIO_FEATURE_FIELDS])
    matrix = np.asarray(rows, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Audio artifact contains non-finite values")
    return matrix


def _variant_records(
    records: list[dict[str, Any]],
    matrix: np.ndarray,
    fields: tuple[str, ...],
    schema: str,
) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "feature_schema": schema,
            "features": dict(zip(fields, row.tolist(), strict=True)),
        }
        for record, row in zip(records, matrix, strict=True)
    ]


def _dataset_fingerprint(records: list[dict[str, Any]]) -> str:
    identity = [
        {
            "annotation_id": record["annotation_id"],
            "video_id": record["video_id"],
            "targets": {field: record["targets"][field] for field in CORE_SCORE_FIELDS},
        }
        for record in records
    ]
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_audio_ablation(
    records: list[dict[str, Any]],
    embedding_artifact: dict[str, Any],
    audio_artifact: dict[str, Any],
    n_splits: int = 4,
    seed: int = 42,
    alpha: float = 10.0,
    bootstrap_iterations: int = 10_000,
) -> dict[str, Any]:
    """Compare audio, text, and fused pointwise rankers on identical grouped folds."""
    embeddings = embedding_matrix_for_records(records, embedding_artifact)
    audio = audio_matrix_for_records(records, audio_artifact)
    handcrafted = np.asarray(
        [[record["features"][field] for field in MODEL_FEATURE_FIELDS] for record in records],
        dtype=float,
    )
    embedding_fields = tuple(f"embedding_{index:03d}" for index in range(embeddings.shape[1]))
    source_schemas = {record.get("feature_schema") for record in records}
    if len(source_schemas) != 1:
        raise ValueError(f"Expected one training feature schema, found {source_schemas}")
    source_schema = next(iter(source_schemas))
    if not isinstance(source_schema, str):
        raise ValueError("Training records require a feature schema")

    variants = {
        "audio_ridge": (
            audio,
            AUDIO_FEATURE_FIELDS,
            AUDIO_FEATURE_SCHEMA,
        ),
        "semantic_ridge": (
            embeddings,
            embedding_fields,
            EMBEDDING_SCHEMA,
        ),
        "text_semantic_ridge": (
            np.column_stack([handcrafted, embeddings]),
            (*MODEL_FEATURE_FIELDS, *embedding_fields),
            f"{source_schema}_plus_{EMBEDDING_SCHEMA}",
        ),
        "semantic_audio_ridge": (
            np.column_stack([embeddings, audio]),
            (*embedding_fields, *AUDIO_FEATURE_FIELDS),
            f"{EMBEDDING_SCHEMA}_plus_{AUDIO_FEATURE_SCHEMA}",
        ),
        "full_multimodal_ridge": (
            np.column_stack([handcrafted, embeddings, audio]),
            (*MODEL_FEATURE_FIELDS, *embedding_fields, *AUDIO_FEATURE_FIELDS),
            f"{source_schema}_plus_{EMBEDDING_SCHEMA}_plus_{AUDIO_FEATURE_SCHEMA}",
        ),
    }
    experiments: dict[str, dict[str, Any]] = {}
    for name, (matrix, fields, schema) in variants.items():
        experiments[name] = cross_validate_ridge(
            _variant_records(records, matrix, fields, schema),
            n_splits=n_splits,
            seed=seed,
            alpha=alpha,
            feature_fields=fields,
            feature_schema=schema,
            experiment=f"grouped_{name}_v1",
        )

    predictions = []
    for index, record in enumerate(records):
        reference = experiments["text_semantic_ridge"]["predictions"][index]
        predictions.append(
            {
                "annotation_id": record["annotation_id"],
                "video_id": record["video_id"],
                "fold": reference["fold"],
                "actual_quality_score": reference["actual_quality_score"],
                "predictions": {
                    name: experiment["predictions"][index]["ridge_quality_score"]
                    for name, experiment in experiments.items()
                },
            }
        )
    actual = [value["actual_quality_score"] for value in predictions]
    reference_scores = [
        value["predictions"]["text_semantic_ridge"] for value in predictions
    ]
    challenger_scores = [
        value["predictions"]["full_multimodal_ridge"] for value in predictions
    ]
    models = {
        name: experiment["models"]["ridge"] for name, experiment in experiments.items()
    }
    final_models = {
        name: experiment["final_model"] for name, experiment in experiments.items()
    }
    candidate_names = ("text_semantic_ridge", "full_multimodal_ridge")
    selected_name = max(
        candidate_names,
        key=lambda name: (
            models[name]["ranking"]["top_1_hit_rate"],
            -models[name]["ranking"]["mean_top_1_regret"],
            models[name]["ranking"]["pairwise_accuracy"],
        ),
    )
    return {
        "experiment": "audio_delivery_representation_ablation_v1",
        "dataset": experiments["text_semantic_ridge"]["dataset"],
        "development_dataset_sha256": _dataset_fingerprint(records),
        "audio": audio_artifact["metadata"],
        "semantic": embedding_artifact["metadata"],
        "cross_validation": experiments["text_semantic_ridge"]["cross_validation"],
        "regularization": {
            "model": "multi-output ridge regression",
            "alpha": alpha,
            "selection": "fixed for every representation before the ablation",
        },
        "selection_rule": (
            "Between text_semantic_ridge and full_multimodal_ridge, maximize grouped top-1 hit "
            "rate, then minimize top-1 regret, then maximize pairwise accuracy."
        ),
        "selected_model": selected_name,
        "models": models,
        "uncertainty": {
            "full_multimodal_vs_text_semantic": paired_video_bootstrap(
                records,
                actual,
                reference_scores,
                challenger_scores,
                iterations=bootstrap_iterations,
                seed=seed,
            )
        },
        "final_models": final_models,
        "predictions": predictions,
    }


def build_frozen_model_artifact(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Package the selected development model and immutable evaluation contract."""
    selected = evaluation["selected_model"]
    return {
        "freeze_schema": "creatorcut_ranker_freeze_v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "status": "frozen_for_external_holdout_evaluation",
        "selected_model": selected,
        "selection_rule": evaluation["selection_rule"],
        "development_dataset": {
            **evaluation["dataset"],
            "sha256": evaluation["development_dataset_sha256"],
        },
        "evaluation_protocol": evaluation["cross_validation"],
        "audio_feature_metadata": evaluation["audio"],
        "semantic_feature_metadata": evaluation["semantic"],
        "model": evaluation["final_models"][selected],
        "holdout_policy": {
            "new_video_ids_only": True,
            "fit_or_tune_on_holdout": False,
            "report_before_any_retraining": True,
        },
    }


def extract_main() -> None:
    """Extract compact audio features from the reviewed development clips."""
    parser = argparse.ArgumentParser(description="Extract CreatorCut audio-delivery features")
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/training_clips.jsonl")
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/videos.json"))
    parser.add_argument("--media-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/audio_features.json")
    )
    args = parser.parse_args()

    artifact = extract_audio_artifact(
        load_jsonl(args.input), load_manifest(args.manifest), args.media_dir
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact), encoding="utf-8")
    print(
        f"Wrote {len(artifact['records'])} audio feature rows with "
        f"{len(AUDIO_FEATURE_FIELDS)} fields to {args.output}"
    )


def evaluate_main() -> None:
    """Run the audio ablation and freeze the selected development model."""
    parser = argparse.ArgumentParser(description="Evaluate CreatorCut audio feature fusion")
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/training_clips.jsonl")
    )
    parser.add_argument(
        "--embeddings", type=Path, default=Path("data/processed/semantic_embeddings.json")
    )
    parser.add_argument(
        "--audio", type=Path, default=Path("data/processed/audio_features.json")
    )
    parser.add_argument(
        "--evaluation", type=Path, default=Path("data/processed/audio_evaluation.json")
    )
    parser.add_argument(
        "--models", type=Path, default=Path("data/processed/audio_models.json")
    )
    parser.add_argument(
        "--predictions", type=Path, default=Path("data/processed/audio_predictions.jsonl")
    )
    parser.add_argument(
        "--frozen-model", type=Path, default=Path("data/processed/frozen_model_v1.json")
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()

    evaluation = run_audio_ablation(
        load_jsonl(args.input),
        json.loads(args.embeddings.read_text(encoding="utf-8")),
        json.loads(args.audio.read_text(encoding="utf-8")),
        n_splits=args.folds,
        seed=args.seed,
        alpha=args.alpha,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    predictions = evaluation.pop("predictions")
    final_models = evaluation.pop("final_models")
    evaluation_with_models = {**evaluation, "final_models": final_models}
    frozen = build_frozen_model_artifact(evaluation_with_models)
    args.evaluation.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    args.models.parent.mkdir(parents=True, exist_ok=True)
    args.models.write_text(json.dumps(final_models), encoding="utf-8")
    write_jsonl(predictions, args.predictions)
    args.frozen_model.parent.mkdir(parents=True, exist_ok=True)
    args.frozen_model.write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    summary = {
        name: {
            "quality_score": result["regression"]["quality_score"],
            "ranking": result["ranking"],
        }
        for name, result in evaluation["models"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"Selected and froze {evaluation['selected_model']}")
    print(f"Wrote evaluation to {args.evaluation}")
    print(f"Wrote frozen model to {args.frozen_model}")


if __name__ == "__main__":
    evaluate_main()
