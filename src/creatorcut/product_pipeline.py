"""Production-style upload inference and exact clip export for the local MVP."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any

import av

from creatorcut.candidates import (
    generate_candidates,
    interval_iou,
    words_to_sentence_units,
)
from creatorcut.holdout import score_frozen_model
from creatorcut.media import inspect_media
from creatorcut.product_store import ProductStore
from creatorcut.semantic import (
    DEFAULT_MAX_LENGTH,
    _download_model_files,
    build_embedding_artifact,
    encode_texts_onnx,
)
from creatorcut.training import candidate_model_features
from creatorcut.transcription import transcribe_video

MAXIMUM_CANDIDATES_TO_EMBED = 1_200
MAXIMUM_EXPORT_ADJUSTMENT_SECONDS = 15.0
MINIMUM_EXPORT_DURATION_SECONDS = 5.0
MAXIMUM_EXPORT_DURATION_SECONDS = 90.0
MAXIMUM_EXPORT_EDGE_PIXELS = 1_920


def candidate_prefilter_score(candidate: dict[str, Any]) -> float:
    """Use cheap structural signals to limit semantic inference on very long uploads."""
    features = candidate_model_features(candidate)
    return (
        features["clean_start"]
        + features["complete_end"]
        + features["hook_signal"]
        + features["duration_fit"]
        + features["density_fit"]
        + features["structure_fit"]
        + features["lexical_fit"]
        + features["filler_control"]
        - features["intro_outro"]
    )


def prefilter_candidates(
    candidates: list[dict[str, Any]], limit: int = MAXIMUM_CANDIDATES_TO_EMBED
) -> list[dict[str, Any]]:
    """Keep the strongest cheap candidates while preserving coverage across the full video."""
    if len(candidates) <= limit:
        return candidates
    ordered = sorted(
        candidates,
        key=lambda candidate: (-candidate_prefilter_score(candidate), candidate["candidate_id"]),
    )
    bucket_count = min(20, limit)
    per_bucket = max(1, limit // bucket_count)
    maximum_start = max(float(candidate["start_seconds"]) for candidate in candidates) or 1.0
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(bucket_count)]
    for candidate in ordered:
        bucket = min(
            bucket_count - 1,
            int((float(candidate["start_seconds"]) / maximum_start) * bucket_count),
        )
        if len(buckets[bucket]) < per_bucket:
            buckets[bucket].append(candidate)
    selected = [candidate for bucket in buckets for candidate in bucket]
    selected_ids = {candidate["candidate_id"] for candidate in selected}
    selected.extend(
        candidate for candidate in ordered if candidate["candidate_id"] not in selected_ids
    )
    return selected[:limit]


def select_diverse_top_clips(
    candidates: list[dict[str, Any]], count: int = 3, maximum_iou: float = 0.25
) -> list[dict[str, Any]]:
    """Greedily choose high-ranked clips that do not repeat the same moment."""
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate["personalized_score"]),
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
            return selected

    selected_ids = {candidate["candidate_id"] for candidate in selected}
    selected.extend(
        candidate for candidate in ordered if candidate["candidate_id"] not in selected_ids
    )
    return selected[:count]


def explain_prediction(predicted_targets: dict[str, float]) -> str:
    """Turn the model's strongest output dimensions into restrained product copy."""
    descriptions = {
        "hook": "a strong opening",
        "completeness": "a self-contained thought",
        "payoff": "a clear payoff",
        "clarity": "easy-to-follow delivery",
    }
    strongest = sorted(
        predicted_targets,
        key=lambda field: (-float(predicted_targets[field]), field),
    )[:2]
    return "Selected for " + " and ".join(descriptions[field] for field in strongest) + "."


def validate_export_interval(
    clip: dict[str, Any],
    start_seconds: float,
    end_seconds: float,
    media_duration_seconds: float,
) -> tuple[float, float]:
    """Validate a small user edit before media work or feedback persistence."""
    try:
        start = float(start_seconds)
        end = float(end_seconds)
    except (TypeError, ValueError) as error:
        raise ValueError("Clip timestamps must be numbers") from error
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("Clip timestamps must be finite numbers")
    if start < 0 or end > media_duration_seconds or end <= start:
        raise ValueError("Clip timestamps fall outside the uploaded video")
    if abs(start - float(clip["start_seconds"])) > MAXIMUM_EXPORT_ADJUSTMENT_SECONDS:
        raise ValueError("The edited start must stay within 15 seconds of the recommendation")
    if abs(end - float(clip["end_seconds"])) > MAXIMUM_EXPORT_ADJUSTMENT_SECONDS:
        raise ValueError("The edited end must stay within 15 seconds of the recommendation")
    duration = end - start
    if duration < MINIMUM_EXPORT_DURATION_SECONDS or duration > MAXIMUM_EXPORT_DURATION_SECONDS:
        raise ValueError("Edited clips must be between 5 and 90 seconds")
    return round(start, 3), round(end, 3)


def export_dimensions(width: int, height: int) -> tuple[int, int]:
    """Fit video inside a 1920px box and retain encoder-safe even dimensions."""
    scale = min(1.0, MAXIMUM_EXPORT_EDGE_PIXELS / max(width, height))
    output_width = max(2, int(round(width * scale)) // 2 * 2)
    output_height = max(2, int(round(height * scale)) // 2 * 2)
    return output_width, output_height


def transcode_clip(source_path: Path, output_path: Path, start: float, end: float) -> None:
    """Decode and re-encode a frame-accurate MP4 interval using bundled PyAV codecs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.mp4")
    with av.open(str(source_path)) as source, av.open(str(temporary_path), mode="w") as output:
        video_input = next(iter(source.streams.video), None)
        audio_input = next(iter(source.streams.audio), None)
        if video_input is None:
            raise ValueError("Uploaded media has no video stream")

        rate = video_input.average_rate or 30
        video_output = output.add_stream("libx264", rate=rate)
        video_output.width, video_output.height = export_dimensions(
            video_input.codec_context.width,
            video_input.codec_context.height,
        )
        video_output.pix_fmt = "yuv420p"
        video_output.options = {"crf": "21", "preset": "veryfast"}

        audio_output = None
        if audio_input is not None:
            sample_rate = audio_input.codec_context.sample_rate or 48_000
            audio_output = output.add_stream("aac", rate=sample_rate)
            audio_output.bit_rate = 128_000
            if audio_input.codec_context.layout is not None:
                audio_output.layout = audio_input.codec_context.layout.name

        stream_pairs = {video_input.index: video_output}
        if audio_input is not None and audio_output is not None:
            stream_pairs[audio_input.index] = audio_output
        source.seek(int(start * av.time_base), backward=True, any_frame=False)
        completed: set[int] = set()

        for packet in source.demux([video_input] + ([audio_input] if audio_input else [])):
            if packet.stream.index not in stream_pairs:
                continue
            output_stream = stream_pairs[packet.stream.index]
            for frame in packet.decode():
                if frame.time is None:
                    continue
                frame_time = float(frame.time)
                if frame_time < start:
                    continue
                if frame_time >= end:
                    completed.add(packet.stream.index)
                    continue
                if frame.pts is not None and frame.time_base is not None:
                    frame.pts -= int(round(start / float(frame.time_base)))
                for encoded_packet in output_stream.encode(frame):
                    output.mux(encoded_packet)
            if len(completed) == len(stream_pairs):
                break

        for output_stream in stream_pairs.values():
            for encoded_packet in output_stream.encode(None):
                output.mux(encoded_packet)
    temporary_path.replace(output_path)


class ProductProcessor:
    """Own heavyweight inference resources and process one uploaded video at a time."""

    def __init__(
        self,
        store: ProductStore,
        work_dir: Path,
        frozen_model_path: Path,
        model_cache: Path,
        semantic_cache: Path,
    ) -> None:
        self.store = store
        self.work_dir = work_dir
        self.frozen_model_path = frozen_model_path
        self.model_cache = model_cache
        self.semantic_cache = semantic_cache
        self._whisper_model: Any | None = None
        self._resource_lock = threading.Lock()

    def _transcriber(self) -> Any:
        with self._resource_lock:
            if self._whisper_model is None:
                from faster_whisper import WhisperModel

                self._whisper_model = WhisperModel(
                    "small.en",
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self.model_cache),
                )
            return self._whisper_model

    def process_video(self, video_id: str) -> None:
        """Run validation, ASR, candidate generation, frozen scoring, and personalization."""
        try:
            source_path = self.store.media_path_for_video(video_id)
            self.store.update_video(video_id, "validating")
            media = inspect_media(source_path)
            if not media["valid"]:
                raise ValueError("; ".join(media["errors"]))
            media_duration = float(media["duration_seconds"])
            self.store.update_video(video_id, "transcribing", duration_seconds=media_duration)

            job_dir = self.work_dir / video_id
            transcript_path = job_dir / "transcript.json"
            transcript = transcribe_video(
                self._transcriber(),
                video_id,
                source_path,
                transcript_path,
                "small.en",
                "int8",
                1,
                True,
            )
            self.store.update_video(video_id, "generating_candidates")
            words = [
                word
                for segment in transcript["segments"]
                for word in segment.get("words", [])
            ]
            candidates = generate_candidates(video_id, words_to_sentence_units(words))
            if len(candidates) < 3:
                raise ValueError("The video did not contain enough spoken content for three clips")
            candidates = prefilter_candidates(candidates)

            queue: list[dict[str, Any]] = []
            for candidate in candidates:
                queue.append(
                    {
                        "annotation_id": f"{candidate['candidate_id']}_production",
                        "candidate_id": candidate["candidate_id"],
                        "video_id": video_id,
                        "start_seconds": candidate["start_seconds"],
                        "end_seconds": candidate["end_seconds"],
                        "duration_seconds": candidate["duration_seconds"],
                        "transcript_text": candidate["text"],
                        "labels": {},
                    }
                )
            self.store.update_video(video_id, "ranking_candidates")
            frozen_model = json.loads(self.frozen_model_path.read_text(encoding="utf-8"))
            semantic_metadata = frozen_model["semantic_feature_metadata"]
            tokenizer_path, encoder_path = _download_model_files(
                semantic_metadata["model_id"],
                semantic_metadata["revision"],
                semantic_metadata["onnx_filename"],
                self.semantic_cache,
            )
            embeddings = encode_texts_onnx(
                [record["transcript_text"] for record in queue],
                tokenizer_path,
                encoder_path,
                batch_size=16,
                maximum_length=semantic_metadata.get("maximum_length", DEFAULT_MAX_LENGTH),
            )
            embedding_artifact = build_embedding_artifact(
                queue,
                embeddings,
                model_id=semantic_metadata["model_id"],
                revision=semantic_metadata["revision"],
                onnx_filename=semantic_metadata["onnx_filename"],
                maximum_length=semantic_metadata.get("maximum_length", DEFAULT_MAX_LENGTH),
            )
            predictions = score_frozen_model(queue, embedding_artifact, frozen_model)["records"]
            candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
            scored = [
                {
                    **candidate_by_id[prediction["candidate_id"]],
                    "transcript_text": candidate_by_id[prediction["candidate_id"]]["text"],
                    "global_score": prediction["predicted_quality_score"],
                    "predicted_targets": prediction["predicted_targets"],
                }
                for prediction in predictions
            ]
            video = self.store.get_video(video_id)
            personalized, _ = self.store.personalize_candidates(video["creator_id"], scored)
            selected = select_diverse_top_clips(personalized)
            ranked = []
            for rank, clip in enumerate(selected, start=1):
                ranked.append(
                    {
                        **clip,
                        "id": f"{video_id}_clip_{rank}",
                        "rank": rank,
                        "explanation": explain_prediction(clip["predicted_targets"]),
                    }
                )
            self.store.save_ranked_clips(video_id, ranked)
            self.store.update_video(video_id, "ready")
        except Exception as error:  # background failures must become visible job state
            self.store.update_video(video_id, "failed", error_message=str(error)[:500])

    def export_clip(self, clip_id: str, start: float, end: float) -> Path:
        """Validate and cache one original or adjusted clip download."""
        clip = self.store.get_clip(clip_id)
        video = self.store.get_video(clip["video_id"])
        checked_start, checked_end = validate_export_interval(
            clip,
            start,
            end,
            float(video["duration_seconds"]),
        )
        export_dir = self.work_dir / "exports"
        export_name = (
            f"{clip_id}_{round(checked_start * 1000)}_{round(checked_end * 1000)}.mp4"
        )
        output_path = export_dir / export_name
        if not output_path.exists():
            transcode_clip(Path(clip["media_path"]), output_path, checked_start, checked_end)
        return output_path
