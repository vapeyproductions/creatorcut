"""Production-style upload inference and exact clip export for the local MVP."""

from __future__ import annotations

import json
import math
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import av
import numpy as np

from creatorcut.backtest import align_short_to_source, audio_envelope, evaluate_backtest
from creatorcut.candidates import (
    generate_candidates,
    interval_iou,
    words_to_sentence_units,
)
from creatorcut.delivery import DELIVERY_FEATURE_SCHEMA, extract_candidate_delivery_features
from creatorcut.holdout import score_frozen_model
from creatorcut.media import inspect_media
from creatorcut.platforms import (
    PLATFORM_PROFILES,
    attach_platform_scores,
    explain_platform_fit,
    plan_platform_clips,
    validate_clip_plan,
)
from creatorcut.product_store import ProductStore
from creatorcut.publishability import (
    attach_publishability,
    publishability_summary,
    summarize_publishability_batch,
)
from creatorcut.reframing import SubjectAwareCropper, crop_left
from creatorcut.repurposing import generate_platform_pack
from creatorcut.semantic import (
    DEFAULT_MAX_LENGTH,
    _download_model_files,
    build_embedding_artifact,
    encode_texts_onnx,
)
from creatorcut.serving_release import validate_serving_release
from creatorcut.training import candidate_model_features
from creatorcut.transcription import transcribe_video

MAXIMUM_CANDIDATES_TO_EMBED = 1_200
MAXIMUM_CANDIDATES_FOR_DELIVERY = 200
MAXIMUM_EXPORT_ADJUSTMENT_SECONDS = 15.0
MINIMUM_EXPORT_DURATION_SECONDS = 5.0
MAXIMUM_EXPORT_DURATION_SECONDS = 90.0
MAXIMUM_EXPORT_EDGE_PIXELS = 1_920
VERTICAL_EXPORT_WIDTH = 720
VERTICAL_EXPORT_HEIGHT = 1_280
SUPPORTED_EXPORT_FORMATS = {"original", "vertical", "vertical_captions"}


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
        + float(candidate.get("publishability", {}).get("score", 1.0))
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


def select_multimodal_candidates(
    candidates: list[dict[str, Any]], limit: int = MAXIMUM_CANDIDATES_FOR_DELIVERY
) -> list[dict[str, Any]]:
    """Bound media feature work while preserving strong clips and source-wide coverage."""
    if limit < 1:
        raise ValueError("The multimodal candidate limit must be positive")
    if len(candidates) <= limit:
        return candidates
    ordered = sorted(
        candidates,
        key=lambda candidate: (-float(candidate["global_score"]), candidate["candidate_id"]),
    )
    top_count = max(1, round(limit * 0.75))
    selected = ordered[:top_count]
    selected_ids = {candidate["candidate_id"] for candidate in selected}
    remaining = [
        candidate
        for candidate in candidates
        if candidate["candidate_id"] not in selected_ids
    ]
    if len(selected) >= limit:
        return selected[:limit]
    maximum_start = max(float(candidate["start_seconds"]) for candidate in candidates) or 1.0
    bucket_count = min(20, limit - len(selected))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(bucket_count)]
    for candidate in remaining:
        bucket = min(
            bucket_count - 1,
            int(float(candidate["start_seconds"]) / maximum_start * bucket_count),
        )
        buckets[bucket].append(candidate)
    for bucket in buckets:
        bucket.sort(
            key=lambda candidate: (-float(candidate["global_score"]), candidate["candidate_id"])
        )
        selected.extend(bucket[:1])
    selected_ids = {candidate["candidate_id"] for candidate in selected}
    selected.extend(
        candidate for candidate in ordered if candidate["candidate_id"] not in selected_ids
    )
    return selected[:limit]


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


def validate_custom_interval(
    start_seconds: float,
    end_seconds: float,
    media_duration_seconds: float,
) -> tuple[float, float]:
    """Validate a creator-authored clip anywhere inside the source video."""
    try:
        start = float(start_seconds)
        end = float(end_seconds)
    except (TypeError, ValueError) as error:
        raise ValueError("Custom clip timestamps must be numbers") from error
    if not all(math.isfinite(value) for value in (start, end, media_duration_seconds)):
        raise ValueError("Custom clip timestamps must be finite numbers")
    if start < 0 or end > media_duration_seconds or end <= start:
        raise ValueError("Custom clip timestamps fall outside the uploaded video")
    if not MINIMUM_EXPORT_DURATION_SECONDS <= end - start <= MAXIMUM_EXPORT_DURATION_SECONDS:
        raise ValueError("Custom clips must be between 5 and 90 seconds")
    return round(start, 3), round(end, 3)


def export_dimensions(width: int, height: int) -> tuple[int, int]:
    """Fit video inside a 1920px box and retain encoder-safe even dimensions."""
    scale = min(1.0, MAXIMUM_EXPORT_EDGE_PIXELS / max(width, height))
    output_width = max(2, int(round(width * scale)) // 2 * 2)
    output_height = max(2, int(round(height * scale)) // 2 * 2)
    return output_width, output_height


def build_caption_cues(
    transcript: dict[str, Any], start: float, end: float
) -> list[dict[str, Any]]:
    """Group word-timestamped ASR output into short on-screen caption cues."""
    words = [
        word
        for segment in transcript.get("segments", [])
        for word in segment.get("words", [])
        if float(word["end"]) > start and float(word["start"]) < end
    ]
    cues: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = []
    for word in words:
        group.append(word)
        text = str(word.get("word", "")).strip()
        group_duration = float(group[-1]["end"]) - float(group[0]["start"])
        if len(group) >= 7 or group_duration >= 2.4 or text.endswith((".", "?", "!")):
            cues.append(
                {
                    "start": max(start, float(group[0]["start"])),
                    "end": min(end, float(group[-1]["end"]) + 0.08),
                    "text": " ".join(str(item.get("word", "")).strip() for item in group),
                }
            )
            group = []
    if group:
        cues.append(
            {
                "start": max(start, float(group[0]["start"])),
                "end": min(end, float(group[-1]["end"]) + 0.08),
                "text": " ".join(str(item.get("word", "")).strip() for item in group),
            }
        )
    return cues


def transcript_text_for_interval(
    transcript: dict[str, Any], start: float, end: float
) -> str:
    """Return exact word-timestamped transcript text for an arbitrary creator interval."""
    return " ".join(
        str(word.get("word", "")).strip()
        for segment in transcript.get("segments", [])
        for word in segment.get("words", [])
        if float(word["end"]) > start and float(word["start"]) < end
    ).strip()


def _active_caption(cues: list[dict[str, Any]], frame_time: float) -> str | None:
    for cue in cues:
        if float(cue["start"]) <= frame_time <= float(cue["end"]):
            return str(cue["text"])
    return None


@lru_cache(maxsize=1)
def _caption_font() -> Any:
    from PIL import ImageFont

    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, 48)
        except OSError:
            continue
    return ImageFont.load_default()


def _vertical_image(
    frame: av.VideoFrame,
    caption: str | None,
    tracker: SubjectAwareCropper | None = None,
    frame_index: int = 0,
) -> np.ndarray:
    """Crop a frame to 9:16 around a tracked face and optionally burn captions."""
    from PIL import Image, ImageDraw

    pixels = frame.to_ndarray(format="rgb24")
    height, width = pixels.shape[:2]
    target_ratio = VERTICAL_EXPORT_WIDTH / VERTICAL_EXPORT_HEIGHT
    if width / height > target_ratio:
        crop_width = max(1, int(round(height * target_ratio)))
        center_x = (
            tracker.center_for_frame(pixels, frame_index)
            if tracker is not None
            else width / 2
        )
        left = crop_left(width, crop_width, center_x)
        pixels = pixels[:, left : left + crop_width]
    else:
        crop_height = max(1, int(round(width / target_ratio)))
        top = max(0, (height - crop_height) // 2)
        pixels = pixels[top : top + crop_height, :]

    image = Image.fromarray(pixels).resize(
        (VERTICAL_EXPORT_WIDTH, VERTICAL_EXPORT_HEIGHT), Image.Resampling.LANCZOS
    )
    if caption:
        draw = ImageDraw.Draw(image)
        font = _caption_font()
        words = caption.split()
        lines: list[str] = []
        current = ""
        for word in words:
            proposal = f"{current} {word}".strip()
            if current and draw.textbbox((0, 0), proposal, font=font)[2] > 620:
                lines.append(current)
                current = word
            else:
                current = proposal
        if current:
            lines.append(current)
        lines = lines[-3:]
        text = "\n".join(lines)
        box = draw.multiline_textbbox((0, 0), text, font=font, spacing=8, align="center")
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        x = (VERTICAL_EXPORT_WIDTH - text_width) / 2
        y = VERTICAL_EXPORT_HEIGHT * 0.72 - text_height / 2
        padding = 18
        draw.rounded_rectangle(
            (x - padding, y - padding, x + text_width + padding, y + text_height + padding),
            radius=10,
            fill=(0, 0, 0),
        )
        draw.multiline_text(
            (x, y),
            text,
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
            spacing=8,
            align="center",
        )
    return np.asarray(image)


def transcode_clip(
    source_path: Path,
    output_path: Path,
    start: float,
    end: float,
    export_format: str = "original",
    caption_cues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Decode and re-encode a frame-accurate MP4 interval using bundled PyAV codecs."""
    if not isinstance(export_format, str) or export_format not in SUPPORTED_EXPORT_FORMATS:
        raise ValueError("Choose a supported export format")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.mp4")
    with av.open(str(source_path)) as source, av.open(str(temporary_path), mode="w") as output:
        video_input = next(iter(source.streams.video), None)
        audio_input = next(iter(source.streams.audio), None)
        if video_input is None:
            raise ValueError("Uploaded media has no video stream")

        rate = video_input.average_rate or 30
        video_output = output.add_stream("libx264", rate=rate)
        if export_format == "original":
            video_output.width, video_output.height = export_dimensions(
                video_input.codec_context.width,
                video_input.codec_context.height,
            )
        else:
            video_output.width = VERTICAL_EXPORT_WIDTH
            video_output.height = VERTICAL_EXPORT_HEIGHT
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
        tracker = SubjectAwareCropper() if export_format != "original" else None
        horizontal_crop = (
            video_input.codec_context.width / video_input.codec_context.height
            > VERTICAL_EXPORT_WIDTH / VERTICAL_EXPORT_HEIGHT
        )
        video_frame_index = 0

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
                if packet.stream.index == video_input.index and export_format != "original":
                    caption = (
                        _active_caption(caption_cues or [], frame_time)
                        if export_format == "vertical_captions"
                        else None
                    )
                    original_pts = frame.pts
                    original_time_base = frame.time_base
                    frame = av.VideoFrame.from_ndarray(
                        _vertical_image(
                            frame,
                            caption,
                            tracker=tracker,
                            frame_index=video_frame_index,
                        ),
                        format="rgb24",
                    )
                    video_frame_index += 1
                    frame.pts = original_pts
                    frame.time_base = original_time_base
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
    if tracker is None:
        return {"schema": "creatorcut_reframing_v1", "mode": "original_framing"}
    return tracker.metadata(horizontal_crop=horizontal_crop)


class ProductProcessor:
    """Own heavyweight inference resources and process one uploaded video at a time."""

    def __init__(
        self,
        store: ProductStore,
        work_dir: Path,
        frozen_model_path: Path,
        model_cache: Path,
        semantic_cache: Path,
        serving_release_path: Path | None = None,
    ) -> None:
        self.store = store
        self.work_dir = work_dir
        self.frozen_model_path = frozen_model_path
        self.model_cache = model_cache
        self.semantic_cache = semantic_cache
        self.serving_release_path = serving_release_path
        self._whisper_model: Any | None = None
        self._resource_lock = threading.Lock()

    def serving_release_status(self) -> dict[str, Any]:
        """Return verified serving lineage, or an explicit unmanaged test state."""
        if self.serving_release_path is None:
            return {"status": "unmanaged_test_configuration"}
        return validate_serving_release(
            self.serving_release_path, self.frozen_model_path
        )

    def require_frozen_serving_release(self) -> dict[str, Any]:
        """Refuse production inference when the configured release does not verify."""
        release = self.serving_release_status()
        if release["status"] == "unmanaged_test_configuration":
            return release
        if release["status"] != "frozen":
            raise ValueError("The configured serving release is not frozen")
        return release

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

    def ensure_clip_semantic_embedding(self, clip_id: str) -> None:
        """Backfill the frozen semantic representation for a previously ranked clip."""
        self.require_frozen_serving_release()
        clip = self.store.get_clip(clip_id)
        if clip.get("semantic_embedding_json"):
            return
        frozen_model = json.loads(self.frozen_model_path.read_text(encoding="utf-8"))
        semantic_metadata = frozen_model["semantic_feature_metadata"]
        tokenizer_path, encoder_path = _download_model_files(
            semantic_metadata["model_id"],
            semantic_metadata["revision"],
            semantic_metadata["onnx_filename"],
            self.semantic_cache,
        )
        embedding = encode_texts_onnx(
            [clip["transcript_text"]],
            tokenizer_path,
            encoder_path,
            batch_size=1,
            maximum_length=semantic_metadata.get("maximum_length", DEFAULT_MAX_LENGTH),
        )[0]
        self.store.save_clip_semantic_embedding(
            clip_id, embedding.astype(float).tolist()
        )

    def create_repurposing_pack(self, clip_id: str) -> dict[str, Any]:
        """Create and persist transcript-grounded copy with creator-topic evidence."""
        clip = self.store.get_clip(clip_id)
        pack = generate_platform_pack(
            clip,
            self.store.creator_clip_documents(clip["creator_id"]),
            self.store.creator_summary(clip["creator_id"]),
        )
        return self.store.save_repurposing_pack(clip_id, pack)

    def process_video(self, video_id: str) -> None:
        """Run validation, ASR, candidate generation, frozen scoring, and personalization."""
        try:
            self.require_frozen_serving_release()
            source_path = self.store.media_path_for_video(video_id)
            backtest_context = self.store.backtest_source_for_video(video_id)
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
            if not candidates:
                raise ValueError("The video did not contain a usable spoken clip")
            assessed_candidates = [attach_publishability(candidate) for candidate in candidates]
            self.store.save_video_publishability_summary(
                video_id, summarize_publishability_batch(assessed_candidates)
            )
            candidates = [
                candidate
                for candidate in assessed_candidates
                if candidate["publishability"]["eligible"]
            ]
            if not candidates:
                blocked_count = len(assessed_candidates) - len(candidates)
                raise ValueError(
                    "The publishability gate did not leave a safe spoken clip "
                    f"({blocked_count} candidates were blocked)"
                )
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
            embedding_by_candidate_id = {
                record["candidate_id"]: embeddings[index].astype(float).tolist()
                for index, record in enumerate(queue)
            }
            scored = [
                {
                    **candidate_by_id[prediction["candidate_id"]],
                    "transcript_text": candidate_by_id[prediction["candidate_id"]]["text"],
                    "semantic_embedding": embedding_by_candidate_id[
                        prediction["candidate_id"]
                    ],
                    "global_score": prediction["predicted_quality_score"],
                    "predicted_targets": prediction["predicted_targets"],
                }
                for prediction in predictions
            ]
            global_order = sorted(
                scored,
                key=lambda candidate: (
                    -float(candidate["global_score"]),
                    candidate["candidate_id"],
                ),
            )
            global_rank_by_id = {
                candidate["candidate_id"]: rank
                for rank, candidate in enumerate(global_order, start=1)
            }
            scored = [
                {**candidate, "global_rank": global_rank_by_id[candidate["candidate_id"]]}
                for candidate in scored
            ]
            video = self.store.get_video(video_id)
            scored = select_multimodal_candidates(scored)
            try:
                delivery_by_candidate = extract_candidate_delivery_features(
                    source_path, scored
                )
            except Exception:
                delivery_by_candidate = {
                    candidate["candidate_id"]: {
                        "schema": DELIVERY_FEATURE_SCHEMA,
                        "status": "unavailable",
                        "audio_urgency_score": 0.5,
                        "visual_excitement_score": 0.5,
                    }
                    for candidate in scored
                }
            scored = [
                {
                    **candidate,
                    "multimodal_features": delivery_by_candidate[candidate["candidate_id"]],
                }
                for candidate in scored
            ]
            requested_plan = video.get("clip_plan", {})
            if not requested_plan.get("platforms"):
                requested_plan = validate_clip_plan({"youtube": None})
            resolved_plan = {
                **requested_plan,
                "platforms": {},
                "source_duration_seconds": media_duration,
            }
            ranked: list[dict[str, Any]] = []
            storage_rank = 1
            for platform, request in requested_plan["platforms"].items():
                if backtest_context is not None and backtest_context["role"] == "holdout":
                    personalized, _ = self.store.personalize_backtest_candidates(
                        backtest_context["experiment_id"], scored, platform
                    )
                else:
                    personalized, _ = self.store.personalize_candidates(
                        video["creator_id"], scored, platform
                    )
                    personalized, _ = self.store.apply_community_editorial(personalized)
                personalized, _ = self.store.apply_source_retention_signal(
                    video_id, personalized
                )
                personalized = [
                    {
                        **candidate,
                        "personalized_score": float(candidate["personalized_score"])
                        + float(candidate["publishability_adjustment"]),
                    }
                    for candidate in personalized
                ]
                if backtest_context is None or backtest_context["role"] != "holdout":
                    personalized, _ = self.store.apply_community_performance(
                        video["creator_id"], personalized, platform
                    )
                selected, plan = plan_platform_clips(
                    personalized,
                    platform,
                    media_duration,
                    request.get("requested_count"),
                )
                resolved_plan["platforms"][platform] = plan
                for platform_rank, clip in enumerate(selected, start=1):
                    ranked.append(
                        {
                            **clip,
                            "id": f"{video_id}_{platform}_clip_{platform_rank}",
                            "rank": storage_rank,
                            "platform_rank": platform_rank,
                            "ranking_model_version": frozen_model.get(
                                "freeze_schema", "unversioned_frozen_model"
                            ),
                            "explanation": (
                                explain_prediction(clip["predicted_targets"])
                                + " "
                                + explain_platform_fit(clip)
                                + " "
                                + publishability_summary(clip["publishability"])
                            ),
                        }
                    )
                    storage_rank += 1
            if not ranked:
                raise ValueError("No distinct publishable clips were available for this plan")
            self.store.save_video_clip_plan(video_id, resolved_plan)
            self.store.save_ranked_clips(video_id, ranked)
            self.store.update_video(video_id, "ready")
            if backtest_context is not None:
                try:
                    if backtest_context["role"] == "reference":
                        self.process_backtest_reference(video_id)
                    else:
                        self.store.freeze_backtest_predictions(
                            video_id, self.serving_release_status()
                        )
                except Exception as error:
                    self.store.mark_backtest_source(
                        backtest_context["experiment_id"],
                        backtest_context["source_key"],
                        "failed",
                        str(error),
                    )
        except Exception as error:  # background failures must become visible job state
            self.store.update_video(video_id, "failed", error_message=str(error)[:500])
            context = self.store.backtest_source_for_video(video_id)
            if context is not None:
                self.store.mark_backtest_source(
                    context["experiment_id"],
                    context["source_key"],
                    "failed",
                    str(error),
                )
            raise

    def create_historical_clips(
        self,
        video_id: str,
        actuals: list[dict[str, Any]],
        *,
        origin: str = "historical_reference",
    ) -> dict[str, str]:
        """Score aligned organization-selected intervals in one frozen-model batch."""
        if not actuals:
            return {}
        video = self.store.get_video(video_id)
        source_path = self.store.media_path_for_video(video_id)
        media_duration = float(video.get("duration_seconds") or 0.0)
        transcript_path = self.work_dir / video_id / "transcript.json"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        queue = []
        candidates = []
        for actual in actuals:
            start = float(actual["source_start_seconds"])
            end = float(actual["source_end_seconds"])
            if start < 0 or end <= start or end > media_duration or end - start > 180:
                raise ValueError(f"{actual['content_id']}: aligned interval is invalid")
            text = transcript_text_for_interval(transcript, start, end)
            if not text:
                raise ValueError(f"{actual['content_id']}: aligned interval has no speech")
            candidate_id = f"{video_id}_{actual['id']}_historical"
            queue.append(
                {
                    "annotation_id": f"{candidate_id}_production",
                    "candidate_id": candidate_id,
                    "video_id": video_id,
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": end - start,
                    "transcript_text": text,
                    "labels": {},
                }
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": end - start,
                    "transcript_text": text,
                    "word_count": len(text.split()),
                }
            )
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
        artifact = build_embedding_artifact(
            queue,
            embeddings,
            model_id=semantic_metadata["model_id"],
            revision=semantic_metadata["revision"],
            onnx_filename=semantic_metadata["onnx_filename"],
            maximum_length=semantic_metadata.get("maximum_length", DEFAULT_MAX_LENGTH),
        )
        predictions = score_frozen_model(queue, artifact, frozen_model)["records"]
        delivery = extract_candidate_delivery_features(source_path, candidates)
        saved: dict[str, str] = {}
        for index, (actual, candidate, prediction) in enumerate(
            zip(actuals, candidates, predictions, strict=True)
        ):
            prepared = attach_publishability(
                {
                    **candidate,
                    "semantic_embedding": embeddings[index].astype(float).tolist(),
                    "global_score": prediction["predicted_quality_score"],
                    "predicted_targets": prediction["predicted_targets"],
                    "multimodal_features": delivery[candidate["candidate_id"]],
                    "personalized_score": prediction["predicted_quality_score"],
                    "editorial_adjustment": 0.0,
                    "performance_adjustment": 0.0,
                    "semantic_performance_adjustment": 0.0,
                    "source_retention_adjustment": 0.0,
                }
            )
            clip = {
                **attach_platform_scores([prepared], "youtube")[0],
                "platform": "youtube",
                "platform_rank": None,
                "ranking_model_version": frozen_model.get(
                    "freeze_schema", "unversioned_frozen_model"
                ),
                "explanation": (
                    "Organization-selected historical interval. Stored as reference evidence."
                ),
            }
            saved[actual["id"]] = self.store.save_custom_clip(
                video_id,
                clip,
                origin=origin,
                event_type="historical_selected",
            )
        return saved

    def process_backtest_reference(self, video_id: str) -> None:
        """Align uploaded historical Shorts and register eligible outcome evidence."""
        context = self.store.backtest_source_for_video(video_id)
        if context is None or context["role"] != "reference":
            return
        pending = self.store.pending_backtest_actual_clips(
            context["experiment_id"], context["source_key"]
        )
        if not pending:
            self.store.mark_backtest_source(
                context["experiment_id"],
                context["source_key"],
                "failed",
                "No matching Short files were uploaded",
            )
            return
        source_path = self.store.media_path_for_video(video_id)
        envelope = audio_envelope(source_path)
        aligned = []
        for actual in pending:
            alignment = align_short_to_source(
                source_path, Path(actual["media_path"]), envelope
            )
            self.store.save_backtest_alignment(actual["id"], alignment)
            if alignment["status"] == "aligned":
                aligned.append({**actual, **alignment})
        saved = self.create_historical_clips(video_id, aligned)
        for actual in aligned:
            clip_id = saved[actual["id"]]
            self.store.save_backtest_alignment(actual["id"], actual, clip_id)
            self.store.save_performance_report(
                clip_id,
                {
                    "platform": "youtube",
                    "views": actual.get("views"),
                    "likes": actual.get("likes"),
                    "comments": actual.get("comments"),
                },
            )
        warning_count = len(pending) - len(aligned)
        self.store.mark_backtest_source(
            context["experiment_id"],
            context["source_key"],
            "ready_with_warnings" if warning_count else "ready",
            f"{warning_count} Short alignments need review" if warning_count else None,
        )

    def evaluate_backtest_holdout(self, experiment_id: str) -> dict[str, Any]:
        """Align revealed holdout Shorts and score the frozen recommendation snapshot."""
        experiment = self.store.get_backtest_experiment(experiment_id)
        if not experiment["holdout"]["predictions_frozen"]:
            raise ValueError("Generate and freeze test-video recommendations first")
        holdout_source = next(
            source for source in experiment["sources"] if source["role"] == "holdout"
        )
        video_id = holdout_source["video_id"]
        if not video_id:
            raise ValueError("The test video has not been uploaded")
        pending = self.store.pending_backtest_actual_clips(
            experiment_id, experiment["holdout_key"]
        )
        if not pending:
            raise ValueError("Upload the organization-selected test Shorts")
        source_path = self.store.media_path_for_video(video_id)
        envelope = audio_envelope(source_path)
        for actual in pending:
            alignment = align_short_to_source(
                source_path, Path(actual["media_path"]), envelope
            )
            self.store.save_backtest_alignment(actual["id"], alignment)
        refreshed = self.store.get_backtest_experiment(experiment_id)
        evaluation = evaluate_backtest(
            refreshed["prediction_snapshot"]["recommendations"],
            refreshed["holdout"]["actual_clips"],
        )
        self.store.save_backtest_evaluation(experiment_id, evaluation)
        return self.store.get_backtest_experiment(experiment_id)

    def correct_backtest_alignment(
        self,
        actual_id: str,
        creator_id: str,
        start_seconds: float,
        end_seconds: float,
    ) -> dict[str, Any]:
        """Apply a human-reviewed source interval and refresh calibration or evaluation."""
        actual = self.store.get_backtest_actual_clip(actual_id, creator_id)
        video = self.store.get_video(actual["video_id"])
        start = float(start_seconds)
        end = float(end_seconds)
        if start < 0 or end <= start or end > float(video["duration_seconds"]):
            raise ValueError("The corrected interval falls outside the source video")
        alignment = {
            "status": "aligned",
            "source_start_seconds": round(start, 3),
            "source_end_seconds": round(end, 3),
            "confidence": 1.0,
            "method": "manual",
            "compound_edit_detected": False,
            "segments": [],
        }
        clip_id = actual.get("model_clip_id")
        if actual["role"] == "reference" and clip_id is None:
            saved = self.create_historical_clips(
                actual["video_id"], [{**actual, **alignment}]
            )
            clip_id = saved[actual["id"]]
            self.store.save_performance_report(
                clip_id,
                {
                    "platform": "youtube",
                    "views": actual.get("views"),
                    "likes": actual.get("likes"),
                    "comments": actual.get("comments"),
                },
            )
        self.store.save_backtest_alignment(
            actual["id"], alignment, clip_id, manual=True
        )
        experiment = self.store.get_backtest_experiment(actual["experiment_id"])
        if actual["role"] == "holdout" and experiment["prediction_snapshot"]:
            evaluation = evaluate_backtest(
                experiment["prediction_snapshot"]["recommendations"],
                self.store.get_backtest_experiment(actual["experiment_id"])["holdout"][
                    "actual_clips"
                ],
            )
            self.store.save_backtest_evaluation(actual["experiment_id"], evaluation)
        return self.store.get_backtest_experiment(actual["experiment_id"], creator_id)

    def create_custom_clip(
        self,
        video_id: str,
        start_seconds: float,
        end_seconds: float,
        platform: str = "youtube",
    ) -> str:
        """Score and persist a creator-authored interval as explicit preference evidence."""
        self.require_frozen_serving_release()
        video = self.store.get_video(video_id)
        if video["status"] != "ready" or not video.get("duration_seconds"):
            raise ValueError("The source video must finish processing first")
        start, end = validate_custom_interval(
            start_seconds, end_seconds, float(video["duration_seconds"])
        )
        transcript_path = self.work_dir / video_id / "transcript.json"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        text = transcript_text_for_interval(transcript, start, end)
        if not text:
            raise ValueError("The custom interval does not contain transcribed speech")

        frozen_model = json.loads(self.frozen_model_path.read_text(encoding="utf-8"))
        semantic_metadata = frozen_model["semantic_feature_metadata"]
        tokenizer_path, encoder_path = _download_model_files(
            semantic_metadata["model_id"],
            semantic_metadata["revision"],
            semantic_metadata["onnx_filename"],
            self.semantic_cache,
        )
        candidate_id = f"{video_id}_custom_candidate"
        queue = [
            {
                "annotation_id": f"{candidate_id}_production",
                "candidate_id": candidate_id,
                "video_id": video_id,
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": end - start,
                "transcript_text": text,
                "labels": {},
            }
        ]
        embeddings = encode_texts_onnx(
            [text],
            tokenizer_path,
            encoder_path,
            batch_size=1,
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
        prediction = score_frozen_model(queue, embedding_artifact, frozen_model)["records"][0]
        candidate = {
            "candidate_id": candidate_id,
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": end - start,
            "transcript_text": text,
            "semantic_embedding": embeddings[0].astype(float).tolist(),
            "global_score": prediction["predicted_quality_score"],
            "predicted_targets": prediction["predicted_targets"],
            "multimodal_features": {
                "schema": DELIVERY_FEATURE_SCHEMA,
                "status": "not_extracted_for_custom_interval",
                "audio_urgency_score": 0.5,
                "visual_excitement_score": 0.5,
            },
        }
        candidate = attach_publishability(candidate)
        personalized, _ = self.store.personalize_candidates(
            video["creator_id"], [candidate], platform
        )
        personalized, _ = self.store.apply_community_editorial(personalized)
        adjusted, _ = self.store.apply_source_retention_signal(video_id, personalized)
        prepared = {
            **adjusted[0],
            "personalized_score": float(adjusted[0]["personalized_score"])
            + float(adjusted[0]["publishability_adjustment"]),
        }
        community_adjusted, _ = self.store.apply_community_performance(
            video["creator_id"], [prepared], platform
        )
        prepared = community_adjusted[0]
        clip = {
            **attach_platform_scores([prepared], platform)[0],
            "platform": platform,
            "platform_rank": None,
            "ranking_model_version": frozen_model.get(
                "freeze_schema", "unversioned_frozen_model"
            ),
            "explanation": (
                "Creator-defined interval. Scored for evidence, not selected by the model. "
                + f"Saved for {PLATFORM_PROFILES[platform]['label']}."
            ),
        }
        return self.store.save_custom_clip(video_id, clip)

    def export_clip(
        self,
        clip_id: str,
        start: float,
        end: float,
        export_format: str = "original",
    ) -> tuple[Path, dict[str, Any]]:
        """Validate and cache one original or adjusted clip download."""
        clip = self.store.get_clip(clip_id)
        video = self.store.get_video(clip["video_id"])
        checked_start, checked_end = validate_export_interval(
            clip,
            start,
            end,
            float(video["duration_seconds"]),
        )
        if not isinstance(export_format, str) or export_format not in SUPPORTED_EXPORT_FORMATS:
            raise ValueError("Choose a supported export format")
        export_dir = self.work_dir / "exports"
        format_version = (
            export_format
            if export_format == "original"
            else f"{export_format}_face_track_v1"
        )
        export_name = (
            f"{clip_id}_{round(checked_start * 1000)}_{round(checked_end * 1000)}_"
            f"{format_version}.mp4"
        )
        output_path = export_dir / export_name
        metadata_path = output_path.with_suffix(".metadata.json")
        if not output_path.exists():
            cues: list[dict[str, Any]] = []
            if export_format == "vertical_captions":
                transcript_path = self.work_dir / clip["video_id"] / "transcript.json"
                transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
                cues = build_caption_cues(transcript, checked_start, checked_end)
            reframe_metadata = transcode_clip(
                Path(clip["media_path"]),
                output_path,
                checked_start,
                checked_end,
                export_format,
                cues,
            )
            metadata_path.write_text(
                json.dumps(reframe_metadata, indent=2), encoding="utf-8"
            )
        elif metadata_path.is_file():
            reframe_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            reframe_metadata = {
                "schema": "creatorcut_reframing_v1",
                "mode": "cached_export_without_tracking_metadata",
            }
        return output_path, reframe_metadata
