"""Timestamped video transcription with faster-whisper."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from creatorcut.dataset import load_manifest


def serialize_segment(segment: Any) -> dict[str, Any]:
    """Convert a faster-whisper segment into stable JSON data."""
    words = [
        {
            "start": word.start,
            "end": word.end,
            "word": word.word,
            "probability": word.probability,
        }
        for word in (segment.words or [])
    ]
    return {
        "id": segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text.strip(),
        "avg_logprob": segment.avg_logprob,
        "no_speech_probability": segment.no_speech_prob,
        "words": words,
    }


def transcribe_video(
    transcriber: Any,
    video_id: str,
    video_path: Path,
    output_path: Path,
    model_name: str,
    compute_type: str,
    batch_size: int,
    vad_filter: bool,
) -> dict[str, Any]:
    """Transcribe one video and persist segment- and word-level timestamps."""
    started_at = time.monotonic()
    transcription_options = {
        "language": "en",
        "beam_size": 5,
        "word_timestamps": True,
        "vad_filter": vad_filter,
    }
    if vad_filter:
        transcription_options["vad_parameters"] = {"min_silence_duration_ms": 500}
    if batch_size > 1:
        transcription_options["batch_size"] = batch_size
    segment_generator, info = transcriber.transcribe(str(video_path), **transcription_options)
    segments = [serialize_segment(segment) for segment in segment_generator]
    transcript = {
        "schema_version": 1,
        "video_id": video_id,
        "source_path": video_path.as_posix(),
        "created_at": datetime.now(UTC).isoformat(),
        "model": {
            "library": "faster-whisper",
            "name": model_name,
            "device": "cpu",
            "compute_type": compute_type,
            "beam_size": 5,
            "batch_size": batch_size,
            "inference_mode": "batched" if batch_size > 1 else "sequential",
            "vad_filter": vad_filter,
        },
        "language": info.language,
        "language_probability": info.language_probability,
        "media_duration_seconds": info.duration,
        "speech_duration_seconds": info.duration_after_vad,
        "elapsed_seconds": time.monotonic() - started_at,
        "segments": segments,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)
    return transcript


def main() -> None:
    """Transcribe videos listed in the project manifest."""
    parser = argparse.ArgumentParser(description="Transcribe CreatorCut source videos")
    parser.add_argument("--manifest", type=Path, default=Path("data/videos.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/transcripts"))
    parser.add_argument(
        "--media-dir",
        type=Path,
        help="Optional local directory containing files named like the manifest entries",
    )
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument(
        "--disable-vad",
        action="store_true",
        help="Decode the full audio track; requires --batch-size 1",
    )
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    manifest = load_manifest(args.manifest)
    selected = [
        item for item in manifest if not args.video_ids or item["video_id"] in args.video_ids
    ]
    if not selected:
        raise SystemExit("No manifest videos matched --video-id")
    if args.disable_vad and args.batch_size > 1:
        raise SystemExit("--disable-vad requires --batch-size 1 for long-form audio")

    model_cache = Path("artifacts/models")
    print(f"Loading {args.model} on CPU with {args.compute_type} compute")
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type=args.compute_type,
        download_root=str(model_cache),
        cpu_threads=args.cpu_threads,
    )
    transcriber = BatchedInferencePipeline(model=model) if args.batch_size > 1 else model

    for item in selected:
        video_id = item["video_id"]
        manifest_video_path = Path(item["local_filename"])
        video_path = (
            args.media_dir / manifest_video_path.name if args.media_dir else manifest_video_path
        )
        output_path = args.output_dir / f"{video_id}.json"
        if output_path.exists() and not args.force:
            print(f"Skipping {video_id}; {output_path} already exists")
            continue

        print(f"Transcribing {video_id} from {video_path}")
        result = transcribe_video(
            transcriber,
            video_id,
            video_path,
            output_path,
            args.model,
            args.compute_type,
            args.batch_size,
            not args.disable_vad,
        )
        word_count = sum(len(segment["words"]) for segment in result["segments"])
        print(
            f"Finished {video_id}: {len(result['segments'])} segments, "
            f"{word_count} words, {result['elapsed_seconds']:.1f}s"
        )
