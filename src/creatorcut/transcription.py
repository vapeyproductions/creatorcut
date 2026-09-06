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
    model: Any,
    video_id: str,
    video_path: Path,
    output_path: Path,
    model_name: str,
    compute_type: str,
) -> dict[str, Any]:
    """Transcribe one video and persist segment- and word-level timestamps."""
    started_at = time.monotonic()
    segment_generator, info = model.transcribe(
        str(video_path),
        language="en",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
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
            "vad_filter": True,
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
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    manifest = load_manifest(args.manifest)
    selected = [
        item for item in manifest if not args.video_ids or item["video_id"] in args.video_ids
    ]
    if not selected:
        raise SystemExit("No manifest videos matched --video-id")

    model_cache = Path("artifacts/models")
    print(f"Loading {args.model} on CPU with {args.compute_type} compute")
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type=args.compute_type,
        download_root=str(model_cache),
    )

    for item in selected:
        video_id = item["video_id"]
        video_path = Path(item["local_filename"])
        output_path = args.output_dir / f"{video_id}.json"
        if output_path.exists() and not args.force:
            print(f"Skipping {video_id}; {output_path} already exists")
            continue

        print(f"Transcribing {video_id} from {video_path}")
        result = transcribe_video(
            model,
            video_id,
            video_path,
            output_path,
            args.model,
            args.compute_type,
        )
        word_count = sum(len(segment["words"]) for segment in result["segments"])
        print(
            f"Finished {video_id}: {len(result['segments'])} segments, "
            f"{word_count} words, {result['elapsed_seconds']:.1f}s"
        )
