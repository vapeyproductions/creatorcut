"""Inspect source media before it enters the ML pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import av
from av.stream import Disposition


def rational_as_float(value: Any) -> float | None:
    """Convert an optional PyAV rational value to a JSON-safe float."""
    return float(value) if value is not None else None


def inspect_media(path: Path) -> dict[str, Any]:
    """Return container, video, audio, and validation metadata."""
    with av.open(str(path)) as container:
        duration_seconds = (
            container.duration / av.time_base if container.duration is not None else None
        )
        video_streams = [
            {
                "index": stream.index,
                "codec": stream.codec_context.name,
                "width": stream.codec_context.width,
                "height": stream.codec_context.height,
                "average_frame_rate": rational_as_float(stream.average_rate),
                "attached_picture": bool(stream.disposition & Disposition.attached_pic),
            }
            for stream in container.streams.video
        ]
        audio_streams = [
            {
                "index": stream.index,
                "codec": stream.codec_context.name,
                "sample_rate": stream.codec_context.sample_rate,
                "channels": stream.codec_context.channels,
            }
            for stream in container.streams.audio
        ]

    errors: list[str] = []
    warnings: list[str] = []
    content_video_streams = [stream for stream in video_streams if not stream["attached_picture"]]
    if not content_video_streams:
        errors.append("No video stream")
    if not audio_streams:
        errors.append("No audio stream")
    if duration_seconds is None or duration_seconds <= 0:
        errors.append("Container duration is unavailable or invalid")
    if content_video_streams and min(stream["height"] for stream in content_video_streams) < 720:
        warnings.append("Video resolution is below 720p; visual feature quality may be limited")

    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "duration_seconds": duration_seconds,
        "video_streams": video_streams,
        "audio_streams": audio_streams,
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def main() -> None:
    """Inspect one or more media files from the command line."""
    parser = argparse.ArgumentParser(description="Inspect CreatorCut source media")
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()

    results = [inspect_media(path) for path in args.paths]
    print(json.dumps(results, indent=2))
    if any(not result["valid"] for result in results):
        raise SystemExit(1)
