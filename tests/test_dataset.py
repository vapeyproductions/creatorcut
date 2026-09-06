import json
from pathlib import Path

from creatorcut.dataset import transcript_text_for_interval, validate_dataset


def test_transcript_text_for_interval_uses_overlapping_words() -> None:
    transcript = {
        "segments": [
            {
                "words": [
                    {"start": 0.0, "end": 0.5, "word": " Hello"},
                    {"start": 0.5, "end": 1.0, "word": " world"},
                    {"start": 1.0, "end": 1.5, "word": " again"},
                ]
            }
        ]
    }

    text, count = transcript_text_for_interval(transcript, 0.45, 1.05)

    assert text == "Hello world again"
    assert count == 3


def test_validate_dataset_reports_constant_score(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    manifest_path = tmp_path / "videos.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "video_id": "video_001",
                    "source_url": "https://example.com/video",
                    "local_filename": "video.mp4",
                }
            ]
        ),
        encoding="utf-8",
    )
    annotations_path = tmp_path / "labels.jsonl"
    annotations_path.write_text(
        json.dumps(
            {
                "annotation_id": "clip_001",
                "video_id": "video_001",
                "start_seconds": 1.0,
                "end_seconds": 2.0,
                "hook": 5,
                "completeness": 5,
                "payoff": 5,
                "clarity": 5,
                "presentation": 5,
                "technically_exportable": True,
                "notes": "Good clip",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_dataset(manifest_path, annotations_path, tmp_path)

    assert result["errors"] == []
    assert "presentation has no label variance: [5]" in result["warnings"]
