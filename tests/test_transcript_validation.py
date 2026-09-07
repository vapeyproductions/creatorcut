import json

from creatorcut.transcript_validation import validate_transcript, validate_transcript_directory


def valid_transcript(video_id="video_001"):
    return {
        "video_id": video_id,
        "media_duration_seconds": 12.0,
        "segments": [
            {
                "start": 1.0,
                "end": 3.0,
                "words": [
                    {"start": 1.0, "end": 1.5, "word": " Hello", "probability": 0.9},
                    {"start": 1.6, "end": 2.0, "word": " world.", "probability": 0.8},
                ],
            }
        ],
    }


def test_validate_transcript_accepts_ordered_timestamped_words():
    result = validate_transcript(valid_transcript())

    assert result["segment_count"] == 1
    assert result["word_count"] == 2
    assert result["warnings"] == []
    assert result["errors"] == []


def test_validate_transcript_rejects_out_of_order_words():
    transcript = valid_transcript()
    transcript["segments"][0]["words"][1]["start"] = 0.5

    result = validate_transcript(transcript)

    assert any("not ordered" in error for error in result["errors"])


def test_validate_transcript_warns_on_low_confidence_and_repetition():
    transcript = valid_transcript()
    repeated = " one two three four" * 9
    transcript["segments"][0]["text"] = repeated
    for word in transcript["segments"][0]["words"]:
        word["probability"] = 0.4

    result = validate_transcript(transcript)

    assert result["mean_word_probability"] == 0.4
    assert result["low_confidence_word_fraction"] == 1.0
    assert result["repetitive_segment_count"] == 1
    assert any("mean word confidence" in warning for warning in result["warnings"])
    assert any("repeated" in warning for warning in result["warnings"])


def test_validate_transcript_directory_reports_duplicate_ids(tmp_path):
    for name in ("first.json", "second.json"):
        (tmp_path / name).write_text(json.dumps(valid_transcript("video_001")), encoding="utf-8")

    result = validate_transcript_directory(tmp_path)

    assert result["transcript_count"] == 2
    assert result["total_words"] == 4
    assert any("duplicate video IDs" in error for error in result["errors"])
