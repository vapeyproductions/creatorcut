from fractions import Fraction

import av
import numpy as np
import pytest

from creatorcut.product_pipeline import (
    build_caption_cues,
    export_dimensions,
    prefilter_candidates,
    select_diverse_top_clips,
    transcode_clip,
    validate_export_interval,
)


def candidate(candidate_id, start, end, score=3.0):
    return {
        "candidate_id": candidate_id,
        "video_id": "upload_test",
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": end - start,
        "sentence_count": 3,
        "word_count": 60,
        "text": "Here is a complete and useful idea that you need to know.",
        "personalized_score": score,
        "global_score": score,
    }


def test_select_diverse_top_clips_avoids_repeating_the_same_moment():
    candidates = [
        candidate("first", 0.0, 40.0, 5.0),
        candidate("overlap", 2.0, 42.0, 4.9),
        candidate("second", 60.0, 100.0, 4.5),
        candidate("third", 120.0, 160.0, 4.0),
    ]

    selected = select_diverse_top_clips(candidates)

    assert [item["candidate_id"] for item in selected] == ["first", "second", "third"]


def test_prefilter_candidates_is_deterministic_and_limited():
    candidates = [
        candidate(f"candidate_{index:03d}", index * 5.0, index * 5.0 + 30.0)
        for index in range(80)
    ]

    first = prefilter_candidates(candidates, limit=20)
    second = prefilter_candidates(candidates, limit=20)

    assert len(first) == 20
    assert [item["candidate_id"] for item in first] == [item["candidate_id"] for item in second]
    assert min(item["start_seconds"] for item in first) < 40
    assert max(item["start_seconds"] for item in first) > 300


def test_validate_export_interval_accepts_a_small_edit():
    clip = candidate("clip", 30.0, 70.0)

    assert validate_export_interval(clip, 31.2, 67.8, 120.0) == (31.2, 67.8)


def test_build_caption_cues_uses_word_timestamps_and_clip_bounds():
    transcript = {
        "segments": [
            {
                "words": [
                    {"start": 9.8, "end": 10.2, "word": " Before"},
                    {"start": 10.2, "end": 10.6, "word": " this"},
                    {"start": 10.6, "end": 11.0, "word": " works."},
                    {"start": 11.2, "end": 11.6, "word": " Next"},
                    {"start": 11.6, "end": 12.0, "word": " idea."},
                ]
            }
        ]
    }

    cues = build_caption_cues(transcript, 10.0, 11.8)

    assert cues == [
        {"start": 10.0, "end": 11.08, "text": "Before this works."},
        {"start": 11.2, "end": 11.8, "text": "Next idea."},
    ]


def test_transcode_clip_creates_vertical_captioned_mp4(tmp_path):
    source_path = tmp_path / "source.mp4"
    output_path = tmp_path / "vertical.mp4"
    with av.open(str(source_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = 320
        stream.height = 180
        stream.pix_fmt = "yuv420p"
        for index in range(10):
            pixels = np.full((180, 320, 3), (80, 130, 180), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    transcode_clip(
        source_path,
        output_path,
        0.0,
        0.9,
        "vertical_captions",
        [{"start": 0.0, "end": 0.9, "text": "A useful caption"}],
    )

    with av.open(str(output_path)) as container:
        video = container.streams.video[0]
        frames = list(container.decode(video))
    assert (video.codec_context.width, video.codec_context.height) == (720, 1280)
    assert frames
    pixels = frames[len(frames) // 2].to_ndarray(format="rgb24")
    assert int(pixels[850:1100].min()) < 20


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [(3840, 2160, (1920, 1080)), (1080, 1920, (1080, 1920)), (1280, 720, (1280, 720))],
)
def test_export_dimensions_preserve_aspect_ratio_within_social_video_limits(
    width, height, expected
):
    assert export_dimensions(width, height) == expected


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (14.9, 70.0, "edited start"),
        (30.0, 85.1, "edited end"),
        (30.0, 125.0, "outside the uploaded video"),
    ],
)
def test_validate_export_interval_rejects_unsafe_edits(start, end, message):
    clip = candidate("clip", 30.0, 70.0)

    with pytest.raises(ValueError, match=message):
        validate_export_interval(clip, start, end, 120.0)
