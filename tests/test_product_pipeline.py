import json
from fractions import Fraction

import av
import numpy as np
import pytest

import creatorcut.product_pipeline as product_pipeline
from creatorcut.product_pipeline import (
    ProductProcessor,
    build_caption_cues,
    export_dimensions,
    prefilter_candidates,
    select_diverse_top_clips,
    transcode_clip,
    transcript_text_for_interval,
    validate_custom_interval,
    validate_export_interval,
)
from creatorcut.product_store import ProductStore


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


def test_validate_custom_interval_accepts_any_safe_source_interval():
    assert validate_custom_interval(91.2, 126.8, 180.0) == (91.2, 126.8)

    with pytest.raises(ValueError, match="between 5 and 90"):
        validate_custom_interval(10.0, 13.0, 180.0)


def test_processor_scores_and_saves_creator_authored_interval(tmp_path, monkeypatch):
    store = ProductStore(tmp_path / "product.sqlite")
    creator = store.ensure_creator("Example", "creator_example")
    source = tmp_path / "source.mp4"
    source.touch()
    video = store.create_video(creator["id"], source.name, source)
    store.update_video(video["id"], "ready", duration_seconds=60.0)
    work_dir = tmp_path / "work"
    transcript_dir = work_dir / video["id"]
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "transcript.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "words": [
                            {"start": 10.0, "end": 10.5, "word": "Custom"},
                            {"start": 10.5, "end": 11.0, "word": "story"},
                            {"start": 11.0, "end": 11.5, "word": "works."},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    frozen_model_path = tmp_path / "model.json"
    frozen_model_path.write_text(
        json.dumps(
            {
                "freeze_schema": "creatorcut_ranker_freeze_v1",
                "semantic_feature_metadata": {
                    "model_id": "example",
                    "revision": "fixed",
                    "onnx_filename": "model.onnx",
                    "maximum_length": 256,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        product_pipeline,
        "_download_model_files",
        lambda *args, **kwargs: (tmp_path / "tokenizer.json", tmp_path / "model.onnx"),
    )
    monkeypatch.setattr(
        product_pipeline,
        "encode_texts_onnx",
        lambda *args, **kwargs: np.asarray([[0.6, 0.8]]),
    )
    monkeypatch.setattr(product_pipeline, "build_embedding_artifact", lambda *a, **k: {})
    monkeypatch.setattr(
        product_pipeline,
        "score_frozen_model",
        lambda *args, **kwargs: {
            "records": [
                {
                    "predicted_quality_score": 4.1,
                    "predicted_targets": {
                        "hook": 4.0,
                        "completeness": 4.2,
                        "payoff": 4.1,
                        "clarity": 4.1,
                    },
                }
            ]
        },
    )
    processor = ProductProcessor(
        store,
        work_dir,
        frozen_model_path,
        tmp_path / "model-cache",
        tmp_path / "semantic-cache",
    )

    clip_id = processor.create_custom_clip(video["id"], 9.5, 15.0)
    clip = store.get_clip(clip_id)

    assert clip["origin"] == "creator"
    assert clip["transcript_text"] == "Custom story works."
    assert clip["global_score"] == pytest.approx(4.1)


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
    assert transcript_text_for_interval(transcript, 10.0, 11.8) == (
        "Before this works. Next idea."
    )


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

    metadata = transcode_clip(
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
    assert metadata["schema"] == "creatorcut_reframing_v1"
    assert metadata["mode"] == "center_crop_fallback"
    assert metadata["sampled_frame_count"] > 0


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
