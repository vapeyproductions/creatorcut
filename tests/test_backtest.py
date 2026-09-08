import csv
import io

import pytest

from creatorcut.backtest import (
    _match_envelope,
    evaluate_backtest,
    parse_backtest_tracker,
)
from creatorcut.product_app import ProductApplication
from creatorcut.product_store import ProductStore


def tracker_csv(reference_clips=3, holdout_clips=2):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Link", "Views", "Likes", "Comments", "video ID"])
    for source_key in ("training_vid_1", "training_vid_2", "test_vid"):
        writer.writerow([f"https://example.test/{source_key}", "1.2M", "17k", 100, source_key])
        count = holdout_clips if source_key == "test_vid" else reference_clips
        for index in range(1, count + 1):
            writer.writerow(
                [
                    f"https://example.test/{source_key}_clip{index}",
                    f"{40 + index * 10}k",
                    500 + index * 100,
                    10 + index,
                    f"{source_key}_clip{index}",
                ]
            )
    return output.getvalue().encode()


def parsed_tracker(reference_clips=3, holdout_clips=2):
    return parse_backtest_tracker(
        "tracker.csv",
        tracker_csv(reference_clips, holdout_clips),
        ["training_vid_1", "training_vid_2"],
        "test_vid",
    )


def stored_clip(index, score):
    return {
        "start_seconds": index * 20.0,
        "end_seconds": index * 20.0 + 30.0,
        "duration_seconds": 30.0,
        "transcript_text": f"Useful historical topic number {index}.",
        "semantic_embedding": [1.0, float(index), 0.5],
        "global_score": score,
        "personalized_score": score,
        "predicted_targets": {
            "hook": score,
            "completeness": 4.0,
            "payoff": score,
            "clarity": 4.0,
        },
        "explanation": "Historical evidence.",
        "ranking_model_version": "test_frozen",
        "platform": "youtube",
        "platform_score": score,
        "multimodal_features": {
            "audio_urgency_score": index / 10,
            "visual_excitement_score": index / 10,
        },
    }


def candidate(index):
    score = 3.0 + index / 10
    return {
        "candidate_id": f"candidate_{index}",
        "start_seconds": index * 40.0,
        "end_seconds": index * 40.0 + 30.0,
        "duration_seconds": 30.0,
        "global_score": score,
        "predicted_targets": {
            "hook": score,
            "completeness": 4.0,
            "payoff": score,
            "clarity": 4.0,
        },
        "semantic_embedding": [1.0, float(index), 0.5],
        "multimodal_features": {
            "audio_urgency_score": index / 10,
            "visual_excitement_score": index / 10,
        },
    }


def test_tracker_parser_partitions_and_expands_compact_counts():
    tracker = parsed_tracker()

    assert tracker["clip_counts"] == {
        "training_vid_1": 3,
        "training_vid_2": 3,
        "test_vid": 2,
    }
    assert tracker["source_rows"]["training_vid_1"]["views"] == 1_200_000
    assert next(
        row for row in tracker["clip_rows"] if row["content_id"] == "test_vid_clip1"
    )["role"] == "holdout"


def test_fft_alignment_finds_known_offset():
    rng = pytest.importorskip("numpy").random.default_rng(7)
    source = rng.normal(size=4_000)
    template = source[1_250:1_850] + rng.normal(scale=0.01, size=600)

    index, score, competing = _match_envelope(source, template)

    assert index == 1_250
    assert score > 0.99
    assert score > competing


def test_store_quarantines_holdout_until_predictions_are_frozen(tmp_path):
    store = ProductStore(tmp_path / "creatorcut.sqlite")
    creator = store.ensure_creator("Test channel")
    experiment = store.create_backtest_experiment(
        creator["id"], "Recommendation test", "tracker.csv", parsed_tracker()
    )

    assert experiment["holdout"]["actuals_revealed"] is False
    assert experiment["holdout"]["actual_clips"] == []

    reference_rows = experiment["reference_clips"]
    for source_key in experiment["reference_keys"]:
        source = tmp_path / f"{source_key}.mp4"
        source.touch()
        video = store.create_video(creator["id"], source.name, source)
        store.attach_backtest_source(
            experiment["id"], creator["id"], source_key, "reference", video["id"]
        )
        for row in [item for item in reference_rows if item["source_key"] == source_key]:
            store.save_backtest_alignment(
                row["id"],
                {
                    "status": "aligned",
                    "source_start_seconds": 10.0,
                    "source_end_seconds": 40.0,
                    "confidence": 0.9,
                    "method": "test",
                    "segments": [],
                },
            )
        store.mark_backtest_source(experiment["id"], source_key, "ready")

    experiment = store.get_backtest_experiment(experiment["id"], creator["id"])
    assert experiment["references_ready"] is True
    test_source = tmp_path / "test_vid.mp4"
    test_source.touch()
    video = store.create_video(
        creator["id"], test_source.name, test_source, {"platforms": {"youtube": {}}}
    )
    store.attach_backtest_source(
        experiment["id"], creator["id"], "test_vid", "holdout", video["id"]
    )
    store.save_ranked_clips(
        video["id"],
        [
            {
                **stored_clip(1, 4.1),
                "id": f"{video['id']}_youtube_clip_1",
                "rank": 1,
                "platform_rank": 1,
                "backtest_lineage": {"estimated_relative_performance": 72.0},
            }
        ],
    )
    store.freeze_backtest_predictions(video["id"], {"status": "frozen"})

    revealed = store.get_backtest_experiment(experiment["id"], creator["id"])
    assert revealed["holdout"]["actuals_revealed"] is True
    assert len(revealed["holdout"]["actual_clips"]) == 2
    assert revealed["prediction_snapshot"]["holdout_outcomes_used"] is False


def test_staged_backtest_video_is_invisible_to_worker_until_enqueue(tmp_path):
    store = ProductStore(tmp_path / "creatorcut.sqlite")
    creator = store.ensure_creator("Test channel")
    source = tmp_path / "training_vid_1.mp4"
    source.touch()

    video = store.create_video(
        creator["id"], source.name, source, enqueue_processing=False
    )

    assert video["status"] == "staging"
    assert store.claim_next_processing_job("worker", 120) is None
    store.enqueue_video(video["id"])
    claimed = store.claim_next_processing_job("worker", 120)
    assert claimed["video_id"] == video["id"]


def test_demo_source_reset_removes_private_files_and_allows_reupload(tmp_path):
    store = ProductStore(tmp_path / "creatorcut.sqlite")
    creator = store.ensure_creator("Administrator")
    experiment = store.create_backtest_experiment(
        creator["id"], "Recommendation test", "tracker.csv", parsed_tracker()
    )
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    long_video = upload_dir / "source.mp4"
    long_video.write_bytes(b"video")
    video = store.create_video(creator["id"], long_video.name, long_video)
    store.attach_backtest_source(
        experiment["id"],
        creator["id"],
        "training_vid_1",
        "reference",
        video["id"],
    )
    short_paths = []
    for row in [
        item
        for item in experiment["reference_clips"]
        if item["source_key"] == "training_vid_1"
    ]:
        path = upload_dir / f"{row['content_id']}.mp4"
        path.write_bytes(b"short")
        short_paths.append(path)
        store.save_backtest_short_upload(
            experiment["id"],
            creator["id"],
            "training_vid_1",
            path.name,
            path,
        )
    work_dir = tmp_path / "work"
    job_dir = work_dir / video["id"]
    job_dir.mkdir(parents=True)
    (job_dir / "transcript.json").write_text("{}", encoding="utf-8")

    class ProcessorStub:
        pass

    processor = ProcessorStub()
    processor.work_dir = work_dir
    application = ProductApplication(store, processor, upload_dir)
    reset = application.delete_backtest_source(
        {"creator_id": creator["id"]}, experiment["id"], "training_vid_1"
    )

    source = next(
        item for item in reset["sources"] if item["source_key"] == "training_vid_1"
    )
    rows = [
        item
        for item in reset["reference_clips"]
        if item["source_key"] == "training_vid_1"
    ]
    assert source["video_id"] is None
    assert source["status"] == "awaiting_upload"
    assert all(row["uploaded_filename"] is None for row in rows)
    assert not long_video.exists()
    assert not any(path.exists() for path in short_paths)
    assert not job_dir.exists()
    assert store.processing_queue_summary()["queued"] == 0
    with pytest.raises(KeyError):
        store.get_video(video["id"])


def test_demo_source_reset_waits_for_an_active_worker(tmp_path):
    store = ProductStore(tmp_path / "creatorcut.sqlite")
    creator = store.ensure_creator("Administrator")
    experiment = store.create_backtest_experiment(
        creator["id"], "Recommendation test", "tracker.csv", parsed_tracker()
    )
    source = tmp_path / "training_vid_1.mp4"
    source.write_bytes(b"video")
    video = store.create_video(creator["id"], source.name, source)
    store.attach_backtest_source(
        experiment["id"],
        creator["id"],
        "training_vid_1",
        "reference",
        video["id"],
    )
    store.claim_next_processing_job("worker", 120)

    with pytest.raises(ValueError, match="actively processing"):
        store.reset_backtest_source(
            experiment["id"], creator["id"], "training_vid_1"
        )


def test_reference_only_calibration_activates_across_two_source_videos(tmp_path):
    store = ProductStore(tmp_path / "creatorcut.sqlite")
    creator = store.ensure_creator("Test channel")
    experiment = store.create_backtest_experiment(
        creator["id"], "Recommendation test", "tracker.csv", parsed_tracker()
    )
    for source_index, source_key in enumerate(experiment["reference_keys"]):
        path = tmp_path / f"{source_key}.mp4"
        path.touch()
        video = store.create_video(creator["id"], path.name, path)
        store.attach_backtest_source(
            experiment["id"], creator["id"], source_key, "reference", video["id"]
        )
        rows = [
            row for row in experiment["reference_clips"] if row["source_key"] == source_key
        ]
        for local_index, row in enumerate(rows, start=1):
            index = source_index * 3 + local_index
            clip_id = store.save_custom_clip(
                video["id"],
                stored_clip(index, 3.0 + index / 10),
                origin="historical_reference",
                event_type="historical_selected",
            )
            store.save_backtest_alignment(
                row["id"],
                {
                    "status": "aligned",
                    "source_start_seconds": index * 20.0,
                    "source_end_seconds": index * 20.0 + 30.0,
                    "confidence": 0.9,
                    "method": "test",
                    "segments": [],
                },
                clip_id,
            )
            store.save_performance_report(
                clip_id,
                {
                    "platform": "youtube",
                    "views": 1_000 + index * 100,
                    "likes": 20 + index * index * 5,
                    "comments": 2 + index * 2,
                },
            )

    adjusted, metadata = store.personalize_backtest_candidates(
        experiment["id"], [candidate(2), candidate(5)]
    )

    assert metadata["active"] is True
    assert metadata["eligible_clip_count"] == 6
    assert metadata["source_video_count"] == 2
    assert all(item["backtest_lineage"]["active"] for item in adjusted)
    assert all(
        0 <= item["backtest_lineage"]["estimated_relative_performance"] <= 100
        for item in adjusted
    )


def test_evaluation_reports_recovery_and_outcome_rank_agreement():
    predictions = [
        {
            "rank": 1,
            "clip_id": "pred_1",
            "start_seconds": 100.0,
            "end_seconds": 140.0,
            "estimated_relative_performance": 80.0,
        },
        {
            "rank": 2,
            "clip_id": "pred_2",
            "start_seconds": 200.0,
            "end_seconds": 240.0,
            "estimated_relative_performance": 30.0,
        },
    ]
    actuals = [
        {
            "content_id": "actual_1",
            "source_start_seconds": 102.0,
            "source_end_seconds": 142.0,
            "views": 10_000,
            "likes": 1_000,
            "comments": 100,
            "segments": [],
        },
        {
            "content_id": "actual_2",
            "source_start_seconds": 201.0,
            "source_end_seconds": 241.0,
            "views": 1_000,
            "likes": 10,
            "comments": 1,
            "segments": [],
        },
    ]

    report = evaluate_backtest(predictions, actuals)

    assert report["top_k_recall_at_iou_50"] == 1.0
    assert report["performance_rank_correlation"] == 1.0
    assert [match["recovered"] for match in report["matches"]] == [True, True]
