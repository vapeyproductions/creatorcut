import json

from creatorcut.feedback_export import export_feedback_snapshot
from creatorcut.product_store import ProductStore


def test_feedback_snapshot_preserves_lineage_labels_and_boundary_edits(tmp_path):
    database = tmp_path / "creatorcut.sqlite"
    store = ProductStore(database)
    creator = store.ensure_creator("Example", "creator_example")
    source = tmp_path / "source.mp4"
    source.touch()
    video = store.create_video(creator["id"], source.name, source)
    store.update_video(video["id"], "ready", duration_seconds=120.0)
    clip = {
        "id": f"{video['id']}_clip_1",
        "rank": 1,
        "global_rank": 4,
        "ranking_model_version": "creatorcut_ranker_freeze_v1",
        "start_seconds": 20.0,
        "end_seconds": 50.0,
        "duration_seconds": 30.0,
        "transcript_text": "A complete pricing strategy story.",
        "semantic_embedding": [0.6, 0.8],
        "global_score": 4.0,
        "personalized_score": 4.1,
        "predicted_targets": {
            "hook": 4.0,
            "completeness": 4.0,
            "payoff": 4.0,
            "clarity": 4.0,
        },
        "editorial_adjustment": 0.1,
        "explanation": "Selected for a strong opening.",
    }
    store.save_ranked_clips(video["id"], [clip])
    store.record_editorial_event(
        clip["id"],
        "download_edited",
        21.5,
        48.0,
        {"start_adjustment_seconds": 1.5, "end_adjustment_seconds": -2.0},
    )
    store.save_performance_report(
        clip["id"],
        {
            "platform": "youtube",
            "engaged_views": 700,
            "average_view_percentage": 78.0,
            "published_at": None,
        },
    )

    summary = export_feedback_snapshot(
        database,
        tmp_path / "feedback.jsonl",
        tmp_path / "summary.json",
    )
    record = json.loads((tmp_path / "feedback.jsonl").read_text(encoding="utf-8"))

    assert summary["record_count"] == 1
    assert summary["labeled_record_count"] == 1
    assert summary["performance_record_count"] == 1
    assert len(summary["records_sha256"]) == 64
    assert record["ranking_model_version"] == "creatorcut_ranker_freeze_v1"
    assert record["editorial_label"] == "download_edited"
    assert record["start_delta_seconds"] == 1.5
    assert record["end_delta_seconds"] == -2.0
    assert record["latest_performance"]["engaged_views"] == 700
    assert "media_path" not in record
