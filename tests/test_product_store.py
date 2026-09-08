import pytest

from creatorcut.product_store import ProductStore


def ranked_clip(rank, hook, start):
    return {
        "id": f"video_clip_{rank}",
        "rank": rank,
        "start_seconds": start,
        "end_seconds": start + 30.0,
        "duration_seconds": 30.0,
        "transcript_text": "A candidate clip.",
        "global_score": 3.5,
        "personalized_score": 3.5,
        "predicted_targets": {
            "hook": hook,
            "completeness": 4.0,
            "payoff": 4.0,
            "clarity": 4.0,
        },
        "explanation": "Selected for a strong opening.",
    }


def populated_store(tmp_path):
    store = ProductStore(tmp_path / "creatorcut.sqlite")
    creator = store.ensure_creator("Example channel", "creator_test")
    source = tmp_path / "source.mp4"
    source.touch()
    video = store.create_video(creator["id"], "source.mp4", source)
    store.update_video(video["id"], "ranking_candidates", duration_seconds=180.0)
    clips = [
        ranked_clip(1, 5.0, 10.0),
        ranked_clip(2, 3.0, 60.0),
        ranked_clip(3, 1.0, 110.0),
    ]
    for clip in clips:
        clip["id"] = f"{video['id']}_clip_{clip['rank']}"
    store.save_ranked_clips(video["id"], clips)
    store.update_video(video["id"], "ready")
    return store, creator, video, clips


def test_store_persists_upload_clips_and_presentations(tmp_path):
    store, creator, video, clips = populated_store(tmp_path)

    saved = store.get_video(video["id"])

    assert saved["creator_id"] == creator["id"]
    assert saved["status"] == "ready"
    assert [clip["id"] for clip in saved["clips"]] == [clip["id"] for clip in clips]
    assert store.creator_summary(creator["id"]) == {
        "decision_count": 0,
        "performance_report_count": 0,
        "personalization_active": False,
    }
    assert [item["id"] for item in store.list_videos(creator["id"])] == [video["id"]]


def test_editorial_feedback_activates_bounded_personalization(tmp_path):
    store, creator, _, clips = populated_store(tmp_path)
    store.record_editorial_event(clips[0]["id"], "download_original", 10.0, 40.0)
    store.record_editorial_event(clips[1]["id"], "download_edited", 61.0, 90.0)
    store.record_editorial_event(clips[2]["id"], "reject", 110.0, 140.0)

    candidates = [
        {
            "candidate_id": "high_hook",
            "duration_seconds": 30.0,
            "global_score": 3.5,
            "predicted_targets": {
                "hook": 5.0,
                "completeness": 4.0,
                "payoff": 4.0,
                "clarity": 4.0,
            },
        },
        {
            "candidate_id": "low_hook",
            "duration_seconds": 30.0,
            "global_score": 3.5,
            "predicted_targets": {
                "hook": 1.0,
                "completeness": 4.0,
                "payoff": 4.0,
                "clarity": 4.0,
            },
        },
    ]

    personalized, metadata = store.personalize_candidates(creator["id"], candidates)

    assert metadata["active"] is True
    assert personalized[0]["personalized_score"] > personalized[1]["personalized_score"]
    assert store.creator_summary(creator["id"])["decision_count"] == 3


def test_personalization_remains_off_during_cold_start(tmp_path):
    store, creator, _, clips = populated_store(tmp_path)
    store.record_editorial_event(clips[0]["id"], "download_original", 10.0, 40.0)
    candidate = {
        "candidate_id": "new",
        "duration_seconds": 30.0,
        "global_score": 3.5,
        "predicted_targets": {
            "hook": 5.0,
            "completeness": 4.0,
            "payoff": 4.0,
            "clarity": 4.0,
        },
    }

    personalized, metadata = store.personalize_candidates(creator["id"], [candidate])

    assert metadata["active"] is False
    assert personalized[0]["personalized_score"] == candidate["global_score"]


def test_store_keeps_performance_separate_from_editorial_decisions(tmp_path):
    store, creator, _, clips = populated_store(tmp_path)
    store.save_performance_report(
        clips[0]["id"],
        {
            "platform": "youtube",
            "views": 1000,
            "likes": 80,
            "comments": 4,
            "shares": 9,
            "average_view_percentage": 76.5,
            "published_at": None,
        },
    )

    summary = store.creator_summary(creator["id"])

    assert summary["performance_report_count"] == 1
    assert summary["decision_count"] == 0


def test_store_rejects_unknown_editorial_event(tmp_path):
    store, _, _, clips = populated_store(tmp_path)

    with pytest.raises(ValueError, match="Unsupported editorial event"):
        store.record_editorial_event(clips[0]["id"], "liked", 10.0, 40.0)
