import json

import pytest

from creatorcut.product_store import ProductStore


def ranked_clip(rank, hook, start):
    semantic_embedding = [(5.0 - hook) / 4.0, (hook - 1.0) / 4.0]
    topic = (
        "pricing strategy"
        if hook >= 4
        else "general introduction"
        if hook <= 2
        else "practical advice"
    )
    return {
        "id": f"video_clip_{rank}",
        "rank": rank,
        "start_seconds": start,
        "end_seconds": start + 30.0,
        "duration_seconds": 30.0,
        "transcript_text": f"A complete {topic} example.",
        "semantic_embedding": semantic_embedding,
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
        "analytics_import_count": 0,
        "personalization_active": False,
        "performance_personalization_active": False,
        "performance_eligible_clip_count": 0,
        "performance_minimum_clip_count": 5,
        "semantic_performance_active": False,
        "semantic_performance_example_count": 0,
        "positive_semantic_trends": [],
        "negative_semantic_trends": [],
        "performance_insight_summary": None,
    }
    assert [item["id"] for item in store.list_videos(creator["id"])] == [video["id"]]
    assert "semantic_embedding_json" not in saved["clips"][0]
    assert json.loads(store.get_clip(clips[0]["id"])["semantic_embedding_json"])


def test_ml_report_exposes_lineage_labels_edits_and_job_state(tmp_path):
    store, creator, video, clips = populated_store(tmp_path)
    store.save_video_publishability_summary(
        video["id"],
        {
            "rule_version": "creatorcut_publishability_rules_v1",
            "candidate_count": 40,
            "eligible_count": 38,
            "blocked_count": 2,
        },
    )
    store.record_editorial_event(
        clips[0]["id"], "download_edited", 11.0, 39.0
    )
    store.record_editorial_event(
        clips[1]["id"], "reject", clips[1]["start_seconds"], clips[1]["end_seconds"]
    )

    report = store.creator_ml_report(creator["id"])

    assert report["report_schema"] == "creatorcut_ml_operations_report_v1"
    assert report["serving"]["video_status_counts"] == {"ready": 1}
    assert report["serving"]["job_status_counts"] == {"succeeded": 1}
    assert report["feedback"]["presented_model_clip_count"] == 3
    assert report["feedback"]["selected_model_clip_count"] == 1
    assert report["feedback"]["model_clip_selection_rate"] == pytest.approx(1 / 3, abs=0.0001)
    assert report["feedback"]["median_total_boundary_change_seconds"] == 2.0
    assert report["feedback"]["event_counts"]["reject"]["distinct_clips"] == 1
    assert report["lineage"][0]["model_version"] == "unversioned"
    assert report["publishability"] == {
        "assessed_video_count": 1,
        "candidate_count": 40,
        "blocked_candidate_count": 2,
        "rule_versions": ["creatorcut_publishability_rules_v1"],
    }
    assert report["global_model_policy"]["frozen"] is True
    assert report["global_model_policy"]["production_feedback_auto_trains_global_model"] is False


def test_store_can_backfill_a_legacy_clip_embedding(tmp_path):
    store, _, _, clips = populated_store(tmp_path)

    store.save_clip_semantic_embedding(clips[0]["id"], [0.6, 0.8])

    assert json.loads(store.get_clip(clips[0]["id"])["semantic_embedding_json"]) == [
        0.6,
        0.8,
    ]


def test_store_records_creator_authored_clip_as_explicit_preference(tmp_path):
    store, creator, video, _ = populated_store(tmp_path)
    clip_id = store.save_custom_clip(
        video["id"],
        {
            "start_seconds": 145.0,
            "end_seconds": 175.0,
            "duration_seconds": 30.0,
            "transcript_text": "A creator found this pricing story themselves.",
            "semantic_embedding": [0.0, 1.0],
            "global_score": 3.8,
            "personalized_score": 3.8,
            "predicted_targets": {
                "hook": 4.0,
                "completeness": 4.0,
                "payoff": 4.0,
                "clarity": 4.0,
            },
            "explanation": "Creator-defined interval.",
        },
    )

    saved = store.get_video(video["id"])

    assert saved["clips"][-1]["id"] == clip_id
    assert saved["clips"][-1]["origin"] == "creator"
    assert store.creator_summary(creator["id"])["decision_count"] == 1


def test_completed_review_records_unselected_options_once(tmp_path):
    store, creator, video, clips = populated_store(tmp_path)
    store.record_editorial_event(
        clips[0]["id"], "download_original", clips[0]["start_seconds"], clips[0]["end_seconds"]
    )

    assert store.complete_recommendation_review(video["id"]) == 2
    assert store.complete_recommendation_review(video["id"]) == 0
    summary = store.creator_summary(creator["id"])
    assert summary["decision_count"] == 3
    assert summary["personalization_active"] is True


def test_completed_review_requires_a_positive_selection(tmp_path):
    store, _, video, _ = populated_store(tmp_path)

    with pytest.raises(ValueError, match="Choose or add at least one"):
        store.complete_recommendation_review(video["id"])


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
            "semantic_embedding": [0.0, 1.0],
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
            "semantic_embedding": [1.0, 0.0],
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


def test_high_exposure_import_without_an_outcome_metric_stays_ineligible(tmp_path):
    store, creator, _, clips = populated_store(tmp_path)
    report = {
        "recognized_row_count": 1,
        "totals": {"views": 10_000, "thumbnail_impressions": 20_000},
        "retention_rows": [],
        "warnings": ["No timestamped audience-retention curve was included"],
    }

    saved = store.save_analytics_import(
        creator["id"],
        "reach.csv",
        "published_clip",
        report,
        clip_id=clips[0]["id"],
    )

    assert saved["learning_eligible"] is False
    assert "outcome metric" in saved["learning_status"]


def test_source_retention_signal_is_bounded_and_requires_evidence(tmp_path):
    store, creator, video, _ = populated_store(tmp_path)
    report = {
        "recognized_row_count": 101,
        "totals": {"views": 1_000},
        "retention_rows": [
            {
                "elapsed_video_time_ratio": index / 100,
                "relative_retention_performance": index / 100,
            }
            for index in range(1, 101)
        ],
        "warnings": [],
    }
    saved = store.save_analytics_import(
        creator["id"], "retention.zip", "source_video", report, video_id=video["id"]
    )
    candidates = [
        {
            "candidate_id": "early",
            "start_seconds": 10,
            "end_seconds": 40,
            "personalized_score": 3.5,
        },
        {
            "candidate_id": "late",
            "start_seconds": 130,
            "end_seconds": 160,
            "personalized_score": 3.5,
        },
    ]

    adjusted, metadata = store.apply_source_retention_signal(video["id"], candidates)

    assert saved["learning_eligible"] is True
    assert metadata["active"] is True
    assert adjusted[1]["personalized_score"] > adjusted[0]["personalized_score"]
    assert all(abs(item["source_retention_adjustment"]) <= 0.2 for item in adjusted)


def test_short_import_needs_outcome_data_and_one_video(tmp_path):
    store, creator, _, clips = populated_store(tmp_path)
    sparse_report = {
        "recognized_row_count": 1,
        "totals": {"views": 500},
        "report_rows": [{"views": 500}],
        "retention_rows": [],
        "warnings": [],
    }

    saved = store.save_analytics_import(
        creator["id"],
        "short.csv",
        "published_clip",
        sparse_report,
        clip_id=clips[0]["id"],
    )

    assert saved["learning_eligible"] is False
    assert "outcome metric" in saved["learning_status"]

    multiple_videos = {
        **sparse_report,
        "report_rows": [
            {"video_id": "one", "views": 250},
            {"video_id": "two", "views": 250},
        ],
    }
    with pytest.raises(ValueError, match="filtered to one"):
        store.save_analytics_import(
            creator["id"],
            "channel.csv",
            "published_clip",
            multiple_videos,
            clip_id=clips[0]["id"],
        )


def test_performance_layer_learns_only_after_five_comparable_shorts(tmp_path):
    store, creator, _, clips = populated_store(tmp_path)
    all_clips = list(clips)
    source = tmp_path / "second.mp4"
    source.touch()
    second_video = store.create_video(creator["id"], "second.mp4", source)
    store.update_video(second_video["id"], "ranking_candidates", duration_seconds=180.0)
    extra = [ranked_clip(rank, hook, 20.0 * rank) for rank, hook in ((1, 2.0), (2, 4.0))]
    for clip in extra:
        clip["id"] = f"{second_video['id']}_clip_{clip['rank']}"
    store.save_ranked_clips(second_video["id"], extra)
    all_clips.extend(extra)

    ordered_clips = sorted(
        all_clips, key=lambda item: item["predicted_targets"]["hook"]
    )
    for index, clip in enumerate(ordered_clips):
        store.save_performance_report(
            clip["id"],
            {
                "platform": "youtube",
                "views": 500,
                "engaged_views": 300,
                "average_view_percentage": 45 + index * 10,
                "likes": 20 + index * 5,
                "comments": 2 + index,
                "shares": 3 + index,
                "published_at": None,
            },
        )

    candidates = [
        {
            "candidate_id": "high_hook",
            "duration_seconds": 30.0,
            "global_score": 3.5,
            "semantic_embedding": [0.0, 1.0],
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
            "semantic_embedding": [1.0, 0.0],
            "predicted_targets": {
                "hook": 1.0,
                "completeness": 4.0,
                "payoff": 4.0,
                "clarity": 4.0,
            },
        },
    ]

    personalized, metadata = store.personalize_candidates(creator["id"], candidates)

    assert metadata["performance"]["active"] is True
    assert metadata["performance"]["semantic_active"] is True
    assert personalized[0]["performance_adjustment"] > personalized[1][
        "performance_adjustment"
    ]
    assert personalized[0]["semantic_performance_adjustment"] > personalized[1][
        "semantic_performance_adjustment"
    ]
    assert all(
        abs(
            item["performance_adjustment"]
            + item["semantic_performance_adjustment"]
        )
        <= 0.25
        for item in personalized
    )
    summary = store.creator_summary(creator["id"])
    assert summary["semantic_performance_active"] is True
    assert summary["performance_insight_summary"].startswith("Across 5 eligible Shorts")
    assert any(
        trend["term"] == "pricing strategy"
        for trend in summary["positive_semantic_trends"]
    )
    assert any(
        trend["term"] == "general introduction"
        for trend in summary["negative_semantic_trends"]
    )


def test_store_rejects_unknown_editorial_event(tmp_path):
    store, _, _, clips = populated_store(tmp_path)

    with pytest.raises(ValueError, match="Unsupported editorial event"):
        store.record_editorial_event(clips[0]["id"], "liked", 10.0, 40.0)
