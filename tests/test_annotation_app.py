import json

import pytest

from creatorcut.annotation_app import (
    AnnotationApplication,
    ReviewStore,
    parse_byte_range,
    validate_labels,
)


def task(annotation_id="video_005_sample_01", video_id="video_005"):
    return {
        "annotation_id": annotation_id,
        "candidate_id": f"{video_id}_candidate_0001",
        "video_id": video_id,
        "start_seconds": 10.0,
        "end_seconds": 40.0,
        "duration_seconds": 30.0,
        "transcript_text": "A complete thought.",
        "sampling": {"score_band": "high", "proxy_score": 88.0},
    }


def labels():
    return {
        "hook": 4,
        "completeness": 5,
        "payoff": 4,
        "clarity": 5,
        "technically_exportable": True,
        "notes": "Clear and useful",
        "boundary_edit": None,
    }


def test_validate_labels_accepts_review_scores():
    assert validate_labels(labels()) == labels()


def test_validate_labels_rejects_boolean_score():
    value = labels()
    value["hook"] = True

    with pytest.raises(ValueError, match="hook"):
        validate_labels(value)


@pytest.mark.parametrize(
    ("header", "expected"),
    [(None, None), ("bytes=10-19", (10, 19)), ("bytes=90-", (90, 99)), ("bytes=-10", (90, 99))],
)
def test_parse_byte_range(header, expected):
    assert parse_byte_range(header, 100) == expected


def test_review_store_replaces_record_atomically(tmp_path):
    path = tmp_path / "reviews.jsonl"
    store = ReviewStore(path)
    store.save(task(), labels())
    updated = labels()
    updated["hook"] = 2
    store.save(task(), updated)

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["hook"] == 2
    assert records[0]["presentation"] == 5
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_application_hides_sampling_metadata_and_resumes(tmp_path):
    queue = [task(), task("video_005_sample_02")]
    manifest = [
        {
            "video_id": "video_005",
            "local_filename": "data/raw/video_005.mp4",
        }
    ]
    store = ReviewStore(tmp_path / "reviews.jsonl")
    application = AnnotationApplication(queue, manifest, store, tmp_path, batch_size=10)

    first_batch = application.next_tasks()
    assert len(first_batch) == 2
    assert "sampling" not in first_batch[0]
    application.save_review(first_batch[0]["annotation_id"], labels())

    resumed = application.next_tasks()
    assert [item["annotation_id"] for item in resumed] == ["video_005_sample_02"]
    assert application.stats() == {"total": 2, "completed": 1, "remaining": 1}


def test_application_preserves_original_scores_and_normalizes_boundary_edit(tmp_path):
    manifest = [{"video_id": "video_005", "local_filename": "video_005.mp4"}]
    store = ReviewStore(tmp_path / "reviews.jsonl")
    application = AnnotationApplication([task()], manifest, store, tmp_path)
    value = labels()
    value["boundary_edit"] = {
        "start_seconds": 10.0,
        "end_seconds": 37.5,
        "scores": {"hook": 4, "completeness": 5, "payoff": 5, "clarity": 5},
    }

    saved = application.save_review(task()["annotation_id"], value)

    assert saved["hook"] == 4
    assert saved["end_seconds"] == 40.0
    assert saved["boundary_edit"] == {
        "start_seconds": 10.0,
        "end_seconds": 37.5,
        "duration_seconds": 27.5,
        "start_adjustment_seconds": 0.0,
        "end_adjustment_seconds": -2.5,
        "scores": {"hook": 4, "completeness": 5, "payoff": 5, "clarity": 5},
    }


def test_application_rejects_enabled_edit_without_a_boundary_change(tmp_path):
    manifest = [{"video_id": "video_005", "local_filename": "video_005.mp4"}]
    application = AnnotationApplication(
        [task()], manifest, ReviewStore(tmp_path / "reviews.jsonl"), tmp_path
    )
    value = labels()
    value["boundary_edit"] = {
        "start_seconds": 10.0,
        "end_seconds": 40.0,
        "scores": {field: 4 for field in ("hook", "completeness", "payoff", "clarity")},
    }

    with pytest.raises(ValueError, match="Change at least one boundary"):
        application.save_review(task()["annotation_id"], value)


def test_validate_labels_requires_all_edited_scores():
    value = labels()
    value["boundary_edit"] = {
        "start_seconds": 10.0,
        "end_seconds": 37.5,
        "scores": {"hook": 4, "completeness": 5, "payoff": 5},
    }

    with pytest.raises(ValueError, match="boundary_edit.scores.clarity"):
        validate_labels(value)


@pytest.mark.parametrize(
    ("start_seconds", "end_seconds", "message"),
    [(41.0, 60.0, "Edited start"), (10.0, 71.0, "Edited end")],
)
def test_application_rejects_boundary_adjustments_over_thirty_seconds(
    tmp_path, start_seconds, end_seconds, message
):
    manifest = [{"video_id": "video_005", "local_filename": "video_005.mp4"}]
    application = AnnotationApplication(
        [task()], manifest, ReviewStore(tmp_path / "reviews.jsonl"), tmp_path
    )
    value = labels()
    value["boundary_edit"] = {
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "scores": {field: 4 for field in ("hook", "completeness", "payoff", "clarity")},
    }

    with pytest.raises(ValueError, match=message):
        application.save_review(task()["annotation_id"], value)
