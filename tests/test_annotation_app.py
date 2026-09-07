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
