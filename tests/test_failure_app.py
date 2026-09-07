import json

import pytest

from creatorcut.failure_app import (
    FailureAnalysisApplication,
    FailureReviewStore,
    validate_failure_review,
)


def case() -> dict:
    return {
        "analysis_id": "video_005_top1_failure",
        "video_id": "video_005",
        "regret": 1.0,
        "model_selected": {"annotation_id": "selected"},
        "human_best": {"annotation_id": "best"},
    }


def diagnosis() -> dict:
    return {
        "reasons": ["missing_start_context", "weak_hook"],
        "boundary_fixable": True,
        "start_adjustment_seconds": -3,
        "end_adjustment_seconds": None,
        "notes": "Needs the preceding sentence.",
    }


def test_validate_failure_review_accepts_structured_diagnosis() -> None:
    assert validate_failure_review(diagnosis()) == {
        **diagnosis(),
        "start_adjustment_seconds": -3.0,
    }


def test_validate_failure_review_requires_known_reason() -> None:
    value = {**diagnosis(), "reasons": ["mystery"]}

    with pytest.raises(ValueError, match="unknown reason"):
        validate_failure_review(value)


def test_failure_store_replaces_diagnosis_atomically(tmp_path) -> None:
    path = tmp_path / "failure_reviews.jsonl"
    store = FailureReviewStore(path)
    store.save(case(), validate_failure_review(diagnosis()))
    updated = {**diagnosis(), "reasons": ["too_long"], "boundary_fixable": False}
    store.save(case(), validate_failure_review(updated))

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["reasons"] == ["too_long"]
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_failure_application_resumes_and_restricts_media(tmp_path) -> None:
    second = {**case(), "analysis_id": "video_005_second"}
    store = FailureReviewStore(tmp_path / "failure_reviews.jsonl")
    application = FailureAnalysisApplication(
        [case(), second],
        [{"video_id": "video_005", "local_filename": "data/raw/video_005.mp4"}],
        store,
        tmp_path,
    )

    assert len(application.next_cases()) == 1
    application.save_review(case()["analysis_id"], diagnosis())
    assert application.next_cases()[0]["analysis_id"] == "video_005_second"
    assert application.stats() == {"total": 2, "completed": 1, "remaining": 1}
    with pytest.raises(KeyError):
        application.video_path("video_unknown")
