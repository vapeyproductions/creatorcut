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
        "model_selected": {
            "annotation_id": "selected",
            "start_seconds": 10.0,
            "end_seconds": 20.0,
        },
        "human_best": {"annotation_id": "best"},
    }


def diagnosis() -> dict:
    return {
        "reasons": ["missing_start_context", "weak_hook"],
        "preferred_clip": "human_best",
        "boundary_fixable": True,
        "start_adjustment_seconds": -3,
        "end_adjustment_seconds": None,
        "notes": "Needs the preceding sentence.",
    }


def test_validate_failure_review_accepts_structured_diagnosis() -> None:
    assert validate_failure_review(diagnosis()) == {
        **diagnosis(),
        "preference_context": "after_proposed_model_edits",
        "start_adjustment_seconds": -3.0,
    }


def test_validate_failure_review_requires_known_reason() -> None:
    value = {**diagnosis(), "reasons": ["mystery"]}

    with pytest.raises(ValueError, match="unknown reason"):
        validate_failure_review(value)


def test_validate_failure_review_requires_pairwise_editorial_choice() -> None:
    value = {**diagnosis(), "preferred_clip": "unknown"}

    with pytest.raises(ValueError, match="preferred_clip"):
        validate_failure_review(value)


def test_validate_failure_review_records_conditional_preference_context() -> None:
    value = validate_failure_review(diagnosis())

    assert value["preference_context"] == "after_proposed_model_edits"


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
    assert len(application.next_cases(limit=2, include_completed=True)) == 2
    assert application.stats() == {"total": 2, "completed": 1, "remaining": 1}
    with pytest.raises(KeyError):
        application.video_path("video_unknown")


def test_existing_diagnosis_without_preference_is_prefilled_but_incomplete(tmp_path) -> None:
    path = tmp_path / "failure_reviews.jsonl"
    existing = {
        "analysis_id": case()["analysis_id"],
        "reasons": ["weak_hook"],
        "boundary_fixable": False,
    }
    path.write_text(json.dumps(existing) + "\n")
    store = FailureReviewStore(path)
    application = FailureAnalysisApplication(
        [case()],
        [{"video_id": "video_005", "local_filename": "data/raw/video_005.mp4"}],
        store,
        tmp_path,
    )

    pending = application.next_cases()

    assert application.stats()["remaining"] == 1
    assert pending[0]["existing_review"]["reasons"] == ["weak_hook"]


def test_failure_application_rejects_invalid_adjusted_interval(tmp_path) -> None:
    store = FailureReviewStore(tmp_path / "failure_reviews.jsonl")
    application = FailureAnalysisApplication(
        [case()],
        [{"video_id": "video_005", "local_filename": "data/raw/video_005.mp4"}],
        store,
        tmp_path,
    )
    invalid = {
        **diagnosis(),
        "start_adjustment_seconds": 15.0,
        "end_adjustment_seconds": None,
    }

    with pytest.raises(ValueError, match="positive interval"):
        application.save_review(case()["analysis_id"], invalid)
