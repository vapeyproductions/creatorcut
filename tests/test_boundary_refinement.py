import pytest

from creatorcut.boundary_refinement import (
    evaluate_boundary_refinements,
    propose_boundary_refinement,
)


def word(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "word": text, "probability": 0.99}


def case(start: float = 10.0, end: float = 20.0) -> dict:
    return {
        "analysis_id": "video_001_top1_failure",
        "video_id": "video_001",
        "model_selected": {"start_seconds": start, "end_seconds": end},
    }


def test_refiner_moves_mid_sentence_start_to_clean_prior_sentence() -> None:
    window = {
        "words": [
            word(5.0, 5.5, " A"),
            word(5.6, 6.0, " complete"),
            word(6.1, 6.5, " setup."),
            word(10.0, 10.4, " And"),
            word(10.5, 11.0, " then"),
            word(11.1, 11.6, " payoff."),
            word(19.0, 19.4, " Final"),
            word(19.5, 20.0, " thought."),
        ],
        "speech_intervals": [],
    }

    proposal = propose_boundary_refinement(case(), window)

    assert proposal["refined_start_seconds"] == pytest.approx(5.0)
    assert proposal["start_adjustment_seconds"] == pytest.approx(-5.0)
    assert proposal["start_evidence"]["clean_opening"] is True


def test_refiner_extends_end_to_complete_sentence_after_pause() -> None:
    window = {
        "words": [
            word(10.0, 10.5, " Strong"),
            word(10.6, 11.0, " start."),
            word(18.0, 18.5, " The"),
            word(18.6, 19.0, " answer"),
            word(19.1, 20.0, " continues"),
            word(20.1, 20.5, " to"),
            word(20.6, 21.4, " completion."),
        ],
        "speech_intervals": [
            {"start_seconds": 10.0, "end_seconds": 11.0},
            {"start_seconds": 18.0, "end_seconds": 21.4},
        ],
    }

    proposal = propose_boundary_refinement(case(), window)

    assert proposal["refined_end_seconds"] == pytest.approx(21.4)
    assert proposal["end_adjustment_seconds"] == pytest.approx(1.4)
    assert proposal["end_evidence"]["complete_sentence"] is True


def test_refiner_rejects_empty_timestamp_window() -> None:
    with pytest.raises(ValueError, match="no timestamped words"):
        propose_boundary_refinement(case(), {"words": [], "speech_intervals": []})


def test_evaluation_keeps_proposal_generation_separate_from_human_labels() -> None:
    proposals = [
        {
            "analysis_id": "failure_a",
            "video_id": "video_a",
            "start_adjustment_seconds": -3.0,
            "end_adjustment_seconds": 2.0,
        }
    ]
    reviews = [
        {
            "analysis_id": "failure_a",
            "start_adjustment_seconds": -4.0,
            "end_adjustment_seconds": None,
        }
    ]

    evaluation = evaluate_boundary_refinements(proposals, reviews)

    assert evaluation["metrics"]["start"]["labeled_count"] == 1
    assert evaluation["metrics"]["start"]["mean_absolute_error_seconds"] == pytest.approx(1.0)
    assert evaluation["metrics"]["start"][
        "no_change_mean_absolute_error_seconds"
    ] == pytest.approx(4.0)
    assert evaluation["metrics"]["end"]["labeled_count"] == 0


def test_evaluation_rejects_duplicate_review_ids() -> None:
    review = {"analysis_id": "failure_a"}

    with pytest.raises(ValueError, match="duplicate"):
        evaluate_boundary_refinements([], [review, review])
