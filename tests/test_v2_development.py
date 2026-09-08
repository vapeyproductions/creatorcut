import pytest

from creatorcut.v2_development import combine_v2_development_corpus


def training_record(annotation_id: str, video_id: str) -> dict:
    return {"annotation_id": annotation_id, "video_id": video_id}


def queue_item(annotation_id: str, video_id: str) -> dict:
    return {
        "annotation_id": annotation_id,
        "candidate_id": f"candidate_{annotation_id}",
        "video_id": video_id,
        "start_seconds": 1.0,
        "end_seconds": 21.0,
        "duration_seconds": 20.0,
        "transcript_text": "Why does this matter? Here is a complete answer.",
    }


def review(annotation_id: str, video_id: str) -> dict:
    return {
        **queue_item(annotation_id, video_id),
        "hook": 4,
        "completeness": 5,
        "payoff": 4,
        "clarity": 5,
        "technically_exportable": True,
        "boundary_edit": {
            "start_seconds": 1.0,
            "end_seconds": 20.0,
            "scores": {"hook": 5, "completeness": 5, "payoff": 5, "clarity": 5},
        },
    }


def embedding_artifact(annotation_id: str, video_id: str) -> dict:
    return {
        "metadata": {"embedding_schema": "test", "dimension": 2},
        "records": [
            {"annotation_id": annotation_id, "video_id": video_id, "embedding": [0.1, 0.2]}
        ],
    }


def test_combiner_preserves_cohorts_and_exact_embedding_identity() -> None:
    records, embeddings, summary = combine_v2_development_corpus(
        [training_record("old", "video_old")],
        [queue_item("new", "video_new")],
        [review("new", "video_new")],
        embedding_artifact("old", "video_old"),
        embedding_artifact("new", "video_new"),
    )

    assert [record["annotation_id"] for record in records] == ["old", "new"]
    assert [record["annotation_id"] for record in embeddings["records"]] == ["old", "new"]
    assert summary["clips"] == 2
    assert summary["videos"] == 2
    assert records[1]["targets"]["quality_score"] == 5.0
    assert records[1]["target_basis"] == "human_scores_after_boundary_edit"
    assert summary["target_definition"]["boundary_edited_score_rows"] == 1
    assert summary["evaluation_status"]["future_v2_claims_require_a_new_video-level_holdout"]


def test_combiner_rejects_video_overlap_between_cohorts() -> None:
    with pytest.raises(ValueError, match="overlap videos"):
        combine_v2_development_corpus(
            [training_record("old", "same_video")],
            [queue_item("new", "same_video")],
            [review("new", "same_video")],
            embedding_artifact("old", "same_video"),
            embedding_artifact("new", "same_video"),
        )
