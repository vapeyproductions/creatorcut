"""Promote the completed v1 holdout into a versioned v2 development corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from creatorcut.candidates import write_jsonl
from creatorcut.dataset import load_jsonl
from creatorcut.training import build_reviewed_training_records_from_queue


def _unique_records(records: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        annotation_id = record.get("annotation_id")
        if not isinstance(annotation_id, str) or not annotation_id:
            raise ValueError(f"{source} contains an invalid annotation_id")
        if annotation_id in indexed:
            raise ValueError(f"{source} contains duplicate annotation_id {annotation_id}")
        indexed[annotation_id] = record
    return indexed


def combine_v2_development_corpus(
    original_records: list[dict[str, Any]],
    completed_queue: list[dict[str, Any]],
    completed_reviews: list[dict[str, Any]],
    original_embeddings: dict[str, Any],
    completed_embeddings: dict[str, Any],
    use_boundary_edited_scores: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Join two non-overlapping video cohorts with an identical encoder contract."""
    promoted_records = build_reviewed_training_records_from_queue(
        completed_queue, completed_reviews
    )
    reviews_by_id = _unique_records(completed_reviews, "completed reviews")
    original_records = [
        {**record, "target_basis": "human_scores_for_original_boundary"}
        for record in original_records
    ]
    promoted_records = [dict(record) for record in promoted_records]
    edited_target_count = 0
    for record in promoted_records:
        boundary_edit = reviews_by_id[record["annotation_id"]].get("boundary_edit")
        if use_boundary_edited_scores and boundary_edit is not None:
            edited_scores = {
                field: float(boundary_edit["scores"][field])
                for field in ("hook", "completeness", "payoff", "clarity")
            }
            record["targets"] = {
                **edited_scores,
                "quality_score": sum(edited_scores.values()) / len(edited_scores),
            }
            record["target_basis"] = "human_scores_after_boundary_edit"
            edited_target_count += 1
        else:
            record["target_basis"] = "human_scores_for_original_boundary"
    original_by_id = _unique_records(original_records, "original records")
    promoted_by_id = _unique_records(promoted_records, "promoted records")
    duplicate_ids = set(original_by_id) & set(promoted_by_id)
    if duplicate_ids:
        raise ValueError(f"development cohorts overlap annotation IDs: {sorted(duplicate_ids)}")
    original_videos = {record["video_id"] for record in original_records}
    promoted_videos = {record["video_id"] for record in promoted_records}
    duplicate_videos = original_videos & promoted_videos
    if duplicate_videos:
        raise ValueError(f"development cohorts overlap videos: {sorted(duplicate_videos)}")

    original_metadata = original_embeddings.get("metadata")
    promoted_metadata = completed_embeddings.get("metadata")
    if original_metadata != promoted_metadata:
        raise ValueError("embedding metadata differs between development cohorts")
    embedding_records = [
        *original_embeddings.get("records", []),
        *completed_embeddings.get("records", []),
    ]
    embedding_by_id = _unique_records(embedding_records, "embedding records")
    combined_records = [*original_records, *promoted_records]
    expected_ids = {record["annotation_id"] for record in combined_records}
    if set(embedding_by_id) != expected_ids:
        missing = sorted(expected_ids - set(embedding_by_id))
        extra = sorted(set(embedding_by_id) - expected_ids)
        raise ValueError(f"embedding identity mismatch; missing={missing}, extra={extra}")
    for record in combined_records:
        embedded = embedding_by_id[record["annotation_id"]]
        if embedded.get("video_id") != record.get("video_id"):
            raise ValueError(f"{record['annotation_id']}: embedding video_id differs")

    ordered_embeddings = [embedding_by_id[record["annotation_id"]] for record in combined_records]
    artifact = {"metadata": original_metadata, "records": ordered_embeddings}
    summary = {
        "schema": "creatorcut_v2_development_corpus_v1",
        "clips": len(combined_records),
        "videos": len(original_videos | promoted_videos),
        "cohorts": {
            "original_development": {
                "clips": len(original_records),
                "videos": len(original_videos),
            },
            "promoted_after_frozen_v1_evaluation": {
                "clips": len(promoted_records),
                "videos": len(promoted_videos),
            },
        },
        "identity_checks": {
            "overlapping_annotation_ids": 0,
            "overlapping_video_ids": 0,
            "records_with_embeddings": len(ordered_embeddings),
            "encoder_metadata_exact_match": True,
        },
        "evaluation_status": {
            "v1_external_holdout_result_remains_frozen": True,
            "promoted_cohort_is_available_for_v2_development": True,
            "promoted_cohort_must_not_be_called_an_untouched_v2_holdout": True,
            "future_v2_claims_require_a_new_video-level_holdout": True,
        },
        "target_definition": {
            "name": "attainable_quality" if use_boundary_edited_scores else "original_quality",
            "boundary_edited_score_rows": edited_target_count,
            "original_boundary_score_rows": len(combined_records) - edited_target_count,
            "rationale": (
                "The moment ranker predicts achievable content value; boundary selection is "
                "evaluated as a separate task."
                if use_boundary_edited_scores
                else "Every row retains the score for its originally proposed interval."
            ),
        },
    }
    return combined_records, artifact, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CreatorCut's combined v2 development data")
    parser.add_argument("--original-records", type=Path, required=True)
    parser.add_argument("--completed-queue", type=Path, required=True)
    parser.add_argument("--completed-reviews", type=Path, required=True)
    parser.add_argument("--original-embeddings", type=Path, required=True)
    parser.add_argument("--completed-embeddings", type=Path, required=True)
    parser.add_argument("--output-records", type=Path, required=True)
    parser.add_argument("--output-embeddings", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument(
        "--keep-original-boundary-scores",
        action="store_true",
        help="Do not substitute the separately scored edited interval for v2 moment ranking",
    )
    args = parser.parse_args()

    records, embeddings, summary = combine_v2_development_corpus(
        load_jsonl(args.original_records),
        load_jsonl(args.completed_queue),
        load_jsonl(args.completed_reviews),
        json.loads(args.original_embeddings.read_text(encoding="utf-8")),
        json.loads(args.completed_embeddings.read_text(encoding="utf-8")),
        use_boundary_edited_scores=not args.keep_original_boundary_scores,
    )
    write_jsonl(records, args.output_records)
    for path, value in (
        (args.output_embeddings, embeddings),
        (args.output_summary, summary),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
