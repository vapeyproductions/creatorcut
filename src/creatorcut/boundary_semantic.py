"""Evaluate full-clip semantic features for boundary selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from creatorcut.boundary_learning import (
    BOUNDARY_FEATURE_FIELDS,
    cross_validate_boundary_ranker,
)
from creatorcut.candidates import CORE_SCORE_FIELDS
from creatorcut.dataset import load_jsonl
from creatorcut.holdout import _validate_frozen_model
from creatorcut.semantic import DEFAULT_MAX_LENGTH, _download_model_files, encode_texts_onnx
from creatorcut.training import MODEL_FEATURE_FIELDS, queue_model_features

BOUNDARY_SEMANTIC_SCHEMA = "boundary_candidates_semantic_v2"
SEMANTIC_BOUNDARY_FEATURE_FIELDS = (
    "semantic_interval_valid",
    "semantic_candidate_word_count",
    "semantic_similarity_to_original",
    "semantic_quality_score",
    "semantic_quality_delta",
    *(f"semantic_{field}_score" for field in CORE_SCORE_FIELDS),
    *(f"semantic_{field}_delta" for field in CORE_SCORE_FIELDS),
)


def _transcript_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            word
            for segment in transcript.get("segments", [])
            for word in segment.get("words", [])
            if isinstance(word.get("start"), int | float)
            and isinstance(word.get("end"), int | float)
        ),
        key=lambda word: (float(word["start"]), float(word["end"])),
    )


def _clip_text(words: list[dict[str, Any]], start: float, end: float) -> str:
    return " ".join(
        str(word.get("word", "")).strip()
        for word in words
        if float(word["end"]) > start and float(word["start"]) < end
    ).strip()


def _score_frozen_embeddings(
    records: list[dict[str, Any]], embeddings: np.ndarray, frozen: dict[str, Any]
) -> list[dict[str, Any]]:
    model, dimension = _validate_frozen_model(frozen)
    if embeddings.shape != (len(records), dimension):
        raise ValueError("Semantic boundary embeddings do not match the frozen encoder")
    handcrafted = np.asarray(
        [
            [queue_model_features(record)[field] for field in MODEL_FEATURE_FIELDS]
            for record in records
        ],
        dtype=float,
    )
    matrix = np.column_stack([handcrafted, embeddings])
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    standardized = (matrix - means) / scales
    feature_fields = tuple(model["feature_fields"])
    weights = np.column_stack(
        [
            np.asarray(
                [
                    model["standardized_coefficients"][field][name]
                    for name in feature_fields
                ],
                dtype=float,
            )
            for field in CORE_SCORE_FIELDS
        ]
    )
    intercepts = np.asarray(
        [model["intercepts"][field] for field in CORE_SCORE_FIELDS], dtype=float
    )
    target_matrix = standardized @ weights + intercepts
    output = []
    for record_index, record in enumerate(records):
        targets = {
            field: float(target_matrix[record_index, field_index])
            for field_index, field in enumerate(CORE_SCORE_FIELDS)
        }
        output.append(
            {
                "annotation_id": record["annotation_id"],
                "predicted_targets": targets,
                "predicted_quality_score": float(np.mean(list(targets.values()))),
            }
        )
    return output


def build_semantic_boundary_artifact(
    boundary_artifact: dict[str, Any],
    queue: list[dict[str, Any]],
    transcripts_dir: Path,
    frozen_model: dict[str, Any],
    model_cache: Path,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Represent every possible edited clip, not just words around the changed edge."""
    queued_by_id = {row["annotation_id"]: row for row in queue}
    transcript_words = {}
    for video_id in sorted({query["video_id"] for query in boundary_artifact["queries"]}):
        transcript = json.loads(
            (transcripts_dir / f"{video_id}.json").read_text(encoding="utf-8")
        )
        transcript_words[video_id] = _transcript_words(transcript)

    inference_records: list[dict[str, Any]] = []
    option_ids: dict[tuple[str, int], str | None] = {}
    for query in boundary_artifact["queries"]:
        queued = queued_by_id[query["annotation_id"]]
        for option_index, option in enumerate(query["options"]):
            start = (
                float(option["time_seconds"])
                if query["kind"] == "start"
                else float(queued["start_seconds"])
            )
            end = (
                float(queued["end_seconds"])
                if query["kind"] == "start"
                else float(option["time_seconds"])
            )
            text = _clip_text(transcript_words[query["video_id"]], start, end)
            if end <= start or not text:
                option_ids[(query["query_id"], option_index)] = None
                continue
            record_id = f"{query['query_id']}__option_{option_index:04d}"
            option_ids[(query["query_id"], option_index)] = record_id
            inference_records.append(
                {
                    "annotation_id": record_id,
                    "candidate_id": record_id,
                    "video_id": query["video_id"],
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": end - start,
                    "transcript_text": text,
                    "labels": {},
                }
            )

    semantic_metadata = frozen_model["semantic_feature_metadata"]
    tokenizer_path, encoder_path = _download_model_files(
        semantic_metadata["model_id"],
        semantic_metadata["revision"],
        semantic_metadata["onnx_filename"],
        model_cache,
    )
    embeddings = encode_texts_onnx(
        [record["transcript_text"] for record in inference_records],
        tokenizer_path,
        encoder_path,
        batch_size=batch_size,
        maximum_length=semantic_metadata.get("maximum_length", DEFAULT_MAX_LENGTH),
    )
    predictions = _score_frozen_embeddings(inference_records, embeddings, frozen_model)
    prediction_by_id = {row["annotation_id"]: row for row in predictions}
    embedding_by_id = {
        record["annotation_id"]: embeddings[index]
        for index, record in enumerate(inference_records)
    }
    record_by_id = {row["annotation_id"]: row for row in inference_records}

    enriched_queries = []
    invalid_options = 0
    for query in boundary_artifact["queries"]:
        original_index = min(
            range(len(query["options"])),
            key=lambda index: abs(
                float(query["options"][index]["time_seconds"])
                - float(query["original_time_seconds"])
            ),
        )
        original_id = option_ids[(query["query_id"], original_index)]
        if original_id is None:
            raise ValueError(f"{query['query_id']}: original interval is invalid")
        original_prediction = prediction_by_id[original_id]
        original_embedding = embedding_by_id[original_id]
        options = []
        for option_index, option in enumerate(query["options"]):
            record_id = option_ids[(query["query_id"], option_index)]
            if record_id is None:
                invalid_options += 1
                semantic_features = {
                    "semantic_interval_valid": 0.0,
                    "semantic_candidate_word_count": 0.0,
                    "semantic_similarity_to_original": 0.0,
                    "semantic_quality_score": 1.0,
                    "semantic_quality_delta": 1.0
                    - float(original_prediction["predicted_quality_score"]),
                }
                for field in CORE_SCORE_FIELDS:
                    semantic_features[f"semantic_{field}_score"] = 1.0
                    semantic_features[f"semantic_{field}_delta"] = 1.0 - float(
                        original_prediction["predicted_targets"][field]
                    )
            else:
                prediction = prediction_by_id[record_id]
                semantic_features = {
                    "semantic_interval_valid": 1.0,
                    "semantic_candidate_word_count": float(
                        len(record_by_id[record_id]["transcript_text"].split())
                    ),
                    "semantic_similarity_to_original": float(
                        embedding_by_id[record_id] @ original_embedding
                    ),
                    "semantic_quality_score": float(
                        prediction["predicted_quality_score"]
                    ),
                    "semantic_quality_delta": float(
                        prediction["predicted_quality_score"]
                        - original_prediction["predicted_quality_score"]
                    ),
                }
                for field in CORE_SCORE_FIELDS:
                    score = float(prediction["predicted_targets"][field])
                    semantic_features[f"semantic_{field}_score"] = score
                    semantic_features[f"semantic_{field}_delta"] = score - float(
                        original_prediction["predicted_targets"][field]
                    )
            if not all(math.isfinite(value) for value in semantic_features.values()):
                raise ValueError(f"{query['query_id']}: non-finite semantic boundary features")
            options.append(
                {**option, "features": {**option["features"], **semantic_features}}
            )
        enriched_queries.append({**query, "options": options})

    return {
        "metadata": {
            **boundary_artifact["metadata"],
            "schema": BOUNDARY_SEMANTIC_SCHEMA,
            "feature_fields": [
                *BOUNDARY_FEATURE_FIELDS,
                *SEMANTIC_BOUNDARY_FEATURE_FIELDS,
            ],
            "semantic_encoder": semantic_metadata,
            "semantic_feature_definition": (
                "frozen-v1 predictions and embeddings of the complete clip produced by each edge"
            ),
            "encoded_valid_options": len(inference_records),
            "invalid_options": invalid_options,
        },
        "queries": enriched_queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full-clip semantic boundary features")
    parser.add_argument(
        "--boundary-artifact",
        type=Path,
        default=Path("data/processed/v2/boundary_candidates_multimodal_v1.json"),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("data/processed/holdout_v1/annotation_queue.jsonl"),
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("data/processed/holdout_v1/transcripts"),
    )
    parser.add_argument(
        "--frozen-model", type=Path, default=Path("models/frozen_model_v1.json")
    )
    parser.add_argument("--model-cache", type=Path, default=Path("artifacts/huggingface"))
    parser.add_argument(
        "--output-artifact",
        type=Path,
        default=Path("data/processed/v2/boundary_candidates_semantic_v2.json"),
    )
    parser.add_argument(
        "--output-evaluation",
        type=Path,
        default=Path("data/processed/v2/boundary_evaluation_semantic_v2.json"),
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("data/processed/v2/boundary_model_semantic_v2.json"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l2", type=float, default=0.3)
    args = parser.parse_args()

    boundary_artifact = json.loads(args.boundary_artifact.read_text(encoding="utf-8"))
    semantic_artifact = build_semantic_boundary_artifact(
        boundary_artifact,
        load_jsonl(args.queue),
        args.transcripts_dir,
        json.loads(args.frozen_model.read_text(encoding="utf-8")),
        args.model_cache,
        batch_size=args.batch_size,
    )
    evaluation = cross_validate_boundary_ranker(
        semantic_artifact, n_splits=args.splits, seed=args.seed, l2=args.l2
    )
    for path in (args.output_artifact, args.output_evaluation, args.output_model):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_artifact.write_text(json.dumps(semantic_artifact), encoding="utf-8")
    args.output_evaluation.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    args.output_model.write_text(json.dumps(evaluation["final_models"], indent=2), encoding="utf-8")
    print(json.dumps(evaluation["metrics"], indent=2))


if __name__ == "__main__":
    main()
