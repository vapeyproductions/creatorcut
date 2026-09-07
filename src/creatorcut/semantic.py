"""Pretrained transcript embeddings and controlled representation ablations."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from creatorcut.candidates import write_jsonl
from creatorcut.dataset import load_jsonl
from creatorcut.training import (
    FEATURE_SCHEMA,
    MODEL_FEATURE_FIELDS,
    cross_validate_ridge,
    paired_video_bootstrap,
)

DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_ONNX_FILENAME = (
    "onnx/model_qint8_arm64.onnx"
    if platform.machine().lower() == "arm64"
    else "onnx/model.onnx"
)
DEFAULT_MAX_LENGTH = 256
EMBEDDING_SCHEMA = "all_minilm_l6_v2_onnx_v1"


def mean_pool_and_normalize(
    token_embeddings: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    """Mean-pool unmasked token vectors and L2-normalize each transcript embedding."""
    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape [batch, tokens, dimensions]")
    if attention_mask.shape != token_embeddings.shape[:2]:
        raise ValueError("attention_mask must match the first two embedding dimensions")
    expanded_mask = attention_mask.astype(np.float32)[..., None]
    pooled = (token_embeddings * expanded_mask).sum(axis=1)
    pooled /= np.clip(expanded_mask.sum(axis=1), 1e-9, None)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.clip(norms, 1e-12, None)


def _download_model_files(
    model_id: str,
    revision: str,
    onnx_filename: str,
    cache_dir: Path,
) -> tuple[Path, Path]:
    """Resolve a pinned tokenizer and ONNX encoder from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Semantic dependencies are missing; install the project with .[semantic]"
        ) from error

    common = {
        "repo_id": model_id,
        "revision": revision,
        "cache_dir": str(cache_dir),
    }
    tokenizer_path = Path(hf_hub_download(filename="tokenizer.json", **common))
    model_path = Path(hf_hub_download(filename=onnx_filename, **common))
    return tokenizer_path, model_path


def encode_texts_onnx(
    texts: list[str],
    tokenizer_path: Path,
    model_path: Path,
    batch_size: int = 16,
    maximum_length: int = DEFAULT_MAX_LENGTH,
) -> np.ndarray:
    """Encode transcripts using the pinned MiniLM ONNX graph."""
    if not texts:
        raise ValueError("At least one transcript is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if maximum_length <= 0:
        raise ValueError("maximum_length must be positive")
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError(
            "Semantic dependencies are missing; install the project with .[semantic]"
        ) from error

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=maximum_length)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = {value.name for value in session.get_inputs()}
    batches: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        encodings = tokenizer.encode_batch(texts[start : start + batch_size])
        input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.asarray(
            [encoding.attention_mask for encoding in encodings], dtype=np.int64
        )
        available_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.asarray(
                [encoding.type_ids for encoding in encodings], dtype=np.int64
            ),
        }
        feed = {name: available_inputs[name] for name in input_names if name in available_inputs}
        missing_inputs = input_names - feed.keys()
        if missing_inputs:
            raise RuntimeError(f"Unsupported ONNX model inputs: {sorted(missing_inputs)}")
        outputs = session.run(None, feed)
        token_output = next((value for value in outputs if value.ndim == 3), None)
        if token_output is None:
            raise RuntimeError("The ONNX model did not return token embeddings")
        batches.append(mean_pool_and_normalize(token_output, attention_mask))
    return np.vstack(batches)


def build_embedding_artifact(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    onnx_filename: str = DEFAULT_ONNX_FILENAME,
    maximum_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, Any]:
    """Build a validated, self-describing embedding cache."""
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise ValueError("Expected one two-dimensional embedding row per training record")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")
    if len({record["annotation_id"] for record in records}) != len(records):
        raise ValueError("Training records contain duplicate annotation IDs")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embeddings must be L2-normalized")

    return {
        "metadata": {
            "embedding_schema": EMBEDDING_SCHEMA,
            "model_id": model_id,
            "revision": revision,
            "onnx_filename": onnx_filename,
            "maximum_length": maximum_length,
            "dimension": int(embeddings.shape[1]),
            "normalized": True,
            "pooling": "attention-mask-aware mean pooling",
        },
        "records": [
            {
                "annotation_id": record["annotation_id"],
                "video_id": record["video_id"],
                "embedding": embedding.astype(float).tolist(),
            }
            for record, embedding in zip(records, embeddings, strict=True)
        ],
    }


def embedding_matrix_for_records(
    records: list[dict[str, Any]], artifact: dict[str, Any]
) -> np.ndarray:
    """Join cached embeddings to training rows by annotation identity."""
    metadata = artifact.get("metadata", {})
    if metadata.get("embedding_schema") != EMBEDDING_SCHEMA:
        raise ValueError(f"Expected embedding schema {EMBEDDING_SCHEMA}")
    dimension = metadata.get("dimension")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Embedding metadata contains an invalid dimension")
    if metadata.get("normalized") is not True:
        raise ValueError("Embedding metadata must declare L2 normalization")

    indexed: dict[str, dict[str, Any]] = {}
    for embedded in artifact.get("records", []):
        annotation_id = embedded.get("annotation_id")
        if not isinstance(annotation_id, str) or annotation_id in indexed:
            raise ValueError("Embedding records require unique annotation IDs")
        indexed[annotation_id] = embedded
    expected_ids = {record["annotation_id"] for record in records}
    if set(indexed) != expected_ids:
        missing = sorted(expected_ids - indexed.keys())
        extra = sorted(indexed.keys() - expected_ids)
        raise ValueError(
            f"Embedding IDs do not match training data; missing={missing}, extra={extra}"
        )

    rows: list[list[float]] = []
    for record in records:
        embedded = indexed[record["annotation_id"]]
        if embedded.get("video_id") != record["video_id"]:
            raise ValueError(f"{record['annotation_id']}: embedding video_id does not match")
        vector = embedded.get("embedding")
        if not isinstance(vector, list) or len(vector) != dimension:
            raise ValueError(f"{record['annotation_id']}: invalid embedding dimension")
        rows.append(vector)
    matrix = np.asarray(rows, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding artifact contains non-finite values")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4):
        raise ValueError("Embedding artifact contains unnormalized vectors")
    return matrix


def _records_with_features(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    include_handcrafted: bool,
) -> tuple[list[dict[str, Any]], tuple[str, ...], str]:
    """Create records for the semantic-only or hybrid representation."""
    embedding_fields = tuple(f"embedding_{index:03d}" for index in range(embeddings.shape[1]))
    feature_fields = (
        (*MODEL_FEATURE_FIELDS, *embedding_fields) if include_handcrafted else embedding_fields
    )
    schema = (
        f"{FEATURE_SCHEMA}_plus_{EMBEDDING_SCHEMA}" if include_handcrafted else EMBEDDING_SCHEMA
    )
    transformed = []
    for record, embedding in zip(records, embeddings, strict=True):
        features = dict(record["features"]) if include_handcrafted else {}
        features.update(zip(embedding_fields, embedding.tolist(), strict=True))
        transformed.append({**record, "feature_schema": schema, "features": features})
    return transformed, feature_fields, schema


def run_representation_ablation(
    records: list[dict[str, Any]],
    artifact: dict[str, Any],
    n_splits: int = 4,
    seed: int = 42,
    alpha: float = 10.0,
) -> dict[str, Any]:
    """Compare handcrafted, semantic-only, and hybrid ridge rankers on identical folds."""
    embeddings = embedding_matrix_for_records(records, artifact)
    variants: dict[str, tuple[list[dict[str, Any]], tuple[str, ...], str]] = {
        "handcrafted_ridge": (records, MODEL_FEATURE_FIELDS, FEATURE_SCHEMA),
        "semantic_ridge": _records_with_features(records, embeddings, False),
        "hybrid_ridge": _records_with_features(records, embeddings, True),
    }
    experiments: dict[str, dict[str, Any]] = {}
    for name, (variant_records, feature_fields, schema) in variants.items():
        experiments[name] = cross_validate_ridge(
            variant_records,
            n_splits=n_splits,
            seed=seed,
            alpha=alpha,
            feature_fields=feature_fields,
            feature_schema=schema,
            experiment=f"grouped_{name}_v1",
        )

    handcrafted = experiments["handcrafted_ridge"]
    models = {"fold_train_mean": handcrafted["models"]["fold_train_mean"]}
    final_models: dict[str, Any] = {}
    for name, experiment in experiments.items():
        models[name] = experiment["models"]["ridge"]
        final_models[name] = experiment["final_model"]

    predictions = []
    for index, record in enumerate(records):
        first = handcrafted["predictions"][index]
        predictions.append(
            {
                "annotation_id": record["annotation_id"],
                "video_id": record["video_id"],
                "fold": first["fold"],
                "actual": first["actual"],
                "actual_quality_score": first["actual_quality_score"],
                "predictions": {
                    "fold_train_mean": {
                        "targets": first["mean_prediction"],
                        "quality_score": first["mean_quality_score"],
                    },
                    **{
                        name: {
                            "targets": experiment["predictions"][index]["ridge_prediction"],
                            "quality_score": experiment["predictions"][index][
                                "ridge_quality_score"
                            ],
                        }
                        for name, experiment in experiments.items()
                    },
                },
            }
        )

    actual_quality = [prediction["actual_quality_score"] for prediction in predictions]
    handcrafted_quality = [
        prediction["predictions"]["handcrafted_ridge"]["quality_score"]
        for prediction in predictions
    ]
    hybrid_quality = [
        prediction["predictions"]["hybrid_ridge"]["quality_score"]
        for prediction in predictions
    ]

    return {
        "experiment": "pretrained_semantic_representation_ablation_v1",
        "dataset": handcrafted["dataset"],
        "embedding": artifact["metadata"],
        "targets": handcrafted["targets"],
        "cross_validation": handcrafted["cross_validation"],
        "regularization": {
            "model": "multi-output ridge regression",
            "alpha": alpha,
            "selection": "fixed before evaluating representation variants",
        },
        "models": models,
        "uncertainty": {
            "hybrid_vs_handcrafted": paired_video_bootstrap(
                records,
                actual_quality,
                handcrafted_quality,
                hybrid_quality,
                iterations=10_000,
                seed=seed,
            )
        },
        "final_models": final_models,
        "predictions": predictions,
    }


def encode_main() -> None:
    """Generate a private cache of normalized transcript embeddings."""
    parser = argparse.ArgumentParser(description="Encode CreatorCut transcripts with MiniLM")
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/training_clips.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/semantic_embeddings.json")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/huggingface"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--onnx-filename", default=DEFAULT_ONNX_FILENAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--maximum-length", type=int, default=DEFAULT_MAX_LENGTH)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    tokenizer_path, model_path = _download_model_files(
        args.model_id, args.revision, args.onnx_filename, args.cache_dir
    )
    embeddings = encode_texts_onnx(
        [record["transcript_text"] for record in records],
        tokenizer_path,
        model_path,
        batch_size=args.batch_size,
        maximum_length=args.maximum_length,
    )
    artifact = build_embedding_artifact(
        records,
        embeddings,
        model_id=args.model_id,
        revision=args.revision,
        onnx_filename=args.onnx_filename,
        maximum_length=args.maximum_length,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact), encoding="utf-8")
    print(
        f"Encoded {len(records)} transcripts into {embeddings.shape[1]} dimensions "
        f"and wrote {args.output}"
    )


def evaluate_main() -> None:
    """Run the frozen representation ablation and save private artifacts."""
    parser = argparse.ArgumentParser(description="Evaluate semantic clip representations")
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/training_clips.jsonl")
    )
    parser.add_argument(
        "--embeddings", type=Path, default=Path("data/processed/semantic_embeddings.json")
    )
    parser.add_argument(
        "--evaluation", type=Path, default=Path("data/processed/semantic_evaluation.json")
    )
    parser.add_argument(
        "--models", type=Path, default=Path("data/processed/semantic_models.json")
    )
    parser.add_argument(
        "--predictions", type=Path, default=Path("data/processed/semantic_predictions.jsonl")
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=10.0)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    artifact = json.loads(args.embeddings.read_text(encoding="utf-8"))
    evaluation = run_representation_ablation(
        records, artifact, n_splits=args.folds, seed=args.seed, alpha=args.alpha
    )
    predictions = evaluation.pop("predictions")
    final_models = evaluation.pop("final_models")
    args.evaluation.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    args.models.parent.mkdir(parents=True, exist_ok=True)
    args.models.write_text(json.dumps(final_models), encoding="utf-8")
    write_jsonl(predictions, args.predictions)

    summary = {
        name: {
            "quality_score": result["regression"]["quality_score"],
            "ranking": result["ranking"],
        }
        for name, result in evaluation["models"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote evaluation to {args.evaluation}")


if __name__ == "__main__":
    evaluate_main()
