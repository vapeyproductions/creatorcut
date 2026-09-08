"""Validate the immutable components and policies in a CreatorCut serving release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from creatorcut.holdout import canonical_sha256
from creatorcut.product_store import (
    CONTRIBUTION_POLICY_VERSION,
    MAXIMUM_COMMUNITY_EDITORIAL_ADJUSTMENT,
    MAXIMUM_COMMUNITY_PERFORMANCE_ADJUSTMENT,
    MAXIMUM_PERSONALIZATION_ADJUSTMENT,
    MAXIMUM_SEMANTIC_PERFORMANCE_ADJUSTMENT,
    MAXIMUM_SOURCE_RETENTION_ADJUSTMENT,
    MAXIMUM_STRUCTURED_PERFORMANCE_ADJUSTMENT,
    MINIMUM_COMMUNITY_EDITORIAL_CREATORS,
    MINIMUM_COMMUNITY_EDITORIAL_DECISIONS,
    MINIMUM_COMMUNITY_PER_CREATOR_CLIPS,
    MINIMUM_COMMUNITY_PERFORMANCE_CLIPS,
    MINIMUM_COMMUNITY_PERFORMANCE_CREATORS,
    MINIMUM_EDITORIAL_DECISIONS,
    MINIMUM_PERFORMANCE_EXAMPLES,
    MINIMUM_PERFORMANCE_VIEWS,
    MINIMUM_SOURCE_RETENTION_POINTS,
    MINIMUM_SOURCE_RETENTION_VIEWS,
)
from creatorcut.publishability import (
    MAXIMUM_PUBLISHABILITY_ADJUSTMENT,
    PUBLISHABILITY_RULE_VERSION,
)
from creatorcut.reframing import REFRAMING_VERSION
from creatorcut.repurposing import REPURPOSING_VERSION

SERVING_RELEASE_SCHEMA = "creatorcut_serving_release_v1"


def _expect(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{label} is {actual!r}; expected {expected!r}")


def validate_serving_release(
    manifest_path: Path, frozen_model_path: Path
) -> dict[str, Any]:
    """Verify a release manifest against the exact artifact and runtime constants."""
    if not manifest_path.is_file():
        raise ValueError(f"Serving release manifest is missing: {manifest_path}")
    if not frozen_model_path.is_file():
        raise ValueError(f"Frozen model artifact is missing: {frozen_model_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_model = json.loads(frozen_model_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    _expect(errors, "release schema", manifest.get("schema"), SERVING_RELEASE_SCHEMA)
    _expect(errors, "release status", manifest.get("status"), "frozen")
    global_ranker = manifest.get("global_ranker", {})
    _expect(
        errors,
        "global ranker digest",
        canonical_sha256(frozen_model),
        global_ranker.get("canonical_sha256"),
    )
    _expect(
        errors,
        "global ranker schema",
        frozen_model.get("freeze_schema"),
        global_ranker.get("freeze_schema"),
    )
    components = manifest.get("components", {})
    _expect(
        errors,
        "publishability version",
        PUBLISHABILITY_RULE_VERSION,
        components.get("publishability"),
    )
    _expect(
        errors,
        "reframing version",
        REFRAMING_VERSION,
        components.get("reframing"),
    )
    _expect(
        errors,
        "repurposing version",
        REPURPOSING_VERSION,
        components.get("repurposing"),
    )
    _expect(
        errors,
        "community learning version",
        CONTRIBUTION_POLICY_VERSION,
        components.get("community_learning"),
    )
    policy = manifest.get("personalization_policy", {})
    expected_policy = {
        "minimum_editorial_decisions": MINIMUM_EDITORIAL_DECISIONS,
        "maximum_editorial_adjustment": MAXIMUM_PERSONALIZATION_ADJUSTMENT,
        "minimum_performance_examples": MINIMUM_PERFORMANCE_EXAMPLES,
        "minimum_performance_views": MINIMUM_PERFORMANCE_VIEWS,
        "maximum_structured_performance_adjustment": (
            MAXIMUM_STRUCTURED_PERFORMANCE_ADJUSTMENT
        ),
        "maximum_semantic_performance_adjustment": (
            MAXIMUM_SEMANTIC_PERFORMANCE_ADJUSTMENT
        ),
        "minimum_source_retention_views": MINIMUM_SOURCE_RETENTION_VIEWS,
        "minimum_source_retention_points": MINIMUM_SOURCE_RETENTION_POINTS,
        "maximum_source_retention_adjustment": MAXIMUM_SOURCE_RETENTION_ADJUSTMENT,
        "maximum_publishability_adjustment": MAXIMUM_PUBLISHABILITY_ADJUSTMENT,
        "minimum_community_editorial_creators": MINIMUM_COMMUNITY_EDITORIAL_CREATORS,
        "minimum_community_editorial_decisions": MINIMUM_COMMUNITY_EDITORIAL_DECISIONS,
        "maximum_community_editorial_adjustment": (
            MAXIMUM_COMMUNITY_EDITORIAL_ADJUSTMENT
        ),
        "minimum_community_performance_creators": (
            MINIMUM_COMMUNITY_PERFORMANCE_CREATORS
        ),
        "minimum_community_performance_clips": MINIMUM_COMMUNITY_PERFORMANCE_CLIPS,
        "minimum_community_clips_per_creator": MINIMUM_COMMUNITY_PER_CREATOR_CLIPS,
        "maximum_community_performance_adjustment": (
            MAXIMUM_COMMUNITY_PERFORMANCE_ADJUSTMENT
        ),
    }
    for field, expected in expected_policy.items():
        _expect(errors, f"personalization policy {field}", policy.get(field), expected)
    evaluation_evidence = manifest.get("evaluation_evidence", {})
    external_holdout = evaluation_evidence.get("external_holdout_v1", {})
    _expect(
        errors,
        "external holdout model digest",
        external_holdout.get("frozen_model_sha256"),
        global_ranker.get("canonical_sha256"),
    )
    if errors:
        raise ValueError("Serving release validation failed: " + "; ".join(errors))
    return {
        "release_id": manifest["release_id"],
        "status": manifest["status"],
        "frozen_at": manifest["frozen_at"],
        "global_ranker_schema": global_ranker["freeze_schema"],
        "global_ranker_sha256": global_ranker["canonical_sha256"],
        "components": components,
        "evaluation_evidence": evaluation_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a frozen CreatorCut serving release")
    parser.add_argument(
        "--manifest", type=Path, default=Path("models/serving_release_v1.json")
    )
    parser.add_argument(
        "--frozen-model", type=Path, default=Path("models/frozen_model_v1.json")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate_serving_release(args.manifest, args.frozen_model),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
