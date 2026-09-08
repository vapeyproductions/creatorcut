"""Export a versioned offline snapshot from CreatorCut's operational feedback store."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from creatorcut.product_store import ProductStore

FEEDBACK_SNAPSHOT_SCHEMA = "creatorcut_operational_feedback_v1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_feedback_snapshot(
    store: ProductStore, creator_id: str | None = None
) -> dict[str, Any]:
    """Build an integrity-hashed export without writing or exposing media paths."""
    records = store.feedback_training_records(creator_id)
    summary = {
        "schema": FEEDBACK_SNAPSHOT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "creator_filter": creator_id,
        "record_count": len(records),
        "creator_count": len({record["creator_id"] for record in records}),
        "video_count": len({record["video_id"] for record in records}),
        "labeled_record_count": sum(
            record["editorial_label"] is not None for record in records
        ),
        "performance_record_count": sum(
            record["latest_performance"] is not None for record in records
        ),
        "model_versions": sorted(
            {
                record["ranking_model_version"]
                for record in records
                if record["ranking_model_version"]
            }
        ),
        "repurposing_feedback_count": sum(
            len(record["repurposing_feedback"]) for record in records
        ),
        "repurposing_algorithm_versions": sorted(
            {
                item["algorithm_version"]
                for record in records
                for item in record["repurposing_feedback"]
            }
        ),
        "records_sha256": canonical_sha256(records),
    }
    return {"summary": summary, "records": records}


def export_feedback_snapshot(
    database: Path,
    output_records: Path,
    output_summary: Path,
    creator_id: str | None = None,
) -> dict[str, Any]:
    """Write JSONL training records plus a small integrity and lineage manifest."""
    snapshot = build_feedback_snapshot(ProductStore(database), creator_id)
    records = snapshot["records"]
    output_records.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_records.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = snapshot["summary"]
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export operational feedback for offline analysis")
    parser.add_argument(
        "--database", type=Path, default=Path("data/product/creatorcut.sqlite")
    )
    parser.add_argument(
        "--output-records",
        type=Path,
        default=Path("data/product/feedback_snapshot.jsonl"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("data/product/feedback_snapshot_summary.json"),
    )
    parser.add_argument("--creator-id")
    args = parser.parse_args()
    summary = export_feedback_snapshot(
        args.database,
        args.output_records,
        args.output_summary,
        args.creator_id,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
