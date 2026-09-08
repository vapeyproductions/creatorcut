import json
from pathlib import Path

import pytest

from creatorcut.serving_release import validate_serving_release


def test_committed_serving_release_verifies_exact_runtime():
    release = validate_serving_release(
        Path("models/serving_release_v1.json"),
        Path("models/frozen_model_v1.json"),
    )

    assert release["release_id"] == "creatorcut_production_v1"
    assert release["status"] == "frozen"
    assert len(release["global_ranker_sha256"]) == 64
    assert release["components"]["publishability"].endswith("_v1")


def test_serving_release_detects_a_changed_model_artifact(tmp_path):
    model = json.loads(Path("models/frozen_model_v1.json").read_text(encoding="utf-8"))
    model["selected_model"] = "silently_changed_model"
    model_path = tmp_path / "changed-model.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")

    with pytest.raises(ValueError, match="global ranker digest"):
        validate_serving_release(
            Path("models/serving_release_v1.json"),
            model_path,
        )


def test_serving_release_detects_a_changed_runtime_policy(tmp_path):
    manifest = json.loads(
        Path("models/serving_release_v1.json").read_text(encoding="utf-8")
    )
    manifest["personalization_policy"]["minimum_performance_examples"] = 1
    manifest_path = tmp_path / "changed-release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="minimum_performance_examples"):
        validate_serving_release(
            manifest_path,
            Path("models/frozen_model_v1.json"),
        )
