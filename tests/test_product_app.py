from pathlib import Path

import pytest

from creatorcut.product_app import ProductApplication, validate_performance_report
from creatorcut.product_pipeline import ProductProcessor
from creatorcut.product_store import ProductStore


def test_validate_performance_report_preserves_missing_metrics():
    assert validate_performance_report(
        {
            "platform": "youtube",
            "views": 1000,
            "likes": None,
            "comments": "",
            "shares": 4,
            "average_view_percentage": 72.5,
        }
    ) == {
        "platform": "youtube",
        "views": 1000,
        "likes": None,
        "comments": None,
        "shares": 4,
        "average_view_percentage": 72.5,
        "published_at": None,
    }


def test_validate_performance_report_requires_a_metric():
    with pytest.raises(ValueError, match="at least one"):
        validate_performance_report({"platform": "tiktok"})


def test_validate_performance_report_rejects_invalid_average():
    with pytest.raises(ValueError, match="between 0 and 100"):
        validate_performance_report(
            {"platform": "instagram", "average_view_percentage": 101}
        )


def test_health_verifies_the_committed_serving_release(tmp_path):
    store = ProductStore(tmp_path / "product.sqlite")
    processor = ProductProcessor(
        store,
        tmp_path / "work",
        Path("models/frozen_model_v1.json"),
        tmp_path / "models",
        tmp_path / "semantic",
        Path("models/serving_release_v1.json"),
    )
    application = ProductApplication(store, processor, tmp_path / "uploads")

    health = application.health()

    assert health["status"] == "ok"
    assert health["serving_release"]["release_id"] == "creatorcut_production_v1"
    assert health["serving_release"]["status"] == "frozen"
