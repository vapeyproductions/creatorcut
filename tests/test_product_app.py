import pytest

from creatorcut.product_app import validate_performance_report


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
