import io
import zipfile

import pytest

from creatorcut.analytics import parse_youtube_analytics_export, performance_values


def youtube_zip(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return stream.getvalue()


def test_parse_small_studio_export_preserves_sparse_totals():
    payload = youtube_zip(
        {
            "Table data.csv": "Date,Views\n2026-09-03,0\n2026-09-04,2\n",
            "Totals.csv": (
                "Content,Views,Watch time (hours),Subscribers,Thumbnail impressions,"
                "Thumbnail click-through rate (%)\nTotal,2,0.0133,0,4,25\n"
            ),
        }
    )

    report = parse_youtube_analytics_export("example.zip", payload)

    assert report["totals"]["views"] == 2
    assert report["totals"]["watch_time_hours"] == pytest.approx(0.0133)
    assert report["totals"]["thumbnail_impressions"] == 4
    assert report["totals"]["thumbnail_ctr"] == 25
    assert report["retention_rows"] == []
    assert "Too little exposure" in report["warnings"][0]


def test_parse_full_short_report_accepts_studio_and_api_names():
    csv_text = (
        "date,video_id,creator_content_type,views,engaged_views,watch_time_minutes,"
        "average_view_duration_seconds,average_view_duration_percentage,likes,comments,"
        "shares,subscribers_gained,subscribers_lost,shown_in_feed,Stayed to watch (%)\n"
        "2026-09-01,abc,SHORTS,1200,800,180,13.5,75,90,8,14,5,1,2000,40\n"
    )

    report = parse_youtube_analytics_export("short.csv", csv_text.encode())
    values = performance_values(report)

    assert report["report_types"] == ["daily_performance"]
    assert values["engaged_views"] == 800
    assert values["watch_time_hours"] == 3
    assert values["average_view_percentage"] == 75
    assert values["chose_to_view_percentage"] == 40


def test_parse_retention_curve_normalizes_percent_positions():
    lines = ["Elapsed video time (%),Audience watch ratio,Relative retention performance"]
    lines.extend(
        f"{index},{0.90 - index / 200},{0.40 + index / 500}" for index in range(1, 21)
    )

    report = parse_youtube_analytics_export("retention.csv", "\n".join(lines).encode())

    assert report["report_types"] == ["audience_retention"]
    assert len(report["retention_rows"]) == 20
    assert report["retention_rows"][0]["elapsed_video_time_ratio"] == pytest.approx(0.01)
    assert report["retention_rows"][-1]["relative_retention_performance"] == pytest.approx(
        0.44
    )


def test_parse_retention_curve_preserves_rewatch_ratios_above_one():
    csv_text = (
        "elapsedVideoTimeRatio,audienceWatchRatio,relativeRetentionPerformance\n"
        "0.25,1.2,0.8\n"
    )

    report = parse_youtube_analytics_export("retention.csv", csv_text.encode())

    assert report["retention_rows"][0]["audience_watch_ratio"] == pytest.approx(1.2)


def test_parser_rejects_unrelated_csv():
    with pytest.raises(ValueError, match="supported YouTube analytics"):
        parse_youtube_analytics_export("other.csv", b"name,color\nexample,blue\n")
