import io
import zipfile

import pytest

from creatorcut.analytics import (
    parse_analytics_export,
    parse_youtube_analytics_export,
    performance_values,
)


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


def test_parse_tiktok_studio_metrics_into_platform_neutral_fields():
    payload = (
        b"Date,Video ID,Video views,Average watch time,Watched full video (%),"
        b"Likes,Comments,Shares,New followers\n"
        b"2026-09-01,tiktok-1,2400,00:00:18,42.5,190,14,31,8\n"
    )

    report = parse_analytics_export("tiktok", "content.csv", payload)
    values = performance_values(report)

    assert report["platform"] == "tiktok"
    assert values["views"] == 2400
    assert values["average_view_duration_seconds"] == pytest.approx(18.0)
    assert values["completion_rate_percentage"] == pytest.approx(42.5)
    assert values["follows"] == 8


def test_parse_instagram_reels_metrics_including_reach_saves_and_replays():
    payload = (
        b"Media ID,Reel views,Accounts reached,Average watch time,"
        b"Likes,Comments,Shares,Saves,Follows,Replays\n"
        b"ig-1,1800,1200,00:00:12.5,120,9,34,27,6,210\n"
    )

    values = performance_values(
        parse_analytics_export("instagram", "reels.csv", payload)
    )

    assert values["views"] == 1800
    assert values["reach"] == 1200
    assert values["saves"] == 27
    assert values["replays"] == 210
    assert values["average_view_duration_seconds"] == pytest.approx(12.5)


def test_parse_text_and_clock_watch_time_durations():
    tiktok = parse_analytics_export(
        "tiktok",
        "video.csv",
        b"Video ID,Views,Total play time,Average watch time\nabc,1200,1h 5m 30s,00:12\n",
    )
    instagram = parse_analytics_export(
        "instagram",
        "reel.csv",
        b"Reel ID,Plays,Total watch time,Average watch time\nabc,900,01:20:00,00:18\n",
    )

    assert tiktok["totals"]["watch_time_seconds"] == 3930
    assert tiktok["totals"]["average_view_duration_seconds"] == 12
    assert instagram["totals"]["watch_time_seconds"] == 4800
    assert performance_values(instagram)["watch_time_hours"] == pytest.approx(4 / 3)


def test_parse_google_sheets_xlsx_export():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
            <sheets><sheet name="Reels" sheetId="1" r:id="rId1"/></sheets></workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
            <Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
            <sheetData>
              <row r="1"><c r="A1" t="inlineStr"><is><t>Reel views</t></is></c>
              <c r="B1" t="inlineStr"><is><t>Saves</t></is></c></row>
              <row r="2"><c r="A2"><v>900</v></c><c r="B2"><v>23</v></c></row>
            </sheetData></worksheet>""",
        )

    report = parse_analytics_export("instagram", "reels.xlsx", stream.getvalue())

    assert report["totals"] == {"views": 900, "saves": 23}
    assert report["source_files"] == ["reels-Reels.csv"]
