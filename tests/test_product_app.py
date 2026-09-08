import io
import json
from pathlib import Path

import pytest

from creatorcut.product_app import (
    ProductApplication,
    create_handler,
    validate_performance_report,
)
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
        "saves": None,
        "reach": None,
        "follows": None,
        "profile_visits": None,
        "replays": None,
        "average_view_percentage": 72.5,
        "completion_rate_percentage": None,
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


def test_login_failure_budget_can_be_cleared_after_success(tmp_path):
    store = ProductStore(tmp_path / "product.sqlite")

    class ProcessorStub:
        pass

    application = ProductApplication(store, ProcessorStub(), tmp_path / "uploads")
    for _ in range(5):
        assert application.login_allowed("127.0.0.1:creator@example.com")
        application.record_login_failure("127.0.0.1:creator@example.com")
    assert not application.login_allowed("127.0.0.1:creator@example.com")
    application.clear_login_failures("127.0.0.1:creator@example.com")
    assert application.login_allowed("127.0.0.1:creator@example.com")


def test_http_sessions_enforce_creator_ownership_csrf_and_admin_role(tmp_path):
    store = ProductStore(tmp_path / "product.sqlite")
    admin = store.register_account(
        "Administrator", "admin@example.com", "administrator password", is_admin=True
    )
    creator = store.register_account(
        "Creator", "creator@example.com", "creator password long"
    )
    source = tmp_path / "source.mp4"
    source.touch()
    admin_video = store.create_video(admin["creator_id"], source.name, source)
    creator_video = store.create_video(creator["creator_id"], "creator.mp4", source)
    store.save_ranked_clips(
        creator_video["id"],
        [
            {
                "id": f"{creator_video['id']}_clip_1",
                "rank": 1,
                "start_seconds": 5.0,
                "end_seconds": 35.0,
                "duration_seconds": 30.0,
                "transcript_text": "A useful complete example.",
                "semantic_embedding": [0.0, 1.0],
                "global_score": 4.0,
                "personalized_score": 4.0,
                "predicted_targets": {
                    "hook": 4.0,
                    "completeness": 4.0,
                    "payoff": 4.0,
                    "clarity": 4.0,
                },
                "explanation": "A concise complete moment.",
                "community_lineage": {"editorial": {"active": False}},
            }
        ],
    )

    class ProcessorStub:
        work_dir = tmp_path / "work"

    application = ProductApplication(
        store,
        ProcessorStub(),
        tmp_path / "uploads",
        admin_dashboard_enabled=True,
    )
    handler = create_handler(application)

    class RequestSocket:
        def __init__(self, request_bytes):
            self.input = io.BytesIO(request_bytes)
            self.output = io.BytesIO()

        def makefile(self, mode, *args, **kwargs):
            return self.input if "r" in mode else self.output

        def sendall(self, data):
            self.output.write(data)

    def request(method, path, body=None, headers=None):
        payload = json.dumps(body).encode() if body is not None else None
        request_headers = dict(headers or {})
        if payload is not None:
            request_headers["Content-Type"] = "application/json"
            request_headers["Content-Length"] = str(len(payload))
        head = [f"{method} {path} HTTP/1.1", "Host: creatorcut.test"]
        head.extend(f"{name}: {value}" for name, value in request_headers.items())
        request_socket = RequestSocket(
            ("\r\n".join(head) + "\r\n\r\n").encode() + (payload or b"")
        )
        handler(request_socket, ("127.0.0.1", 12345), object())
        response_head, response_body = request_socket.output.getvalue().split(
            b"\r\n\r\n", 1
        )
        response_lines = response_head.decode().split("\r\n")
        status = int(response_lines[0].split()[1])
        headers_result = {
            name: value
            for name, value in (
                line.split(": ", 1) for line in response_lines[1:] if ": " in line
            )
        }
        result = (
            json.loads(response_body)
            if headers_result.get("Content-Type", "").startswith("application/json")
            else response_body
        )
        return status, result, headers_result

    status, login, login_headers = request(
        "POST",
        "/api/auth/login",
        {"email": creator["email"], "password": "creator password long"},
    )
    assert status == 200
    assert "HttpOnly" in login_headers["Set-Cookie"]
    assert "SameSite=Lax" in login_headers["Set-Cookie"]
    assert "creatorcut_session=" in login_headers["Set-Cookie"]
    creator_cookie = login_headers["Set-Cookie"].split(";", 1)[0]
    creator_headers = {"Cookie": creator_cookie}
    status, _, _ = request(
        "GET", f"/api/videos/{admin_video['id']}", headers=creator_headers
    )
    assert status == 404
    status, own_video, _ = request(
        "GET", f"/api/videos/{creator_video['id']}", headers=creator_headers
    )
    assert status == 200
    assert own_video["creator_summary"] == {
        "personalization_active": False,
        "platform_performance": {
            platform: {
                "semantic_active": False,
                "insight_summary": None,
                "positive_semantic_trends": [],
                "negative_semantic_trends": [],
            }
            for platform in ("youtube", "instagram", "tiktok")
        },
    }
    assert "global_score" not in own_video["clips"][0]
    assert "community_lineage" not in own_video["clips"][0]
    status, _, _ = request("GET", "/admin", headers=creator_headers)
    assert status == 403
    status, settings, _ = request(
        "GET", "/api/account/contribution-settings", headers=creator_headers
    )
    assert status == 200
    assert settings["performance_enabled"] is False
    assert "editorial_enabled" not in settings
    status, settings, _ = request(
        "POST",
        "/api/account/contribution-settings",
        {"editorial_enabled": False, "performance_enabled": True},
        {**creator_headers, "X-CSRF-Token": login["csrf_token"]},
    )
    assert status == 200
    assert settings["performance_enabled"] is True
    status, _, _ = request(
        "GET", "/api/account/model-report", headers=creator_headers
    )
    assert status == 403
    status, _, _ = request(
        "POST", "/api/auth/logout", {"logout": True}, creator_headers
    )
    assert status == 403
    status, _, _ = request(
        "POST",
        "/api/auth/logout",
        {"logout": True},
        {**creator_headers, "X-CSRF-Token": login["csrf_token"]},
    )
    assert status == 200

    status, admin_login, admin_headers = request(
        "POST",
        "/api/auth/login",
        {"email": admin["email"], "password": "administrator password"},
    )
    assert status == 200
    admin_cookie = admin_headers["Set-Cookie"].split(";", 1)[0]
    status, _, protected_headers = request(
        "GET", "/admin", headers={"Cookie": admin_cookie}
    )
    assert status == 200
    assert "frame-ancestors 'none'" in protected_headers["Content-Security-Policy"]
    assert admin_login["account"]["is_admin"] is True
