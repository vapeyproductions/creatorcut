from datetime import UTC, datetime, timedelta

from creatorcut.product_store import ProductStore
from creatorcut.product_worker import ProcessingWorker


def queued_video(tmp_path):
    store = ProductStore(tmp_path / "creatorcut.sqlite")
    creator = store.ensure_creator("Queue test", "creator_queue")
    source = tmp_path / "source.mp4"
    source.touch()
    return store, store.create_video(creator["id"], source.name, source)


def test_new_upload_is_a_persistent_claimable_job(tmp_path):
    store, video = queued_video(tmp_path)

    queued = store.processing_job_for_video(video["id"])
    claimed = store.claim_next_processing_job("worker-one", lease_seconds=30)

    assert queued["status"] == "queued"
    assert claimed["video_id"] == video["id"]
    assert claimed["status"] == "running"
    assert claimed["attempt_count"] == 1
    assert store.claim_next_processing_job("worker-two", lease_seconds=30) is None


def test_expired_lease_is_recovered_by_another_worker(tmp_path):
    store, video = queued_video(tmp_path)
    started = datetime.now(UTC) + timedelta(seconds=1)
    first = store.claim_next_processing_job(
        "worker-one", lease_seconds=5, now=started
    )

    recovered = store.claim_next_processing_job(
        "worker-two", lease_seconds=5, now=started + timedelta(seconds=6)
    )

    assert recovered["id"] == first["id"]
    assert recovered["video_id"] == video["id"]
    assert recovered["attempt_count"] == 2
    assert recovered["lease_owner"] == "worker-two"


def test_failures_retry_then_become_terminal(tmp_path):
    store, video = queued_video(tmp_path)
    now = datetime.now(UTC) + timedelta(seconds=1)

    for attempt in range(1, 4):
        job = store.claim_next_processing_job(
            "worker-one", lease_seconds=30, now=now
        )
        saved = store.fail_processing_job(
            job["id"], "worker-one", "inference failed", now=now
        )
        assert saved["attempt_count"] == attempt
        assert saved["status"] == ("queued" if attempt < 3 else "failed")

    assert store.get_video(video["id"])["status"] == "failed"
    assert store.processing_queue_summary()["failed"] == 1


def test_worker_completes_a_job_and_persists_success(tmp_path):
    store, video = queued_video(tmp_path)

    class SuccessfulProcessor:
        def process_video(self, video_id):
            store.update_video(video_id, "ready", duration_seconds=42.0)

    worker = ProcessingWorker(
        store,
        SuccessfulProcessor(),
        worker_id="worker-test",
        lease_seconds=30,
        poll_seconds=0.01,
    )

    assert worker.run_once() is True
    assert worker.run_once() is False
    saved = store.get_video(video["id"])
    assert saved["status"] == "ready"
    assert saved["job"]["status"] == "succeeded"
    assert saved["job"]["attempt_count"] == 1


def test_worker_failure_remains_visible_and_retryable(tmp_path):
    store, video = queued_video(tmp_path)

    class FailingProcessor:
        def process_video(self, _video_id):
            raise RuntimeError("model unavailable")

    worker = ProcessingWorker(
        store,
        FailingProcessor(),
        worker_id="worker-test",
        lease_seconds=30,
        poll_seconds=0.01,
    )

    assert worker.run_once() is True
    saved = store.get_video(video["id"])
    assert saved["status"] == "queued"
    assert saved["job"]["status"] == "queued"
    assert saved["job"]["attempt_count"] == 1
    assert saved["job"]["last_error"] == "model unavailable"
