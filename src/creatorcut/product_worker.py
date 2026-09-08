"""Durable single-job inference worker for CreatorCut uploads."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import threading
import uuid
from pathlib import Path

from creatorcut.product_pipeline import ProductProcessor
from creatorcut.product_store import ProductStore
from creatorcut.runtime_logging import configure_logging, log_event

LOGGER = logging.getLogger("creatorcut.worker")


def default_worker_id() -> str:
    """Return an operationally useful, process-unique worker identity."""
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


class ProcessingWorker:
    """Claim persistent jobs, maintain leases, and record retries."""

    def __init__(
        self,
        store: ProductStore,
        processor: ProductProcessor,
        worker_id: str | None = None,
        lease_seconds: float = 120.0,
        poll_seconds: float = 1.0,
    ) -> None:
        if lease_seconds < 3:
            raise ValueError("lease_seconds must be at least 3 seconds")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.store = store
        self.processor = processor
        self.worker_id = worker_id or default_worker_id()
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds

    def _heartbeat(self, job_id: int, stop: threading.Event) -> None:
        interval = max(1.0, self.lease_seconds / 3)
        while not stop.wait(interval):
            if not self.store.extend_processing_lease(
                job_id, self.worker_id, self.lease_seconds
            ):
                log_event(
                    LOGGER,
                    "processing_job_lease_lost",
                    job_id=job_id,
                    worker_id=self.worker_id,
                )
                return

    def run_once(self) -> bool:
        """Process one available job; return false when the queue is empty."""
        job = self.store.claim_next_processing_job(
            self.worker_id, lease_seconds=self.lease_seconds
        )
        if job is None:
            return False
        job_id = int(job["id"])
        video_id = str(job["video_id"])
        log_event(
            LOGGER,
            "processing_job_started",
            job_id=job_id,
            video_id=video_id,
            attempt=job["attempt_count"],
            worker_id=self.worker_id,
        )
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(job_id, heartbeat_stop),
            name=f"creatorcut-lease-{job_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            self.processor.process_video(video_id)
        except Exception as error:
            heartbeat_stop.set()
            heartbeat.join(timeout=2)
            retry_delay = min(60.0, 2 ** max(0, int(job["attempt_count"]) - 1))
            saved = self.store.fail_processing_job(
                job_id,
                self.worker_id,
                str(error),
                retry_delay_seconds=retry_delay,
            )
            log_event(
                LOGGER,
                "processing_job_retry_scheduled"
                if saved["status"] == "queued"
                else "processing_job_failed",
                job_id=job_id,
                video_id=video_id,
                attempt=saved["attempt_count"],
                max_attempts=saved["max_attempts"],
                retry_delay_seconds=retry_delay if saved["status"] == "queued" else None,
                error=str(error)[:500],
            )
            return True
        heartbeat_stop.set()
        heartbeat.join(timeout=2)
        self.store.complete_processing_job(job_id, self.worker_id)
        log_event(
            LOGGER,
            "processing_job_succeeded",
            job_id=job_id,
            video_id=video_id,
            attempt=job["attempt_count"],
            worker_id=self.worker_id,
        )
        return True

    def run_forever(self, stop: threading.Event | None = None) -> None:
        """Poll until shutdown, processing at most one expensive ML job at a time."""
        stop_event = stop or threading.Event()
        log_event(LOGGER, "processing_worker_started", worker_id=self.worker_id)
        while not stop_event.is_set():
            if not self.run_once():
                stop_event.wait(self.poll_seconds)
        log_event(LOGGER, "processing_worker_stopped", worker_id=self.worker_id)


def build_processor(
    store: ProductStore,
    work_dir: Path,
    frozen_model: Path,
    model_cache: Path,
    semantic_cache: Path,
) -> ProductProcessor:
    """Build the inference pipeline from deployment-facing paths."""
    return ProductProcessor(
        store,
        work_dir,
        frozen_model,
        model_cache,
        semantic_cache,
    )


def main() -> None:
    """Run a standalone durable inference worker."""
    parser = argparse.ArgumentParser(description="Run the CreatorCut inference worker")
    parser.add_argument("--database", type=Path, default=Path("data/product/creatorcut.sqlite"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/product/work"))
    parser.add_argument(
        "--frozen-model", type=Path, default=Path("models/frozen_model_v1.json")
    )
    parser.add_argument("--model-cache", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--semantic-cache", type=Path, default=Path("artifacts/huggingface"))
    parser.add_argument("--worker-id")
    parser.add_argument("--lease-seconds", type=float, default=120.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    store = ProductStore(args.database)
    processor = build_processor(
        store,
        args.work_dir,
        args.frozen_model,
        args.model_cache,
        args.semantic_cache,
    )
    worker = ProcessingWorker(
        store,
        processor,
        worker_id=args.worker_id,
        lease_seconds=args.lease_seconds,
        poll_seconds=args.poll_seconds,
    )
    if args.once:
        worker.run_once()
        return

    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    worker.run_forever(stop)


if __name__ == "__main__":
    main()
