"""Single-purpose P1 worker that consumes disk-backed user-video jobs."""

from __future__ import annotations

import signal
import time

from video_rag.diagnosis import VideoJobManager, run_diagnosis_job
from video_rag.retrieval import VideoEvidenceRetriever
from retrieval_engine import RetrievalEngine


running = True


def stop_worker(_signum, _frame) -> None:
    global running
    running = False


def main() -> None:
    signal.signal(signal.SIGINT, stop_worker)
    signal.signal(signal.SIGTERM, stop_worker)
    manager = VideoJobManager()
    retriever = VideoEvidenceRetriever()
    manual_engine = RetrievalEngine(rerank_enabled=False)
    manual_engine.ensure_index()
    while running:
        job_id = manager.claim_next()
        if job_id:
            run_diagnosis_job(manager, job_id, retriever, manual_engine)
        else:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
