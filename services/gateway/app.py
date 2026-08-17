"""Phase-1 HTTP gateway: validated uploads -> bounded async worker -> status API."""
from __future__ import annotations

import asyncio
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from services.inference.detector import YoloOnnxDetector

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_IMAGES = {"image/jpeg", "image/png", "image/webp"}
REQUESTS = Counter("inference_requests_total", "Inference requests", ["status"])
QUEUE_DEPTH = Gauge("inference_queue_depth", "Pending requests")
LATENCY = Histogram("inference_latency_seconds", "End-to-end inference latency")


@dataclass
class Job:
    request_id: str
    content: bytes
    filename: str
    submitted_at: float = field(default_factory=time.time)
    status: Literal["QUEUED", "PROCESSING", "COMPLETED", "FAILED"] = "QUEUED"
    queue_latency_ms: float | None = None
    processing_time_ms: float | None = None
    model_execution_ms: float | None = None
    detections: list[dict] = field(default_factory=list)
    error: str | None = None


class Dispatcher:
    """Local Phase-1 queue. Replace this seam with Redis Streams in Phase 4."""
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue[str] | None = None
        self.worker: asyncio.Task | None = None
        self.detector = YoloOnnxDetector(
            os.getenv("MODEL_PATH", "models/yolov8n.onnx"),
            int(os.getenv("INPUT_SIZE", "640")),
            float(os.getenv("CONFIDENCE_THRESHOLD", "0.35")),
            float(os.getenv("NMS_THRESHOLD", "0.45")),
        )

    async def start(self) -> None:
        self.queue = asyncio.Queue(maxsize=int(os.getenv("MAX_QUEUE_SIZE", "32")))
        self.worker = asyncio.create_task(self._consume())

    async def stop(self) -> None:
        if self.worker:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass
        self.worker = self.queue = None

    async def submit(self, job: Job) -> None:
        if self.queue is None or self.queue.full():
            raise HTTPException(503, "Inference queue is full; retry later.")
        self.jobs[job.request_id] = job
        self.queue.put_nowait(job.request_id)
        QUEUE_DEPTH.set(self.queue.qsize())

    async def _consume(self) -> None:
        assert self.queue is not None
        while True:
            request_id = await self.queue.get()
            job, started = self.jobs[request_id], time.perf_counter()
            job.status = "PROCESSING"
            job.queue_latency_ms = round((time.time() - job.submitted_at) * 1000, 2)
            try:
                with Image.open(io.BytesIO(job.content)) as image:
                    detections, model_ms = await asyncio.to_thread(self.detector.detect, image.copy())
                job.detections = [asdict(item) for item in detections]
                job.model_execution_ms, job.status = round(model_ms, 2), "COMPLETED"
                REQUESTS.labels("success").inc()
            except Exception as error:
                job.status, job.error = "FAILED", str(error)
                REQUESTS.labels("error").inc()
            finally:
                job.processing_time_ms = round((time.perf_counter() - started) * 1000, 2)
                LATENCY.observe(job.processing_time_ms / 1000)
                self.queue.task_done()
                QUEUE_DEPTH.set(self.queue.qsize())


dispatcher = Dispatcher()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await dispatcher.start()
    yield
    await dispatcher.stop()


app = FastAPI(title="AI Inference Lab — Phase 1", lifespan=lifespan)


@app.get("/")
async def frontend():
    return FileResponse(ROOT / "web" / "index.html")


@app.post("/api/v1/inference/image", status_code=202)
async def create_image_inference(file: UploadFile = File(...)):
    if file.content_type not in SUPPORTED_IMAGES:
        raise HTTPException(415, "Supported images: JPEG, PNG, WebP.")
    content = await file.read()
    if not content:
        raise HTTPException(422, "Upload is empty.")
    if len(content) > int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024:
        raise HTTPException(413, "Upload exceeds configured maximum.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
    except Exception as error:
        raise HTTPException(422, "Invalid image upload.") from error
    job = Job(str(uuid.uuid4()), content, file.filename or "upload")
    await dispatcher.submit(job)
    return {"request_id": job.request_id, "status": job.status, "status_url": f"/api/v1/inference/{job.request_id}"}


@app.get("/api/v1/inference/{request_id}")
async def get_inference(request_id: str):
    job = dispatcher.jobs.get(request_id)
    if not job:
        raise HTTPException(404, "Unknown request ID.")
    result = asdict(job)
    result.pop("content")
    return result | {"runtime": "onnxruntime", "execution_provider": "CPUExecutionProvider", "device": "cpu"}


@app.get("/api/v1/health")
async def health(): return {"status": "healthy"}


@app.get("/api/v1/ready")
async def ready(): return {"status": "ready", "worker_running": bool(dispatcher.worker and not dispatcher.worker.done())}


@app.get("/api/v1/metrics")
async def metrics(): return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
