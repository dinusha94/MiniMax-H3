"""FastAPI server: JSON job -> hosted H3-Context-IR -> self-hosted H3-Base -> MP4 on S3.

Run from this directory:
    uvicorn app:app --host 0.0.0.0 --port 8000 --env-file .env
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import settings
from pipeline import (
    DEFAULT_RESOLUTION,
    GenerationRequest,
    Media,
    MediaKind,
    Mode,
    PipelineError,
    generate,
    presign,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("h3_server")

app = FastAPI(
    title="MiniMax H3 Generation Server",
    description="Expands the request with hosted H3-Context-IR, renders it on self-hosted H3-Base, uploads the MP4 to S3.",
)


# ---------------------------------------------------------------- schemas


class JobRequest(BaseModel):
    mode: Mode = Field(description="t2va (text only), fl2va (first/last frame) or ref2va (references)")
    prompt: str = Field(description="Raw user instruction; Context-IR expands it")
    duration: int = Field(5, ge=4, le=15, description="Video length in seconds")
    ratio: str | None = Field(
        None, description="adaptive, 21:9, 16:9, 4:3, 1:1, 3:4, 9:16. Default: 16:9 for t2va, else adaptive"
    )
    resolution: Literal["360p", "480p", "768p"] = Field(
        DEFAULT_RESOLUTION,
        description="Output short edge: 768p (default, verified), or 480p / 360p (faster, unverified; 360p renders at 352)",
    )
    seed: int = 0
    # Media are presigned S3 URLs. They must stay valid until H3-Base has
    # loaded them, i.e. through Context-IR and any SGLang queueing.
    first_frame_url: str | None = Field(None, description="fl2va: first frame image")
    last_frame_url: str | None = Field(None, description="fl2va: last frame image")
    reference_image_urls: list[str] = Field([], description="ref2va: up to 9; order = 'reference image N'")
    reference_video_urls: list[str] = Field([], description="ref2va: up to 3, 2-15 s each")
    reference_audio_urls: list[str] = Field([], description="ref2va: up to 3, 2-15 s each")
    output_s3_folder: str | None = Field(
        None, description="s3://bucket/prefix/ for the result; defaults to S3_OUTPUT_FOLDER"
    )

    def to_generation_request(self) -> GenerationRequest:
        def one(url: str | None) -> Media | None:
            return Media(MediaKind.IMAGE, url) if url else None

        return GenerationRequest(
            mode=self.mode,
            prompt=self.prompt,
            duration=self.duration,
            # Text-only generation cannot use adaptive ratio; media modes follow the inputs.
            ratio=self.ratio or ("16:9" if self.mode is Mode.T2VA else "adaptive"),
            resolution=self.resolution,
            seed=self.seed,
            first_frame=one(self.first_frame_url),
            last_frame=one(self.last_frame_url),
            references=[
                *(Media(MediaKind.IMAGE, u) for u in self.reference_image_urls),
                *(Media(MediaKind.VIDEO, u) for u in self.reference_video_urls),
                *(Media(MediaKind.AUDIO, u) for u in self.reference_audio_urls),
            ],
        )


# ---------------------------------------------------------------- jobs


@dataclass
class Job:
    id: str
    mode: Mode
    resolution: str
    status: str = "queued"  # queued | running | succeeded | failed
    stage: str | None = None  # context_ir | h3_base | upload
    error: str | None = None
    expanded_prompt: str | None = None
    video_s3_uri: str | None = None
    h3_base: dict | None = None  # generation details reported by SGLang
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    stage_started: dict[str, float] = field(default_factory=dict)

    def enter_stage(self, stage: str) -> None:
        self.stage = stage
        self.stage_started[stage] = time.time()

    def timings(self) -> dict:
        """Wall-clock seconds per stage (as seen by this server) and for the whole job."""
        starts = list(self.stage_started.items())
        end = self.finished_at or time.time()
        stages = {
            f"{stage}_s": round((starts[i + 1][1] if i + 1 < len(starts) else end) - started, 2)
            for i, (stage, started) in enumerate(starts)
        }
        return {**stages, "total_s": round(end - self.created_at, 2)}

    def to_dict(self) -> dict:
        return {
            "job_id": self.id,
            "mode": self.mode.value,
            "resolution": self.resolution,
            "status": self.status,
            "stage": self.stage,
            "error": self.error,
            "expanded_prompt": self.expanded_prompt,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


# In-memory job registry: jobs are lost on restart and not shared across workers.
JOBS: dict[str, Job] = {}
_TASKS: set[asyncio.Task] = set()


async def _run_job(job: Job, req: GenerationRequest, s3_folder: str) -> None:
    job.status = "running"
    try:
        job.video_s3_uri = await generate(
            req,
            job.id,
            s3_folder,
            on_stage=job.enter_stage,
            on_prompt=lambda prompt: setattr(job, "expanded_prompt", prompt),
            on_h3_base=lambda stats: setattr(job, "h3_base", stats),
        )
        job.status = "succeeded"
        log.info("job %s succeeded: %s", job.id, job.video_s3_uri)
    except Exception as exc:  # surfaced to the client through GET /v1/jobs/{id}
        job.status, job.error = "failed", str(exc)
        log.exception("job %s failed at stage %s", job.id, job.stage)
    finally:
        job.finished_at = time.time()


def _get_job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


# ---------------------------------------------------------------- routes


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/jobs", status_code=202, summary="Submit a generation job")
async def create_job(body: JobRequest) -> dict:
    """Returns immediately with a job id; poll GET /v1/jobs/{job_id}, then GET /v1/jobs/{job_id}/result."""
    s3_folder = body.output_s3_folder or settings.s3_output_folder
    if not s3_folder or not s3_folder.startswith("s3://"):
        raise HTTPException(400, "output_s3_folder (or S3_OUTPUT_FOLDER) must be an s3://bucket/prefix/ URI")
    req = body.to_generation_request()
    try:
        req.validate()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job = Job(id=uuid.uuid4().hex, mode=body.mode, resolution=req.resolution)
    JOBS[job.id] = job
    task = asyncio.create_task(_run_job(job, req, s3_folder))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return job.to_dict()


@app.get("/v1/jobs/{job_id}", summary="Poll job status")
async def get_job(job_id: str) -> dict:
    return _get_job(job_id).to_dict()


@app.get("/v1/jobs/{job_id}/result", summary="Get the generated video")
async def get_job_result(job_id: str) -> dict:
    """Returns the S3 location plus a freshly presigned download URL (409 until the job succeeds)."""
    job = _get_job(job_id)
    if job.status != "succeeded":
        detail = f"job is {job.status}" + (f": {job.error}" if job.error else "")
        raise HTTPException(409, detail)
    try:
        video_url = await presign(job.video_s3_uri)
    except PipelineError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {
        "job_id": job.id,
        "status": job.status,
        "video_s3_uri": job.video_s3_uri,
        "video_url": video_url,
        "video_url_expires_in": settings.s3_presign_expires_s,
        "expanded_prompt": job.expanded_prompt,
        "generation": {"timings": job.timings(), "h3_base": job.h3_base},
    }
