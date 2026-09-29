"""Hosted H3-Context-IR -> self-hosted H3-Base (SGLang) -> S3 generation pipeline.

Request formats follow scripts/readme/full-2k-*-h3-context-ir.sh and
scripts/readme/full-2k-*-h3-base.sh. Input media are presigned S3 (or other
public http(s)) URLs, passed unchanged to both Context-IR and SGLang.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx

import s3_utils
from config import settings

RATIOS = {"adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
MAX_REF_IMAGES, MAX_REF_VIDEOS, MAX_REF_AUDIOS, MAX_REF_TOTAL = 9, 3, 3, 12
POLL_RETRIES = 5


class Mode(str, Enum):
    T2VA = "t2va"  # text only
    FL2VA = "fl2va"  # first and/or last frame
    REF2VA = "ref2va"  # reference images / videos / audio


class MediaKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class PipelineError(RuntimeError):
    """An upstream service (Context-IR, SGLang, or S3) failed."""


@dataclass
class Media:
    kind: MediaKind
    url: str


@dataclass
class GenerationRequest:
    mode: Mode
    prompt: str
    duration: int
    ratio: str
    seed: int = 0
    first_frame: Media | None = None
    last_frame: Media | None = None
    # Order matters: "reference image 1", "reference audio 2", ... in the
    # prompt refer to the n-th item of that kind in this list.
    references: list[Media] = field(default_factory=list)

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if not 4 <= self.duration <= 15:
            raise ValueError("duration must be between 4 and 15 seconds")
        if self.ratio not in RATIOS:
            raise ValueError(f"ratio must be one of {sorted(RATIOS)}")
        for media in filter(None, [self.first_frame, self.last_frame, *self.references]):
            if not media.url.startswith(("https://", "http://")):
                raise ValueError(f"media URLs must be http(s) (presigned S3 links), got {media.url[:80]!r}")

        has_frames = self.first_frame or self.last_frame
        if self.mode is Mode.T2VA:
            if has_frames or self.references:
                raise ValueError("t2va is text only; use fl2va or ref2va to send media")
            if self.ratio == "adaptive":
                raise ValueError("t2va needs an explicit ratio (adaptive is not allowed)")
        elif self.mode is Mode.FL2VA:
            if not has_frames:
                raise ValueError("fl2va needs a first_frame_url, a last_frame_url, or both")
            if self.references:
                raise ValueError("fl2va does not accept references; use ref2va")
        else:
            if has_frames:
                raise ValueError("ref2va does not accept first/last frames; use fl2va")
            counts = {k: sum(m.kind is k for m in self.references) for k in MediaKind}
            if not self.references:
                raise ValueError("ref2va needs at least one reference image, video, or audio")
            if counts[MediaKind.IMAGE] > MAX_REF_IMAGES:
                raise ValueError(f"at most {MAX_REF_IMAGES} reference images")
            if counts[MediaKind.VIDEO] > MAX_REF_VIDEOS:
                raise ValueError(f"at most {MAX_REF_VIDEOS} reference videos")
            if counts[MediaKind.AUDIO] > MAX_REF_AUDIOS:
                raise ValueError(f"at most {MAX_REF_AUDIOS} reference audio clips")
            if len(self.references) > MAX_REF_TOTAL:
                raise ValueError(f"at most {MAX_REF_TOTAL} reference files in total")


def _context_ir_media(media: Media, role: str) -> dict:
    key = f"{media.kind.value}_url"
    return {"type": key, key: {"url": media.url}, "role": role}


def build_context_ir_payload(req: GenerationRequest) -> dict:
    content: list[dict] = [{"type": "text", "text": req.prompt}]
    if req.first_frame:
        content.append(_context_ir_media(req.first_frame, "first_frame"))
    if req.last_frame:
        content.append(_context_ir_media(req.last_frame, "last_frame"))
    for media in req.references:
        content.append(_context_ir_media(media, f"reference_{media.kind.value}"))
    return {"model": "MiniMax-H3", "content": content, "duration": req.duration, "ratio": req.ratio}


def build_h3_base_payload(req: GenerationRequest, expanded_prompt: str) -> dict:
    conditions: list[dict] = []
    if req.first_frame:
        conditions.append({"type": "image", "uri": req.first_frame.url, "role": "keyframe", "frame_index": 0})
    if req.last_frame:
        conditions.append({"type": "image", "uri": req.last_frame.url, "role": "keyframe", "frame_index": -1})
    for media in req.references:
        conditions.append({"type": media.kind.value, "uri": media.url, "role": "reference"})
    return {
        "task": req.mode.value,
        "prompt": expanded_prompt,
        "conditions": conditions,
        "target": {
            "short_edge": settings.short_edge,
            "aspect_ratio": "auto" if req.ratio == "adaptive" else req.ratio,
            "duration_seconds": req.duration,
        },
        "num_inference_steps": settings.num_inference_steps,
        "seed": req.seed,
        **_cache_dit_fields(),
    }


def _cache_dit_fields() -> dict:
    if not settings.cache_dit_enabled:
        return {}
    return {
        "cache_dit_params": {
            "residual_diff_threshold": settings.cache_dit_threshold,
            "max_continuous_cached_steps": settings.cache_dit_max_cached_steps,
            "max_warmup_steps": settings.cache_dit_warmup_steps,
        }
    }


def _json_or_error(resp: httpx.Response, service: str) -> dict:
    if resp.is_error:
        raise PipelineError(f"{service} returned HTTP {resp.status_code}: {resp.text[:1000]}")
    try:
        return resp.json()
    except ValueError as exc:
        raise PipelineError(f"{service} returned non-JSON response: {resp.text[:1000]}") from exc


async def _poll_get(client: httpx.AsyncClient, url: str, service: str, **kwargs) -> dict:
    """GET a status URL, retrying transient network errors so one dropped poll doesn't fail the job."""
    for attempt in range(POLL_RETRIES):
        try:
            return _json_or_error(await client.get(url, **kwargs), service)
        except httpx.TransportError as exc:
            if attempt == POLL_RETRIES - 1:
                raise PipelineError(f"{service} unreachable after {POLL_RETRIES} attempts: {exc!r}") from exc
            await asyncio.sleep(settings.poll_interval_s)


async def run_context_ir(client: httpx.AsyncClient, req: GenerationRequest) -> str:
    """Submit the request to hosted H3-Context-IR and return the expanded prompt."""
    if not settings.minimax_api_token:
        raise PipelineError("MINIMAX_API_TOKEN is not configured")
    base = settings.minimax_api_base.rstrip("/")
    headers = {"Authorization": f"Bearer {settings.minimax_api_token}"}

    resp = await client.post(f"{base}/v2/h3_context_ir", json=build_context_ir_payload(req), headers=headers)
    submitted = _json_or_error(resp, "Context-IR")
    task_id = submitted.get("task_id")
    if not task_id:
        raise PipelineError(f"Context-IR did not return a task_id: {submitted}")

    deadline = time.monotonic() + settings.context_ir_timeout_s
    while True:
        polled = await _poll_get(client, f"{base}/v2/query/video_generation/{task_id}", "Context-IR query", headers=headers)
        task = polled.get("task") or {}
        status = task.get("status")
        if status == "succeeded":
            prompt = (task.get("content") or {}).get("prompt")
            if not prompt:
                raise PipelineError(f"Context-IR task {task_id} succeeded without a prompt: {task}")
            return prompt
        if status in ("failed", "cancelled"):
            raise PipelineError(f"Context-IR task {task_id} {status}: {task}")
        if time.monotonic() > deadline:
            raise PipelineError(f"Context-IR task {task_id} timed out (last status: {status})")
        await asyncio.sleep(settings.poll_interval_s)


async def run_h3_base(client: httpx.AsyncClient, mode: Mode, payload: dict, out_path: Path) -> dict:
    """Generate on the self-hosted SGLang H3-Base, download the MP4 to out_path, return SGLang's video object."""
    base = (settings.sglang_ref2va_url if mode is Mode.REF2VA else settings.sglang_fl2va_url).rstrip("/")

    created = _json_or_error(await client.post(f"{base}/v1/videos", json=payload), "H3-Base")
    video_id = created.get("id")
    if not video_id:
        raise PipelineError(f"H3-Base did not return a video id: {created}")

    deadline = time.monotonic() + settings.h3_base_timeout_s
    while True:
        info = await _poll_get(client, f"{base}/v1/videos/{video_id}", "H3-Base query")
        status = info.get("status")
        if status in ("completed", "succeeded"):
            break
        # SGLang reports queued -> completed | failed (or deleted); there is no in_progress state.
        if status in ("failed", "deleted", "error", "cancelled"):
            raise PipelineError(f"H3-Base video {video_id} {status}: {info.get('error') or info}")
        if time.monotonic() > deadline:
            raise PipelineError(f"H3-Base video {video_id} timed out (last status: {status})")
        await asyncio.sleep(settings.poll_interval_s)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    async with client.stream("GET", f"{base}/v1/videos/{video_id}/content") as resp:
        if resp.is_error:
            await resp.aread()
            raise PipelineError(f"H3-Base download returned HTTP {resp.status_code}: {resp.text[:1000]}")
        with out_path.open("wb") as f:
            async for chunk in resp.aiter_bytes():
                f.write(chunk)
    return info


def h3_base_stats(info: dict, num_inference_steps: int) -> dict:
    """Generation details from SGLang's completed video object.

    SGLang's API only reports the total render time (text encoding + denoising + VAE decode),
    so avg_step_s is an upper bound on the true per-step denoising time.
    """
    inference_s = info.get("inference_time_s")
    created, completed = info.get("created_at"), info.get("completed_at")
    queue_wait_s = None
    if inference_s is not None and created is not None and completed is not None:
        # created_at/completed_at are whole seconds, so this is accurate to about 1 s.
        queue_wait_s = round(max(0.0, completed - created - inference_s), 1)
    return {
        "video_id": info.get("id"),
        "size": info.get("size"),
        "video_seconds": float(info["seconds"]) if info.get("seconds") else None,
        "num_inference_steps": num_inference_steps,
        "inference_time_s": round(inference_s, 2) if inference_s is not None else None,
        "avg_step_s": round(inference_s / num_inference_steps, 2) if inference_s and num_inference_steps else None,
        "queue_wait_s": queue_wait_s,
        "peak_memory_mb": info.get("peak_memory_mb"),
    }


async def upload_result(local_path: Path, s3_folder: str, job_id: str) -> str:
    """Upload the MP4 to s3_folder/<job_id>/<job_id>.mp4 and return its s3:// URI."""
    s3_uri = await asyncio.to_thread(
        s3_utils.upload_to_s3, str(local_path), s3_folder, custom_filename=f"{job_id}.mp4", job_id=job_id
    )
    if not s3_uri:
        raise PipelineError(f"S3 upload to {s3_folder} failed (see server logs)")
    return s3_uri


async def presign(s3_uri: str) -> str:
    url = await asyncio.to_thread(s3_utils.generate_presigned_url, s3_uri, settings.s3_presign_expires_s)
    if not url or url == s3_uri:
        raise PipelineError(f"could not presign {s3_uri} (see server logs)")
    return url


async def generate(
    req: GenerationRequest,
    job_id: str,
    s3_folder: str,
    on_stage: Callable[[str], None] = lambda stage: None,
    on_prompt: Callable[[str], None] = lambda prompt: None,
    on_h3_base: Callable[[dict], None] = lambda stats: None,
) -> str:
    """Run Context-IR -> H3-Base -> S3 upload and return the result's s3:// URI."""
    req.validate()
    out_path = settings.output_dir / f"{job_id}.mp4"
    timeout = httpx.Timeout(120.0, connect=15.0)
    # SGLang's uvicorn closes idle connections after 5 s (its default keep-alive), the same as
    # POLL_INTERVAL_S; expire ours first so a poll never reuses a socket the server is closing.
    limits = httpx.Limits(keepalive_expiry=2.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            on_stage("context_ir")
            expanded_prompt = await run_context_ir(client, req)
            on_prompt(expanded_prompt)
            on_stage("h3_base")
            payload = build_h3_base_payload(req, expanded_prompt)
            info = await run_h3_base(client, req.mode, payload, out_path)
            on_h3_base({**h3_base_stats(info, payload["num_inference_steps"]), "cache_dit": payload.get("cache_dit_params")})
        on_stage("upload")
        return await upload_result(out_path, s3_folder, job_id)
    finally:
        out_path.unlink(missing_ok=True)
