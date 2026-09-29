"""Server settings, read from environment variables (or a .env file via uvicorn --env-file)."""

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # MiniMax Open Platform (hosted H3-Context-IR).
    minimax_api_base: str = _env("MINIMAX_API_BASE", "https://api.minimax.io")
    minimax_api_token: str | None = _env("MINIMAX_API_TOKEN")

    # Self-hosted H3-Base SGLang deployments. t2va and fl2va use the FL2VA
    # checkpoint; ref2va uses the Ref2VA checkpoint.
    sglang_fl2va_url: str = _env("SGLANG_FL2VA_URL", "http://localhost:30010")
    sglang_ref2va_url: str = _env("SGLANG_REF2VA_URL", "http://localhost:30012")

    # Generated videos are written here, uploaded to S3, then deleted.
    output_dir: Path = Path(_env("OUTPUT_DIR", "./data/outputs")).resolve()

    # Default S3 folder for results (s3://bucket/prefix/); a job can override it.
    s3_output_folder: str | None = _env("S3_OUTPUT_FOLDER")
    # Lifetime of the presigned result URL returned by GET /v1/jobs/{id}/result.
    s3_presign_expires_s: int = int(_env("S3_PRESIGN_EXPIRES_S", "3600"))

    short_edge: int = int(_env("H3_SHORT_EDGE", "768"))
    # Denoising steps per H3-Base render; SGLang defaults to 50 when omitted.
    num_inference_steps: int = int(_env("H3_NUM_INFERENCE_STEPS", "20"))
    # Cache-DiT: reuse DiT block outputs between similar steps. 0.12 / 2 measured 29-40% faster
    # at 20 steps with reviewed-equivalent output. Needs SGLANG_CACHE_DIT_ENABLED=true on SGLang.
    cache_dit_enabled: bool = _env("H3_CACHE_DIT_ENABLED", "false").lower() in ("1", "true", "yes")
    cache_dit_threshold: float = float(_env("H3_CACHE_DIT_THRESHOLD", "0.12"))
    cache_dit_max_cached_steps: int = int(_env("H3_CACHE_DIT_MAX_CACHED_STEPS", "2"))
    cache_dit_warmup_steps: int = int(_env("H3_CACHE_DIT_WARMUP_STEPS", "4"))
    poll_interval_s: float = float(_env("POLL_INTERVAL_S", "5"))
    context_ir_timeout_s: float = float(_env("CONTEXT_IR_TIMEOUT_S", "900"))
    h3_base_timeout_s: float = float(_env("H3_BASE_TIMEOUT_S", "3600"))


settings = Settings()
