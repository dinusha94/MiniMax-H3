# H3 Generation Server

An internal FastAPI server that combines the hosted **H3-Context-IR** API, a **self-hosted H3-Base** (SGLang) and **S3**. For each job, the server:

1. sends the prompt and media URLs to MiniMax `POST /v2/h3_context_ir` and polls until the task has succeeded,
2. passes the expanded prompt and the same media URLs to SGLang `POST /v1/videos`, then polls until the video is ready,
3. downloads the 768p MP4 and uploads it to `<S3_OUTPUT_FOLDER>/<job_id>/<job_id>.mp4` with `s3_utils.upload_to_s3`.

The payloads match `scripts/readme/full-2k-*-h3-context-ir.sh` and `full-2k-*-h3-base.sh`. There is no authentication, so keep the server on an internal network.

## Setup

Start H3-Base (see the main README, "Sglang Deployment"). `fl2va` serves both `t2va` and `fl2va`. `ref2va` needs its own instance.

```bash
cd server
pip install -r requirements.txt
cp .env.example .env        # MINIMAX_API_TOKEN, SGLang URLs, S3_OUTPUT_FOLDER, AWS credentials
uvicorn app:app --host 0.0.0.0 --port 8000 --env-file .env
```

Interactive docs are at `http://localhost:8000/docs`. The server needs AWS credentials that allow `s3:PutObject` and `s3:GetObject` on the output bucket.

### Input URLs

Media are presigned S3 URLs, and the server passes them on unchanged:
- **Context-IR** fetches them from MiniMax's servers, so they must be reachable from the public internet. Presigned S3 URLs are.
- **SGLang** fetches them from the GPU machine, so that machine needs outbound internet access to S3.
- **Expiry:** URLs must stay valid until SGLang has loaded them. That takes a few minutes after submission, but longer if SGLang has a queue. One-hour presigned URLs are fine at low volume.

## API

### 1. Submit: `POST /v1/jobs`

```bash
curl -X POST localhost:8000/v1/jobs -H 'Content-Type: application/json' -d '{
  "mode": "ref2va",
  "prompt": "Character speaks: Follow the wind, live free. Voice timbre follows reference audio 1.",
  "duration": 5,
  "reference_video_urls": ["https://bucket.s3.amazonaws.com/in/person.mp4?X-Amz-..."],
  "reference_audio_urls": ["https://bucket.s3.amazonaws.com/in/voice.mp3?X-Amz-..."]
}'
```
```json
{"job_id": "a3e4c6e9...", "mode": "ref2va", "status": "queued", "stage": null, "error": null, ...}
```

| Field | Modes | Notes |
|---|---|---|
| `mode` | all | `t2va` (text only), `fl2va` (first/last frame), `ref2va` (references) |
| `prompt` | all | Raw instruction. Context-IR expands it |
| `duration` | all | 4–15 seconds (default 5) |
| `ratio` | all | `adaptive`, `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`. Default is `16:9` for t2va (adaptive is not allowed there) and `adaptive` for the other modes |
| `seed` | all | default 0 |
| `first_frame_url`, `last_frame_url` | fl2va | image; one or both |
| `reference_image_urls` | ref2va | ≤ 9 |
| `reference_video_urls` | ref2va | ≤ 3, 2–15 s each |
| `reference_audio_urls` | ref2va | ≤ 3, 2–15 s each |
| `output_s3_folder` | all | optional; overrides `S3_OUTPUT_FOLDER` |

Ref2VA accepts at most 12 files in total. List order sets the numbering in the prompt: the first entry in `reference_audio_urls` is "reference audio 1", and so on.

Invalid input returns `400` (bad field values give `422`) and no job is created.

### 2. Poll: `GET /v1/jobs/{job_id}`

```json
{"job_id": "a3e4c6e9...", "status": "running", "stage": "h3_base",
 "expanded_prompt": "integrated_multimodal_description: ...", "error": null, ...}
```

`status` is `queued`, `running`, `succeeded` or `failed`. `stage` is `context_ir`, `h3_base` or `upload`. When a job fails, `error` holds the message from Context-IR, SGLang or S3. A video takes minutes, so polling every 10–30 s is plenty.

### 3. Result: `GET /v1/jobs/{job_id}/result`

```json
{
  "job_id": "a3e4c6e9...",
  "status": "succeeded",
  "video_s3_uri": "s3://your-bucket/h3-outputs/a3e4c6e9.../a3e4c6e9....mp4",
  "video_url": "https://your-bucket.s3.amazonaws.com/...?X-Amz-...",
  "video_url_expires_in": 3600,
  "expanded_prompt": "integrated_multimodal_description: ..."
}
```

Each call presigns `video_url` again, so if the link has expired, just call again. The endpoint returns `409` until the job succeeds.

## Limitations

- Jobs are stored in memory. A restart loses jobs in progress (the platform sees `404` and must resubmit), and jobs aren't shared across workers, so run a single uvicorn worker. Finished videos stay safe in S3 either way.
- Output is 768p. 2K needs the hosted H3-Regenerate-2K API, which this server does not call.
