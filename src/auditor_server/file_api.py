"""Whole-file transcription, independent from the live WebSocket protocol."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from time import monotonic
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from starlette.requests import ClientDisconnect

from .lifecycle import ModelsBusy, ModelsNotReady
from .metrics import utc_now
from .models import ModelProfile

log = logging.getLogger(__name__)
router = APIRouter()


def probe_audio(path: str) -> float | None:
    """Reject non-audio files before loading a model; duration may be unknown."""
    import av

    try:
        with av.open(path) as container:
            if not any(stream.type == "audio" for stream in container.streams):
                raise ValueError("No audio stream found")
            return float(container.duration / av.time_base) if container.duration is not None else None
    except av.FFmpegError as exc:
        raise ValueError("Unsupported or invalid audio file") from exc


async def wait_for_result_or_disconnect(request: Request, task: asyncio.Task):
    """Stop inference when the uploading HTTP client closes its connection."""
    async def watch():
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    watcher = asyncio.create_task(watch())
    try:
        done, _ = await asyncio.wait((task, watcher), return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return task.result()
        raise ClientDisconnect()
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


async def cleanup_file_job(manager, job, loading, inference, path):
    """Finish even when the HTTP request task is cancelled by a disconnect."""
    try:
        if loading is not None:
            await asyncio.shield(loading)
        if inference is not None and not inference.done():
            inference.cancel()
        job["finished_at"] = utc_now()
        await asyncio.shield(manager.release())
        if inference is not None:
            await asyncio.gather(inference, return_exceptions=True)
    finally:
        if path is not None:
            with suppress(FileNotFoundError):
                Path(path).unlink()


@router.post("/v1/transcriptions")
async def transcribe_file(request: Request, source_language: Literal["en", "ru"]):
    """Upload raw audio bytes and return the full transcript after inference."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/octet-stream" and not content_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail="Use an audio/* or application/octet-stream binary body")

    max_bytes = int(os.getenv("AUDITOR_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            length = int(content_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
        if length > max_bytes:
            raise HTTPException(status_code=413, detail="Audio file is too large")

    manager = request.app.state.manager
    try:
        manager.reserve_file()
    except ModelsBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    profile = ModelProfile(source_language)
    job = {"source_language": source_language, "status": "uploading", "started_at": utc_now(),
           "finished_at": None, "bytes": 0, "duration_seconds": None, "tokens": None,
           "processing_ms": None, "error": None}
    manager.file_job = job
    path = None
    loading = inference = None
    try:
        fd, path = tempfile.mkstemp(prefix="auditor-upload-", suffix=".audio")
        with os.fdopen(fd, "wb") as target:
            async for chunk in request.stream():
                if job["bytes"] + len(chunk) > max_bytes:
                    raise HTTPException(status_code=413, detail="Audio file is too large")
                target.write(chunk)
                job["bytes"] += len(chunk)
        if job["bytes"] == 0:
            raise HTTPException(status_code=422, detail="Audio file is empty")

        job["status"] = "validating"
        try:
            job["duration_seconds"] = await asyncio.to_thread(probe_audio, path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        job["status"] = "loading"
        loading = manager.begin_reserved_file_load(profile)
        if loading is not None:
            await asyncio.shield(loading)
        try:
            models = manager.prepared_file_models(profile)
        except ModelsNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        job["status"] = "transcribing"
        started = monotonic()
        inference = asyncio.create_task(asyncio.to_thread(models.transcribe_file, path))
        try:
            result = await wait_for_result_or_disconnect(request, inference)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="File transcription timed out") from exc
        job["processing_ms"] = (monotonic() - started) * 1000
        job["tokens"] = result.tokens
        job["status"] = "complete"
        return {"source_language": source_language, "text": result.text, "tokens": result.tokens,
                "duration_seconds": job["duration_seconds"], "processing_ms": job["processing_ms"]}
    except HTTPException as exc:
        job["status"] = "error"
        job["error"] = str(exc.detail)
        raise
    except ClientDisconnect:
        job["status"] = "disconnected"
        job["error"] = "Client disconnected"
        raise
    except asyncio.CancelledError:
        job["status"] = "disconnected"
        job["error"] = "Client disconnected or request cancelled"
        raise
    except Exception:
        job["status"] = "error"
        job["error"] = "File transcription failed"
        log.exception("File transcription failed")
        raise HTTPException(status_code=500, detail=job["error"]) from None
    finally:
        # Shield the whole cleanup, not just unload: cancellation can arrive
        # while a model is still loading or a worker thread is being reaped.
        await asyncio.shield(asyncio.create_task(
            cleanup_file_job(manager, job, loading, inference, path)
        ))
