"""Local transcription, model lifecycle and telemetry APIs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, model_validator

from .file_api import router as file_router
from .lifecycle import ModelManager, ModelsBusy, ModelsNotReady
from .metrics import SessionMetrics, SystemMonitor, utc_now
from .models import ModelProfile, Models
from .stream import AudioStream, StreamConfig
from .text import SentenceAssembler, agreed_prefix


log = logging.getLogger(__name__)
AUDIO_FORMAT = {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}


class ProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_language: Literal["en", "ru"]
    target_language: Literal["ru"] | None = None

    @model_validator(mode="after")
    def validate_profile(self):
        self.profile()
        return self

    def profile(self):
        return ModelProfile(self.source_language, self.target_language)


class SessionOverloaded(RuntimeError):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.manager = ModelManager(Models)
    app.state.monitor = SystemMonitor()
    app.state.monitor.start()
    log.info("Server ready; models will load on request")
    try:
        yield
    finally:
        await app.state.manager.shutdown()
        await asyncio.to_thread(app.state.monitor.close)


app = FastAPI(lifespan=lifespan, title="Auditor", version="0.2.0")
app.include_router(file_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/v1/system")
async def system_status():
    manager = app.state.manager
    return {
        "api_version": 2,
        "supported_languages": ["en", "ru"],
        "supported_translations": [{"source_language": "en", "target_language": "ru"}],
        "runtime": manager.snapshot(), "system": app.state.monitor.snapshot(),
        "session": manager.session.metrics_snapshot() if manager.session else None,
        "last_session": manager.last_session,
        "file_job": dict(manager.file_job) if manager.file_job else None,
        "last_file_job": manager.last_file_job,
    }


@app.post("/v1/models/load", status_code=202)
async def load_models(request: ProfileRequest):
    try:
        app.state.manager.begin_load(request.profile())
    except ModelsBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return app.state.manager.snapshot()


@app.post("/v1/models/unload", status_code=202)
async def unload_models():
    try:
        app.state.manager.begin_unload()
    except ModelsBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return app.state.manager.snapshot()


def parse_start(message: str, *, legacy=False) -> ModelProfile | None:
    try:
        data = json.loads(message)
        if not isinstance(data, dict) or data.get("type") != "start" or data.get("audio") != AUDIO_FORMAT:
            return None
        profile = ProfileRequest.model_validate({
            "source_language": data.get("source_language"), "target_language": data.get("target_language"),
        }).profile()
        if legacy and profile != ModelProfile("en", "ru"):
            return None
        return profile
    except (ValueError, TypeError):
        return None


def valid_start(message: str) -> bool:
    return parse_start(message, legacy=True) is not None


class LiveSession:
    def __init__(self, websocket: WebSocket, models: Models, profile: ModelProfile, *, legacy=False) -> None:
        self.websocket, self.models, self.profile, self.legacy = websocket, models, profile, legacy
        self.session_id = str(uuid4())
        self.metrics = SessionMetrics()
        self.audio = AudioStream(StreamConfig(
            speech_rms=float(os.getenv("AUDITOR_SPEECH_RMS", "0.012")),
            silence_seconds=float(os.getenv("AUDITOR_SILENCE_SECONDS", "0.7")),
            max_phrase_seconds=float(os.getenv("AUDITOR_MAX_PHRASE_SECONDS", "6")),
        ))
        self.wake = asyncio.Event()
        self.send_lock = asyncio.Lock()
        self.translation_queue = asyncio.Queue(maxsize=12) if profile.target_language else None
        self.assembler = SentenceAssembler()
        self.ended = False
        self.segment_number = 0
        self.last_partial = ""
        self.last_hypothesis = ""
        self.last_hypothesis_phrase = 0
        self.last_partial_revision = -1
        self.next_partial_at = 0.0
        self.last_metrics_at = asyncio.get_running_loop().time()

    def metrics_snapshot(self):
        return {
            "session_id": self.session_id, **asdict(self.profile), **self.metrics.snapshot(),
            "audio_received_seconds": self.audio.total_samples / 16000,
            "pending_audio_seconds": self.audio.pending_seconds,
            "oldest_pending_seconds": self.audio.oldest_pending_age,
            "translation_queue_size": self.translation_queue.qsize() if self.translation_queue else 0,
        }

    async def send(self, event: dict) -> None:
        async with self.send_lock:
            await self.websocket.send_text(json.dumps(event, ensure_ascii=False))

    async def transcript(self, kind, text, **extra):
        event = {"type": f"{kind}_en" if self.legacy else kind, "text": text, **extra}
        if not self.legacy:
            event["language"] = self.profile.source_language
        await self.send(event)

    async def publish_stable(self, text: str) -> None:
        if self.translation_queue is not None and self.translation_queue.full():
            raise SessionOverloaded("translation backlog exceeded")
        self.segment_number += 1
        segment_id = str(self.segment_number)
        await self.transcript("stable", text, segment_id=segment_id)
        self.metrics.stable_segments += 1
        if self.translation_queue is not None:
            self.translation_queue.put_nowait((segment_id, text, asyncio.get_running_loop().time()))

    async def recognize(self, samples, *, final):
        started = asyncio.get_running_loop().time()
        result = await asyncio.to_thread(self.models.transcribe, samples)
        self.metrics.asr.add(asyncio.get_running_loop().time() - started)
        if final:
            self.metrics.asr_final_tokens += result.tokens
        else:
            self.metrics.asr_partial_tokens += result.tokens
        return result.text

    async def run_asr(self) -> None:
        while True:
            final = self.audio.pop_final_item()
            if final is not None:
                samples, paused, queued_at = final
                text = await self.recognize(samples, final=True)
                self.metrics.finalized_audio_to_asr.add(asyncio.get_running_loop().time() - queued_at)
                if text:
                    for sentence in self.assembler.add(text):
                        await self.publish_stable(sentence)
                if paused or (not self.audio.active and not self.audio.has_final):
                    for sentence in self.assembler.flush():
                        await self.publish_stable(sentence)
                if self.last_partial:
                    await self.transcript("partial", "")
                    self.last_partial = ""
                self.last_hypothesis = ""
                continue

            if self.ended:
                for sentence in self.assembler.flush():
                    await self.publish_stable(sentence)
                if self.translation_queue is not None:
                    await self.translation_queue.put(None)
                return

            now = asyncio.get_running_loop().time()
            if now - self.last_metrics_at >= 60:
                log.info("Session metrics: %s", self.metrics_snapshot())
                self.last_metrics_at = now
            candidate = self.audio.partial()
            if (candidate and not self.audio.has_final and
                    candidate[1] != self.last_partial_revision and now >= self.next_partial_at):
                phrase_id, revision, samples = candidate
                self.last_partial_revision = revision
                text = await self.recognize(samples, final=False)
                elapsed = asyncio.get_running_loop().time() - now
                self.next_partial_at = asyncio.get_running_loop().time() + max(0.4, elapsed)
                current = self.audio.partial()
                if current is not None and current[0] == phrase_id:
                    previous = self.last_hypothesis if self.last_hypothesis_phrase == phrase_id else ""
                    agreed = agreed_prefix(previous, text)
                    self.last_hypothesis = text
                    self.last_hypothesis_phrase = phrase_id
                    if agreed and agreed != self.last_partial:
                        await self.transcript("partial", agreed)
                        self.last_partial = agreed

            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=0.4)
            except TimeoutError:
                pass

    async def run_translation(self) -> None:
        while True:
            item = await self.translation_queue.get()
            try:
                if item is None:
                    return
                segment_id, text, stable_at = item
                started = asyncio.get_running_loop().time()
                try:
                    result = await asyncio.to_thread(self.models.translate, text)
                except Exception:
                    self.metrics.translation_errors += 1
                    log.exception("Translation failed for segment %s", segment_id)
                    if not self.legacy:
                        await self.send({"type": "error", "code": "translation_failed", "segment_id": segment_id,
                                         "message": "Translation failed; transcription continues"})
                    continue
                self.metrics.translation.add(asyncio.get_running_loop().time() - started)
                self.metrics.translation_tokens += result.tokens
                if result.text:
                    event = {"type": "translation_ru" if self.legacy else "translation",
                             "segment_id": segment_id, "text": result.text}
                    if not self.legacy:
                        event["language"] = self.profile.target_language
                    await self.send(event)
                    self.metrics.stable_to_translation.add(asyncio.get_running_loop().time() - stable_at)
            finally:
                self.translation_queue.task_done()


async def reject(websocket, legacy, code, message, close_code):
    with suppress(WebSocketDisconnect, RuntimeError, OSError):
        if not legacy:
            await websocket.send_json({"type": "error", "code": code, "message": message})
        await websocket.close(code=close_code, reason=code)


@app.websocket("/v1/live")
async def live_v1(websocket: WebSocket):
    await live(websocket, legacy=True)


@app.websocket("/v2/live")
async def live_v2(websocket: WebSocket):
    await live(websocket)


async def live(websocket: WebSocket, *, legacy=False) -> None:
    await websocket.accept()
    manager = websocket.app.state.manager
    session = None
    workers = []
    receiver = None
    acquired = False
    legacy_operation = None
    try:
        try:
            first = await asyncio.wait_for(websocket.receive(), timeout=10)
        except TimeoutError:
            await reject(websocket, legacy, "start_timeout", "Start message required within 10 seconds", 1002)
            return
        if first["type"] == "websocket.disconnect":
            return
        profile = parse_start(first.get("text") or "", legacy=legacy)
        if profile is None:
            await reject(websocket, legacy, "invalid_start", "Expected start with PCM 16 kHz mono and a supported language profile", 1003)
            return
        if legacy:
            legacy_operation = manager.begin_load(profile)
            if legacy_operation is not None:
                await asyncio.shield(legacy_operation)
        models = manager.acquire(profile)
        acquired = True
        session = LiveSession(websocket, models, profile, legacy=legacy)
        manager.session = session
        if not legacy:
            await session.send({"type": "ready", "session_id": session.session_id, **asdict(profile)})
        workers.append(asyncio.create_task(session.run_asr()))
        if profile.target_language is not None:
            workers.append(asyncio.create_task(session.run_translation()))
        while True:
            receiver = asyncio.create_task(websocket.receive())
            done, _ = await asyncio.wait([receiver, *workers], return_when=asyncio.FIRST_COMPLETED)
            for worker in workers:
                if worker in done:
                    worker.result()
                    raise RuntimeError("Inference worker stopped unexpectedly")
            message = receiver.result()
            receiver = None
            if message["type"] == "websocket.disconnect":
                return
            chunk = message.get("bytes")
            if chunk is not None:
                if len(chunk) > 256 * 1024:
                    await reject(websocket, legacy, "audio_too_large", "Audio message exceeds 256 KiB", 1009)
                    return
                session.audio.feed(chunk)
                if session.audio.pending_seconds > 12:
                    raise SessionOverloaded("ASR backlog exceeded 12 seconds")
                session.wake.set()
                continue
            try:
                event = json.loads(message.get("text") or "")
            except ValueError:
                event = None
            if isinstance(event, dict) and event.get("type") == "end":
                session.audio.finish()
                session.ended = True
                session.wake.set()
                timeout = 1.85 if legacy else float(os.getenv("AUDITOR_END_TIMEOUT", "10"))
                pending = set(workers)
                deadline = asyncio.get_running_loop().time() + timeout
                receiver = asyncio.create_task(websocket.receive())
                while pending:
                    remaining = max(0, deadline - asyncio.get_running_loop().time())
                    finished, _ = await asyncio.wait([receiver, *pending], timeout=remaining,
                                                     return_when=asyncio.FIRST_COMPLETED)
                    if not finished:
                        break
                    if receiver in finished:
                        if receiver.result()["type"] != "websocket.disconnect":
                            await reject(websocket, legacy, "invalid_control", "No messages allowed after end", 1003)
                        return
                    for worker in finished:
                        worker.result()
                    pending.difference_update(finished)
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
                receiver = None
                if not legacy:
                    await session.send({"type": "ended", "complete": not pending and not session.metrics.translation_errors})
                if pending:
                    log.warning("Final inference exceeded end grace period")
                await websocket.close()
                return
            await reject(websocket, legacy, "invalid_control", "Expected end or binary PCM", 1003)
            return
    except ModelsBusy as exc:
        await reject(websocket, legacy, "busy", str(exc), 1013)
    except ModelsNotReady as exc:
        await reject(websocket, legacy, "models_not_ready", str(exc), 1013)
    except WebSocketDisconnect:
        pass
    except SessionOverloaded as exc:
        await reject(websocket, legacy, "overloaded", str(exc), 1013)
    except Exception:
        log.exception("Live session failed")
        await reject(websocket, legacy, "server_error", "Transcription failed", 1011)
    finally:
        if receiver is not None:
            receiver.cancel()
        for worker in workers:
            worker.cancel()
        if session is not None:
            session.metrics.ended_at = utc_now()
        if acquired:
            # Kill model processes before waiting for cancelled inference threads.
            await asyncio.shield(manager.release())
        elif legacy_operation is not None:
            await asyncio.shield(manager.abandon_legacy_load(legacy_operation))
        await asyncio.gather(*workers, *([receiver] if receiver else []), return_exceptions=True)


def main() -> None:
    import uvicorn
    uvicorn.run(
        "auditor_server.main:app", host=os.getenv("AUDITOR_HOST", "0.0.0.0"),
        port=int(os.getenv("AUDITOR_PORT", "8000")), workers=1,
        log_level=os.getenv("AUDITOR_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
