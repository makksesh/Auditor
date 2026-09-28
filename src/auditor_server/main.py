"""WebSocket server for the AudioSharing macOS client."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .models import Models
from .stream import AudioStream, StreamConfig
from .text import SentenceAssembler, agreed_prefix


log = logging.getLogger(__name__)


class SessionOverloaded(RuntimeError):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = await asyncio.to_thread(Models)
    log.info("ASR and translation models ready")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready"}


def valid_start(message: str) -> bool:
    try:
        data = json.loads(message)
    except (ValueError, TypeError):
        return False
    return (
        isinstance(data, dict)
        and data.get("type") == "start"
        and data.get("audio") == {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}
        and data.get("source_language") == "en"
        and data.get("target_language") == "ru"
    )


class LiveSession:
    def __init__(self, websocket: WebSocket, models: Models) -> None:
        self.websocket = websocket
        self.models = models
        self.audio = AudioStream(StreamConfig(
            speech_rms=float(os.getenv("AUDITOR_SPEECH_RMS", "0.012")),
            silence_seconds=float(os.getenv("AUDITOR_SILENCE_SECONDS", "0.7")),
            max_phrase_seconds=float(os.getenv("AUDITOR_MAX_PHRASE_SECONDS", "6")),
        ))
        self.wake = asyncio.Event()
        self.send_lock = asyncio.Lock()
        self.translation_queue: asyncio.Queue[tuple[str, str, float] | None] = asyncio.Queue(maxsize=12)
        self.assembler = SentenceAssembler()
        self.ended = False
        self.segment_number = 0
        self.last_partial = ""
        self.last_hypothesis = ""
        self.last_hypothesis_phrase = 0
        self.last_partial_revision = -1
        self.next_partial_at = 0.0
        self.last_metrics_at = asyncio.get_running_loop().time()

    async def send(self, event: dict[str, str]) -> None:
        async with self.send_lock:
            await self.websocket.send_text(json.dumps(event, ensure_ascii=False))

    async def publish_stable(self, text: str) -> None:
        if self.translation_queue.full():
            raise SessionOverloaded("translation backlog exceeded")
        self.segment_number += 1
        segment_id = str(self.segment_number)
        await self.send({"type": "stable_en", "segment_id": segment_id, "text": text})
        self.translation_queue.put_nowait((segment_id, text, asyncio.get_running_loop().time()))

    async def run_asr(self) -> None:
        while True:
            final = self.audio.pop_final_item()
            if final is not None:
                samples, paused, queued_at = final
                started = asyncio.get_running_loop().time()
                text = await asyncio.to_thread(self.models.transcribe, samples)
                elapsed = asyncio.get_running_loop().time() - started
                log.debug("ASR %.2fs audio in %.2fs; lag %.2fs", len(samples) / 16000, elapsed, asyncio.get_running_loop().time() - queued_at)
                if text:
                    for sentence in self.assembler.add(text):
                        await self.publish_stable(sentence)
                if paused or (not self.audio.active and not self.audio.has_final):
                    for sentence in self.assembler.flush():
                        await self.publish_stable(sentence)
                if self.last_partial:
                    await self.send({"type": "partial_en", "text": ""})
                    self.last_partial = ""
                self.last_hypothesis = ""
                continue

            if self.ended:
                for sentence in self.assembler.flush():
                    await self.publish_stable(sentence)
                await self.translation_queue.put(None)
                return

            now = asyncio.get_running_loop().time()
            if now - self.last_metrics_at >= 60:
                log.info(
                    "session audio=%.0fs pending_audio=%.1fs oldest_pending=%.1fs translation_queue=%d",
                    self.audio.total_samples / 16000,
                    self.audio.pending_seconds,
                    self.audio.oldest_pending_age,
                    self.translation_queue.qsize(),
                )
                self.last_metrics_at = now
            candidate = self.audio.partial()
            if (candidate and not self.audio.has_final and
                    candidate[1] != self.last_partial_revision and now >= self.next_partial_at):
                phrase_id, revision, samples = candidate
                self.last_partial_revision = revision
                started = now
                text = await asyncio.to_thread(self.models.transcribe, samples)
                elapsed = asyncio.get_running_loop().time() - started
                self.next_partial_at = asyncio.get_running_loop().time() + max(0.4, elapsed)
                # A phrase finalized during inference must not reappear as a draft.
                current = self.audio.partial()
                if current is not None and current[0] == phrase_id:
                    previous = self.last_hypothesis if self.last_hypothesis_phrase == phrase_id else ""
                    agreed = agreed_prefix(previous, text)
                    self.last_hypothesis = text
                    self.last_hypothesis_phrase = phrase_id
                    if agreed and agreed != self.last_partial:
                        await self.send({"type": "partial_en", "text": agreed})
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
                    translation = await asyncio.to_thread(self.models.translate, text)
                    if translation:
                        await self.send({"type": "translation_ru", "segment_id": segment_id, "text": translation})
                except Exception:
                    log.exception("Translation failed for segment %s", segment_id)
                else:
                    log.debug("Translation segment %s in %.2fs; lag %.2fs", segment_id, asyncio.get_running_loop().time() - started, asyncio.get_running_loop().time() - stable_at)
            finally:
                self.translation_queue.task_done()


@app.websocket("/v1/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        first = await asyncio.wait_for(websocket.receive(), timeout=10)
        if first.get("text") is None or not valid_start(first["text"]):
            await websocket.close(code=1003, reason="Expected start with 16 kHz mono pcm_s16le, en to ru")
            return
    except (TimeoutError, WebSocketDisconnect):
        await websocket.close(code=1002, reason="Start message required")
        return

    session = LiveSession(websocket, websocket.app.state.models)
    asr_worker = asyncio.create_task(session.run_asr())
    translation_worker = asyncio.create_task(session.run_translation())
    try:
        while True:
            message = await websocket.receive()
            for worker in (asr_worker, translation_worker):
                if worker.done() and not worker.cancelled() and worker.exception() is not None:
                    raise worker.exception()
            if message["type"] == "websocket.disconnect":
                return
            chunk = message.get("bytes")
            if chunk is not None:
                if len(chunk) > 256 * 1024:
                    await websocket.close(code=1009, reason="Audio message too large")
                    return
                session.audio.feed(chunk)
                if session.audio.pending_seconds > 12:
                    raise SessionOverloaded("ASR backlog exceeded 12 seconds")
                session.wake.set()
                continue
            try:
                event = json.loads(message.get("text") or "")
            except ValueError:
                await websocket.close(code=1003, reason="Invalid JSON control message")
                return
            if isinstance(event, dict) and event.get("type") == "end":
                session.audio.finish()
                session.ended = True
                session.wake.set()
                try:
                    await asyncio.wait_for(asyncio.gather(asr_worker, translation_worker), timeout=1.85)
                except TimeoutError:
                    log.warning("Final inference exceeded client end grace period")
                await websocket.close()
                return
            await websocket.close(code=1003, reason="Unexpected control message")
            return
    except WebSocketDisconnect:
        pass
    except SessionOverloaded as exc:
        log.warning("Live session overloaded: %s", exc)
        with suppress(Exception):
            await websocket.close(code=1013, reason=str(exc))
    except Exception:
        log.exception("Live session failed")
        with suppress(Exception):
            await websocket.close(code=1011, reason="Server error")
    finally:
        for worker in (asr_worker, translation_worker):
            if not worker.done():
                worker.cancel()
        for worker in (asr_worker, translation_worker):
            with suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                await worker


def main() -> None:
    import uvicorn

    uvicorn.run(
        "auditor_server.main:app",
        host=os.getenv("AUDITOR_HOST", "0.0.0.0"),
        port=int(os.getenv("AUDITOR_PORT", "8000")),
        workers=1,
        log_level=os.getenv("AUDITOR_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
