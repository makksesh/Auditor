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


log = logging.getLogger(__name__)


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
            max_phrase_seconds=float(os.getenv("AUDITOR_MAX_PHRASE_SECONDS", "10")),
        ))
        self.wake = asyncio.Event()
        self.ended = False
        self.segment_number = 0
        self.last_partial = ""
        self.last_partial_revision = -1
        self.last_partial_at = 0.0

    async def send(self, event: dict[str, str]) -> None:
        await self.websocket.send_text(json.dumps(event, ensure_ascii=False))

    async def run_worker(self) -> None:
        while True:
            final = self.audio.pop_final()
            if final is not None:
                text = await asyncio.to_thread(self.models.transcribe, final)
                if text:
                    self.segment_number += 1
                    segment_id = str(self.segment_number)
                    await self.send({"type": "stable_en", "segment_id": segment_id, "text": text})
                    if self.last_partial:
                        await self.send({"type": "partial_en", "text": ""})
                        self.last_partial = ""
                    try:
                        translation = await asyncio.to_thread(self.models.translate, text)
                        if translation:
                            await self.send({"type": "translation_ru", "segment_id": segment_id, "text": translation})
                    except Exception:
                        log.exception("Translation failed for segment %s", segment_id)
                continue

            if self.ended:
                return

            now = asyncio.get_running_loop().time()
            candidate = self.audio.partial()
            if candidate and candidate[1] != self.last_partial_revision and now - self.last_partial_at >= 0.4:
                phrase_id, revision, samples = candidate
                self.last_partial_revision = revision
                self.last_partial_at = now
                text = await asyncio.to_thread(self.models.transcribe, samples)
                # A phrase finalized during inference must not reappear as a draft.
                current = self.audio.partial()
                if current is not None and current[0] == phrase_id and text != self.last_partial:
                    await self.send({"type": "partial_en", "text": text})
                    self.last_partial = text

            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=0.4)
            except TimeoutError:
                pass


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
    worker = asyncio.create_task(session.run_worker())
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            chunk = message.get("bytes")
            if chunk is not None:
                if len(chunk) > 256 * 1024:
                    await websocket.close(code=1009, reason="Audio message too large")
                    return
                session.audio.feed(chunk)
                if session.audio.final_count > 8:
                    await websocket.close(code=1013, reason="ASR backlog exceeded")
                    return
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
                    await asyncio.wait_for(worker, timeout=1.85)
                except TimeoutError:
                    log.warning("Final inference exceeded client end grace period")
                await websocket.close()
                return
            await websocket.close(code=1003, reason="Unexpected control message")
            return
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("Live session failed")
        with suppress(Exception):
            await websocket.close(code=1011, reason="Server error")
    finally:
        if not worker.done():
            worker.cancel()
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
