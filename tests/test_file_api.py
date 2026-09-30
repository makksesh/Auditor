import asyncio
import io
import os
import threading
import tempfile
import time
import unittest
import wave
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from auditor_server.main import app
from auditor_server.models import InferenceResult
from test_lifecycle import wait_state
from test_server import FakeModels


def wav(seconds=0.8):
    output = io.BytesIO()
    with wave.open(output, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(b"\x00\x00" * int(seconds * 16000))
    return output.getvalue()


def upload(client, language="ru", audio=None, **headers):
    return client.post(
        f"/v1/transcriptions?source_language={language}",
        content=wav() if audio is None else audio,
        headers={"Content-Type": "audio/wav", **headers},
    )


class FileAPITests(unittest.TestCase):
    def test_ru_file_loads_transcribes_and_unloads(self):
        instances, paths = [], []

        class FileModels(FakeModels):
            def transcribe_file(self, path):
                paths.append(path)
                self.asserted_content = Path(path).read_bytes()
                return InferenceResult("Готовая запись целиком.", 6)

        def factory(profile):
            model = FileModels(profile)
            instances.append(model)
            return model

        with patch("auditor_server.main.Models", factory), TestClient(app) as client:
            response = upload(client)
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertEqual(data["source_language"], "ru")
            self.assertEqual(data["text"], "Готовая запись целиком.")
            self.assertEqual(data["tokens"], 6)
            self.assertAlmostEqual(data["duration_seconds"], 0.8, places=1)
            self.assertGreaterEqual(data["processing_ms"], 0)
            self.assertEqual(instances[0].profile.source_language, "ru")
            self.assertIsNone(instances[0].profile.target_language)
            self.assertTrue(instances[0].closed)
            self.assertEqual(instances[0].asserted_content, wav())
            self.assertFalse(Path(paths[0]).exists())
            status = wait_state(client, "unloaded")
            self.assertIsNone(status["file_job"])
            self.assertEqual(status["last_file_job"]["status"], "complete")
            self.assertEqual(status["last_file_job"]["bytes"], len(wav()))

    def test_en_file_uses_en_profile_and_never_translates(self):
        profiles = []
        class FileModels(FakeModels):
            def transcribe_file(self, path):
                return InferenceResult("Complete English transcript.", 5)
        def factory(profile):
            profiles.append(profile)
            return FileModels(profile)
        with patch("auditor_server.main.Models", factory), TestClient(app) as client:
            self.assertEqual(upload(client, "en").json()["text"], "Complete English transcript.")
            self.assertEqual((profiles[0].source_language, profiles[0].target_language), ("en", None))
            wait_state(client, "unloaded")

    def test_prepared_model_is_reused_then_unloaded(self):
        instances = []
        class FileModels(FakeModels):
            def transcribe_file(self, path):
                return InferenceResult("Ready model reused.", 3)
        def factory(profile):
            instances.append(FileModels(profile))
            return instances[-1]
        with patch("auditor_server.main.Models", factory), TestClient(app) as client:
            self.assertEqual(client.post("/v1/models/load", json={"source_language": "en"}).status_code, 202)
            wait_state(client, "ready")
            self.assertEqual(upload(client, "en").json()["text"], "Ready model reused.")
            self.assertEqual(len(instances), 1)
            self.assertTrue(instances[0].closed)
            wait_state(client, "unloaded")

    def test_invalid_audio_and_limits_leave_no_models_or_files(self):
        instances, paths = [], []
        def factory(profile):
            model = FakeModels(profile)
            instances.append(model)
            return model
        with patch("auditor_server.main.Models", factory), TestClient(app) as client:
            self.assertEqual(upload(client, audio=b"").status_code, 422)
            self.assertEqual(upload(client, audio=b"not audio").status_code, 422)
            self.assertEqual(upload(client, audio=wav(), **{"Content-Type": "text/plain"}).status_code, 415)
            self.assertEqual(upload(client, "de").status_code, 422)
            with patch.dict(os.environ, {"AUDITOR_MAX_UPLOAD_BYTES": "100"}):
                self.assertEqual(upload(client).status_code, 413)
            self.assertEqual(instances, [])
            status = wait_state(client, "unloaded")
            self.assertEqual(status["last_file_job"]["status"], "error")

    def test_file_job_blocks_live_and_other_upload(self):
        started, release = threading.Event(), threading.Event()
        class SlowFile(FakeModels):
            def transcribe_file(self, path):
                started.set()
                release.wait(5)
                return InferenceResult("ok", 1)
            def close(self):
                release.set()
                super().close()
        with patch("auditor_server.main.Models", SlowFile), TestClient(app) as client:
            response = {}
            thread = threading.Thread(target=lambda: response.setdefault("value", upload(client)))
            thread.start()
            try:
                self.assertTrue(started.wait(2))
                status = client.get("/v1/system").json()
                self.assertEqual(status["file_job"]["status"], "transcribing")
                self.assertEqual(status["runtime"]["active_sessions"], 0)
                self.assertTrue(status["runtime"]["busy"])
                self.assertEqual(upload(client).status_code, 409)
                self.assertEqual(client.post("/v1/models/load", json={"source_language": "en"}).status_code, 409)
                with client.websocket_connect("/v2/live") as ws:
                    ws.send_json({"type": "start", "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
                                  "source_language": "ru", "target_language": None})
                    self.assertEqual(ws.receive_json()["code"], "busy")
            finally:
                release.set()
                thread.join(timeout=5)
            self.assertEqual(response["value"].status_code, 200)
            wait_state(client, "unloaded")

    def test_load_or_inference_error_releases_reservation(self):
        class FailingFile(FakeModels):
            def transcribe_file(self, path):
                raise RuntimeError("GPU died")
        with patch("auditor_server.main.Models", FailingFile), TestClient(app) as client:
            self.assertEqual(upload(client).status_code, 500)
            status = wait_state(client, "unloaded")
            self.assertEqual(status["last_file_job"]["status"], "error")
        with patch("auditor_server.main.Models", side_effect=RuntimeError("load failed")), TestClient(app) as client:
            self.assertEqual(upload(client).status_code, 503)
            status = wait_state(client, "unloaded")
            self.assertEqual(status["last_file_job"]["status"], "error")

    def test_file_inference_timeout_clears_model(self):
        instances = []
        class TimeoutFile(FakeModels):
            def transcribe_file(self, path):
                raise TimeoutError("file inference timed out")
        def factory(profile):
            instances.append(TimeoutFile(profile))
            return instances[-1]
        with patch("auditor_server.main.Models", factory), TestClient(app) as client:
            self.assertEqual(upload(client).status_code, 504)
            self.assertTrue(instances[0].closed)
            self.assertEqual(wait_state(client, "unloaded")["last_file_job"]["status"], "error")

    def test_real_decoder_reads_uploaded_wav_from_extensionless_temp_file(self):
        from faster_whisper.audio import decode_audio

        with tempfile.NamedTemporaryFile(suffix=".audio") as file:
            file.write(wav())
            file.flush()
            samples = decode_audio(file.name)
        self.assertEqual(len(samples), int(0.8 * 16000))

    def test_cancelled_http_request_finishes_cleanup(self):
        started, release = threading.Event(), threading.Event()
        instances, paths = [], []
        class BlockingFile(FakeModels):
            def transcribe_file(self, path):
                paths.append(path)
                started.set()
                release.wait(5)
                return InferenceResult("not delivered", 2)
            def close(self):
                release.set()
                super().close()
        def factory(profile):
            instances.append(BlockingFile(profile))
            return instances[-1]

        async def exercise():
            with patch("auditor_server.main.Models", factory):
                async with app.router.lifespan_context(app):
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                        request = asyncio.create_task(client.post(
                            "/v1/transcriptions?source_language=ru", content=wav(),
                            headers={"Content-Type": "audio/wav"},
                        ))
                        try:
                            self.assertTrue(await asyncio.to_thread(started.wait, 2))
                            request.cancel()
                            with suppress(asyncio.CancelledError):
                                await request
                            deadline = time.monotonic() + 2
                            while app.state.manager.state != "unloaded" and time.monotonic() < deadline:
                                await asyncio.sleep(0.01)
                            self.assertEqual(app.state.manager.state, "unloaded")
                            self.assertEqual(app.state.manager.last_file_job["status"], "disconnected")
                            self.assertTrue(instances[0].closed)
                            self.assertFalse(Path(paths[0]).exists())
                        finally:
                            release.set()
                            request.cancel()
                            await asyncio.gather(request, return_exceptions=True)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
