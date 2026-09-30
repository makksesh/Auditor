import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from auditor_server.main import AUDIO_FORMAT, app
from auditor_server.metrics import SystemMonitor, parse_gpus
from auditor_server.models import ASRModel, InferenceResult, ModelProfile, Models, ModelWorker
from test_server import FakeModels, pcm


def wait_state(client, state, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get("/v1/system").json()
        if status["runtime"]["state"] == state:
            return status
        time.sleep(0.01)
    raise AssertionError(f"Expected {state}, got {status}")


def prepare(client, language="ru", target=None):
    response = client.post("/v1/models/load", json={"source_language": language, "target_language": target})
    assert response.status_code == 202, response.text
    return wait_state(client, "ready")


def start(ws, language="ru", target=None):
    ws.send_json({"type": "start", "audio": AUDIO_FORMAT, "source_language": language, "target_language": target})
    return ws.receive_json()


class LifecycleAPITests(unittest.TestCase):
    def test_ru_session_metrics_and_automatic_unload(self):
        instances = []

        def factory(profile):
            model = FakeModels(profile)
            instances.append(model)
            return model

        with patch("auditor_server.main.Models", factory), TestClient(app) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ready"})
            self.assertEqual(client.get("/v1/system").json()["runtime"]["state"], "unloaded")
            self.assertEqual(instances, [])
            prepare(client)
            with client.websocket_connect("/v2/live") as ws:
                self.assertEqual(start(ws)["source_language"], "ru")
                self.assertEqual(client.post("/v1/models/unload").status_code, 409)
                self.assertEqual(client.post("/v1/models/load", json={"source_language": "en"}).status_code, 409)
                ws.send_bytes(pcm(0.8))
                ws.send_json({"type": "end"})
                self.assertEqual(ws.receive_json(), {"type": "stable", "text": "Привет, друг.", "segment_id": "1", "language": "ru"})
                self.assertEqual(ws.receive_json(), {"type": "ended", "complete": True})
            status = wait_state(client, "unloaded")
            self.assertIsNone(status["session"])
            final = status["last_session"]
            self.assertEqual(final["asr_final_tokens"], 4)
            self.assertEqual(final["translation_tokens"], 0)
            self.assertEqual(final["translation"]["count"], 0)
            self.assertEqual(final["stable_segments"], 1)
            self.assertGreaterEqual(final["asr"]["last_ms"], 0)
            self.assertGreaterEqual(final["finalized_audio_to_asr"]["last_ms"], final["asr"]["last_ms"])
            self.assertIsNotNone(final["ended_at"])
            self.assertTrue(instances[0].closed)

    def test_switch_language_and_manual_unload(self):
        instances = []
        def factory(profile):
            instances.append(FakeModels(profile))
            return instances[-1]
        with patch("auditor_server.main.Models", factory), TestClient(app) as client:
            prepare(client, "en")
            prepare(client, "en")  # Idempotent, no reload.
            self.assertEqual(len(instances), 1)
            prepare(client, "ru")
            self.assertTrue(instances[0].closed)
            self.assertEqual(len(instances), 2)
            self.assertEqual(client.post("/v1/models/unload").status_code, 202)
            wait_state(client, "unloaded")
            self.assertTrue(instances[1].closed)
            client.post("/v1/models/unload")
            wait_state(client, "unloaded")

    def test_validation_and_mismatched_profile(self):
        with patch("auditor_server.main.Models", FakeModels), TestClient(app) as client:
            for payload in ({"source_language": "de"}, {"source_language": "ru", "target_language": "ru"},
                            {"source_language": "ru", "model": "arbitrary"}):
                self.assertEqual(client.post("/v1/models/load", json=payload).status_code, 422)
            with client.websocket_connect("/v2/live") as ws:
                self.assertEqual(start(ws)["code"], "models_not_ready")
            prepare(client, "en")
            with client.websocket_connect("/v2/live") as ws:
                self.assertEqual(start(ws)["code"], "models_not_ready")
            self.assertEqual(client.get("/v1/system").json()["runtime"]["profile"]["source_language"], "en")

    def test_single_owner_and_disconnect_reconnect(self):
        with patch("auditor_server.main.Models", FakeModels), TestClient(app) as client:
            prepare(client)
            with client.websocket_connect("/v2/live") as first:
                self.assertEqual(start(first)["type"], "ready")
                with client.websocket_connect("/v2/live") as second:
                    self.assertEqual(start(second)["code"], "busy")
                self.assertEqual(client.get("/v1/system").json()["runtime"]["active_sessions"], 1)
            wait_state(client, "unloaded")
            prepare(client, "en")
            with client.websocket_connect("/v2/live") as ws:
                self.assertEqual(start(ws, "en")["type"], "ready")
                ws.send_bytes(pcm(0.8))
                ws.send_json({"type": "end"})
                self.assertEqual(ws.receive_json()["language"], "en")
                self.assertTrue(ws.receive_json()["complete"])
            wait_state(client, "unloaded")

    def test_loading_is_nonblocking_and_conflicts_are_rejected(self):
        release = threading.Event()
        def factory(profile):
            release.wait(5)
            return FakeModels(profile)
        with patch("auditor_server.main.Models", factory), TestClient(app) as client:
            try:
                self.assertEqual(client.post("/v1/models/load", json={"source_language": "ru"}).status_code, 202)
                self.assertEqual(client.get("/v1/system").json()["runtime"]["state"], "loading")
                self.assertEqual(client.post("/v1/models/load", json={"source_language": "en"}).status_code, 409)
                self.assertEqual(client.post("/v1/models/unload").status_code, 409)
            finally:
                release.set()
            wait_state(client, "ready")

    def test_failed_load_can_be_retried(self):
        calls = 0
        def factory(profile):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("not enough memory")
            return FakeModels(profile)
        with patch("auditor_server.main.Models", factory), TestClient(app) as client:
            client.post("/v1/models/load", json={"source_language": "ru"})
            status = wait_state(client, "error")
            self.assertIn("not enough memory", status["runtime"]["error"])
            self.assertIsNone(status["runtime"]["models"])
            prepare(client)

    def test_disconnect_during_inference_closes_model(self):
        started, release = threading.Event(), threading.Event()
        class BlockingModels(FakeModels):
            def transcribe(self, samples):
                started.set()
                release.wait(5)
                return super().transcribe(samples)
            def close(self):
                release.set()
                super().close()
        with patch("auditor_server.main.Models", BlockingModels), TestClient(app) as client:
            prepare(client)
            with client.websocket_connect("/v2/live") as ws:
                start(ws)
                ws.send_bytes(pcm(0.8))
                self.assertTrue(started.wait(2))
            wait_state(client, "unloaded")
            self.assertTrue(release.is_set())

    def test_asr_failure_without_further_client_messages_unloads(self):
        class FailingModels(FakeModels):
            def transcribe(self, samples):
                raise RuntimeError("worker died")
        with patch("auditor_server.main.Models", FailingModels), TestClient(app) as client:
            prepare(client)
            with client.websocket_connect("/v2/live") as ws:
                start(ws)
                ws.send_bytes(pcm(0.8))
                self.assertEqual(ws.receive_json()["code"], "server_error")
                with self.assertRaises(WebSocketDisconnect) as error:
                    ws.receive_json()
                self.assertEqual(error.exception.code, 1011)
            wait_state(client, "unloaded")

    def test_end_timeout_reports_incomplete_and_frees_models(self):
        release = threading.Event()
        class SlowModels(FakeModels):
            def transcribe(self, samples):
                release.wait(5)
                return super().transcribe(samples)
            def close(self):
                release.set()
                super().close()
        with patch("auditor_server.main.Models", SlowModels), patch.dict(os.environ, {"AUDITOR_END_TIMEOUT": "0.01"}), TestClient(app) as client:
            prepare(client)
            with client.websocket_connect("/v2/live") as ws:
                start(ws)
                ws.send_bytes(pcm(0.8))
                ws.send_json({"type": "end"})
                self.assertEqual(ws.receive_json(), {"type": "ended", "complete": False})
            wait_state(client, "unloaded")
            self.assertTrue(release.is_set())

    def test_legacy_cancelled_loading_does_not_leak_models(self):
        loading, release = threading.Event(), threading.Event()
        instances = []
        def factory(profile):
            loading.set()
            release.wait(5)
            instances.append(FakeModels(profile))
            return instances[-1]
        with patch("auditor_server.main.Models", factory), TestClient(app) as client:
            try:
                with client.websocket_connect("/v1/live") as ws:
                    ws.send_json({"type": "start", "audio": AUDIO_FORMAT, "source_language": "en", "target_language": "ru"})
                    self.assertTrue(loading.wait(2))
            finally:
                release.set()
            wait_state(client, "unloaded")
            self.assertTrue(instances[0].closed)

    def test_disconnect_after_end_interrupts_final_processing(self):
        started, release = threading.Event(), threading.Event()
        class BlockingModels(FakeModels):
            def transcribe(self, samples):
                started.set()
                release.wait(15)
                return super().transcribe(samples)
            def close(self):
                release.set()
                super().close()
        with patch("auditor_server.main.Models", BlockingModels), TestClient(app) as client:
            prepare(client)
            with client.websocket_connect("/v2/live") as ws:
                start(ws)
                ws.send_bytes(pcm(0.8))
                ws.send_json({"type": "end"})
                self.assertTrue(started.wait(2))
                ws.close()
                self.assertEqual(wait_state(client, "unloaded", timeout=2)["runtime"]["active_sessions"], 0)
            self.assertTrue(release.is_set())

    def test_inference_timeout_is_not_a_start_timeout(self):
        class TimeoutModels(FakeModels):
            def transcribe(self, samples):
                raise TimeoutError("model timed out")
        with patch("auditor_server.main.Models", TimeoutModels), TestClient(app) as client:
            prepare(client)
            with client.websocket_connect("/v2/live") as ws:
                start(ws)
                ws.send_bytes(pcm(0.8))
                self.assertEqual(ws.receive_json()["code"], "server_error")
            wait_state(client, "unloaded")

    def test_v2_optional_translation(self):
        with patch("auditor_server.main.Models", FakeModels), TestClient(app) as client:
            prepare(client, "en", "ru")
            with client.websocket_connect("/v2/live") as ws:
                start(ws, "en", "ru")
                ws.send_bytes(pcm(0.8))
                ws.send_json({"type": "end"})
                stable, translation, ended = [ws.receive_json() for _ in range(3)]
                self.assertEqual(stable["type"], "stable")
                self.assertEqual(translation["type"], "translation")
                self.assertEqual(translation["language"], "ru")
                self.assertEqual(translation["segment_id"], stable["segment_id"])
                self.assertTrue(ended["complete"])
            self.assertEqual(wait_state(client, "unloaded")["last_session"]["translation_tokens"], 5)


class EchoModel:
    def __init__(self, profile):
        self.info = {"pid": os.getpid()}
    def run(self, payload):
        if isinstance(payload, dict) and "block" in payload:
            Path(payload["block"]).write_text("started")
            time.sleep(30)
        return payload


class FailingModel:
    def __init__(self, profile):
        raise RuntimeError("intentional initialization failure")


class ModelTests(unittest.TestCase):
    def test_worker_is_separate_process_and_is_reaped(self):
        worker = ModelWorker(EchoModel, ModelProfile("ru"))
        try:
            self.assertNotEqual(worker.info["pid"], os.getpid())
            self.assertEqual(worker.run("Привет"), "Привет")
        finally:
            worker.close()
        self.assertFalse(worker.process.is_alive())
        worker.close()

    def test_closing_worker_interrupts_inflight_request(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "started"
            worker = ModelWorker(EchoModel, ModelProfile("ru"))
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = executor.submit(worker.run, {"block": str(marker)})
                try:
                    deadline = time.monotonic() + 3
                    while not marker.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(marker.exists())
                finally:
                    worker.close()
                with self.assertRaises((RuntimeError, OSError)):
                    result.result(timeout=2)
                self.assertFalse(worker.process.is_alive())

    def test_worker_timeout_terminates_process_and_discards_late_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = ModelWorker(EchoModel, ModelProfile("en"))
            try:
                with patch.dict(os.environ, {"AUDITOR_INFERENCE_TIMEOUT": "0.05"}):
                    with self.assertRaises(TimeoutError):
                        worker.run({"block": str(Path(directory) / "started")})
                self.assertFalse(worker.process.is_alive())
                with self.assertRaises(OSError):
                    worker.run("next segment")
            finally:
                worker.close()

    def test_failed_worker_start_leaves_no_child(self):
        before = {p.pid for p in multiprocessing.active_children()}
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            ModelWorker(FailingModel, ModelProfile("ru"))
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_models_only_load_translation_when_requested(self):
        with patch("auditor_server.models.ModelWorker") as worker:
            models = Models(ModelProfile("ru"))
            self.assertEqual(worker.call_count, 1)
            self.assertIsNone(models.translator)
            models.close()
            worker.return_value.close.assert_called_once()
        with patch("auditor_server.models.ModelWorker") as worker:
            Models(ModelProfile("en", "ru"))
            self.assertEqual(worker.call_count, 2)

    def test_translation_load_failure_disposes_asr(self):
        asr = MagicMock()
        with patch("auditor_server.models.ModelWorker", side_effect=[asr, RuntimeError("bad translation")]):
            with self.assertRaisesRegex(RuntimeError, "bad translation"):
                Models(ModelProfile("en", "ru"))
        asr.close.assert_called_once()

    def test_asr_selects_model_and_forces_language(self):
        for language, expected in (("en", "distil-large-v3"), ("ru", "large-v3-turbo")):
            whisper = MagicMock()
            whisper.return_value.transcribe.return_value = ([SimpleNamespace(text="hello", tokens=[1, 2, 3])], None)
            with patch.dict(os.environ, {"AUDITOR_WARMUP": "0"}, clear=True), \
                 patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=whisper)}):
                model = ASRModel(ModelProfile(language))
                result = model.run(np.zeros(16000))
                file_result = model.run("/tmp/sample.wav")
            self.assertEqual(whisper.call_args.args[0], expected)
            self.assertEqual(whisper.return_value.transcribe.call_args.args[0], "/tmp/sample.wav")
            self.assertEqual(whisper.return_value.transcribe.call_args.kwargs["language"], language)
            self.assertEqual(whisper.return_value.transcribe.call_args.kwargs["task"], "transcribe")
            self.assertEqual(result.tokens, 3)
            self.assertEqual(file_result.tokens, 3)


class MetricsTests(unittest.TestCase):
    def test_gpu_units_and_missing_utilization(self):
        result = parse_gpus('0, GPU-123, Test GPU, 25, 8192, 2048, 6144\n1, GPU-456, Other GPU, [N/A], 4096, 0, 4096\n')
        self.assertEqual(result[0]["vram"]["used_bytes"], 2048 * 1024 * 1024)
        self.assertEqual(result[0]["vram"]["percent"], 25)
        self.assertIsNone(result[1]["utilization_percent"])
        self.assertEqual(result[1]["vram"]["percent"], 0)

    def test_missing_gpu_is_not_reported_as_zero(self):
        monitor = SystemMonitor()
        with patch("auditor_server.metrics.subprocess.run", side_effect=FileNotFoundError("nvidia-smi")):
            sample = monitor._sample()
        self.assertEqual(sample["gpus"], [])
        self.assertTrue(sample["errors"])

    def test_cpu_first_sample_unknown_then_delta(self):
        monitor = SystemMonitor()
        with patch("auditor_server.metrics.Path.read_text", side_effect=["cpu 100 0 100 800 0 0 0 0 50 0\n", "cpu 150 0 150 900 0 0 0 0 80 0\n"]):
            self.assertIsNone(monitor._cpu())
            self.assertEqual(monitor._cpu(), 50)


if __name__ == "__main__":
    unittest.main()
