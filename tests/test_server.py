import json
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from auditor_server.main import app, valid_start
from auditor_server.stream import AudioStream


START = {
    "type": "start",
    "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1},
    "source_language": "en",
    "target_language": "ru",
}


def pcm(seconds: float, amplitude: int = 6000) -> bytes:
    count = int(seconds * 16000)
    return np.full(count, amplitude, dtype="<i2").tobytes()


class FakeModels:
    def transcribe(self, samples):
        return "Hello, friend."

    def translate(self, text):
        return "Привет, друг."


class ServerTests(unittest.TestCase):
    def test_start_validation(self):
        self.assertTrue(valid_start(json.dumps(START)))
        self.assertFalse(valid_start("garbage"))
        self.assertFalse(valid_start(json.dumps({**START, "source_language": "ru"})))

    def test_pcm_frames_can_span_websocket_messages(self):
        stream = AudioStream()
        sound = pcm(0.8)
        stream.feed(sound[:123])
        stream.feed(sound[123:])
        self.assertIsNotNone(stream.partial())
        stream.feed(pcm(0.8, 0))
        final = stream.pop_final()
        self.assertIsNotNone(final)
        self.assertGreater(len(final), 16000)

    def test_websocket_final_and_translation(self):
        with patch("auditor_server.main.Models", FakeModels), TestClient(app) as client:
            with client.websocket_connect("/v1/live") as ws:
                ws.send_json(START)
                audio = pcm(0.8)
                ws.send_bytes(audio[:127])
                ws.send_bytes(audio[127:])
                ws.send_json({"type": "end"})
                events = [ws.receive_json() for _ in range(2)]
                stable, translation = events
                self.assertEqual(stable, {"type": "stable_en", "segment_id": "1", "text": "Hello, friend."})
                self.assertEqual(translation, {"type": "translation_ru", "segment_id": "1", "text": "Привет, друг."})

    def test_partial_then_stable(self):
        with patch("auditor_server.main.Models", FakeModels), TestClient(app) as client:
            with client.websocket_connect("/v1/live") as ws:
                ws.send_json(START)
                ws.send_bytes(pcm(0.8))
                self.assertEqual(ws.receive_json(), {"type": "partial_en", "text": "Hello, friend."})
                ws.send_bytes(pcm(0.8, 0))
                self.assertEqual(ws.receive_json(), {"type": "stable_en", "segment_id": "1", "text": "Hello, friend."})
                self.assertEqual(ws.receive_json(), {"type": "partial_en", "text": ""})
                self.assertEqual(ws.receive_json(), {"type": "translation_ru", "segment_id": "1", "text": "Привет, друг."})
                ws.send_json({"type": "end"})


if __name__ == "__main__":
    unittest.main()
