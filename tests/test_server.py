import json
import threading
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from auditor_server.main import app, valid_start
from auditor_server.models import Models
from auditor_server.stream import AudioStream
from auditor_server.text import SentenceAssembler, agreed_prefix


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

    def test_agreement_and_sentence_bounds(self):
        self.assertEqual(agreed_prefix("Hello my fri", "Hello my friend"), "Hello")
        assembler = SentenceAssembler(max_words=5)
        self.assertEqual(assembler.add("One two three. Four five"), ["One two three."])
        self.assertEqual(assembler.add("six seven eight nine ten eleven."), ["Four five six seven eight", "nine ten eleven."])
        self.assertEqual(assembler.flush(), [])
        assembler = SentenceAssembler()
        self.assertEqual(assembler.add("And with the Apple Watch series 12,"), [])
        self.assertEqual(
            assembler.add("This is another example."),
            ["And with the Apple Watch series 12,", "This is another example."],
        )

    def test_translation_does_not_block_english(self):
        release = threading.Event()

        class SlowTranslation(FakeModels):
            def translate(self, text):
                release.wait(timeout=5)
                return super().translate(text)

        try:
            with patch("auditor_server.main.Models", SlowTranslation), TestClient(app) as client:
                with client.websocket_connect("/v1/live") as ws:
                    ws.send_json(START)
                    for _ in range(2):
                        ws.send_bytes(pcm(0.8))
                        ws.send_bytes(pcm(0.8, 0))
                    first = ws.receive_json()
                    second = ws.receive_json()
                    self.assertEqual([first["type"], second["type"]], ["stable_en", "stable_en"])
                    self.assertEqual([first["segment_id"], second["segment_id"]], ["1", "2"])
                    release.set()
                    self.assertEqual(ws.receive_json()["type"], "translation_ru")
                    self.assertEqual(ws.receive_json()["type"], "translation_ru")
                    ws.send_json({"type": "end"})
        finally:
            release.set()

    def test_partial_uses_two_matching_passes(self):
        first_pass = threading.Event()

        class SignalingModels(FakeModels):
            def transcribe(self, samples):
                first_pass.set()
                return super().transcribe(samples)

        with patch("auditor_server.main.Models", SignalingModels), TestClient(app) as client:
            with client.websocket_connect("/v1/live") as ws:
                ws.send_json(START)
                ws.send_bytes(pcm(0.8))
                self.assertTrue(first_pass.wait(timeout=2))
                ws.send_bytes(pcm(0.4))
                self.assertEqual(ws.receive_json(), {"type": "partial_en", "text": "Hello,"})
                ws.send_json({"type": "end"})

    def test_translation_decoding_is_bounded(self):
        class Pieces:
            def encode(self, text, out_type):
                return list(text)

            def decode(self, tokens):
                return "".join(tokens)

        class Translator:
            def translate_batch(self, source, **options):
                self.options = options
                return [type("Result", (), {"hypotheses": [["ok"]]})()]

        model = Models.__new__(Models)
        model.source_spm = Pieces()
        model.target_spm = Pieces()
        model.translator = Translator()
        model._translation_lock = threading.Lock()
        self.assertEqual(model.translate("a" * 300), "ok")
        self.assertEqual(model.translator.options["max_decoding_length"], 128)
        self.assertGreater(model.translator.options["repetition_penalty"], 1)
        self.assertGreater(model.translator.options["no_repeat_ngram_size"], 0)

    def test_two_hour_stream_does_not_retain_old_audio(self):
        stream = AudioStream()
        speech = pcm(6.0)
        silence = pcm(0.8, 0)
        for _ in range(1060):  # just over two hours at 6.8 s per cycle
            stream.feed(speech)
            stream.feed(silence)
            while stream.pop_final() is not None:
                pass
            self.assertLessEqual(stream.pending_seconds, 6.0)
        self.assertGreaterEqual(stream.total_samples / 16000, 7200)
        self.assertEqual(stream.final_count, 0)
        self.assertFalse(stream.active)

    def test_asr_overload_closes_instead_of_accumulating_audio(self):
        started = threading.Event()
        release = threading.Event()

        class SlowASR(FakeModels):
            def transcribe(self, samples):
                started.set()
                release.wait(timeout=5)
                return super().transcribe(samples)

        with patch("auditor_server.main.Models", SlowASR), TestClient(app) as client:
            with client.websocket_connect("/v1/live") as ws:
                try:
                    ws.send_json(START)
                    ws.send_bytes(pcm(6.0))
                    self.assertTrue(started.wait(timeout=2))
                    for _ in range(3):
                        ws.send_bytes(pcm(6.0))
                    with self.assertRaises(WebSocketDisconnect) as closed:
                        ws.receive_json()
                    self.assertEqual(closed.exception.code, 1013)
                finally:
                    release.set()


if __name__ == "__main__":
    unittest.main()
