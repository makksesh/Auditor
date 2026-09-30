"""On-demand model processes. Exiting a worker releases its CUDA context and RAM."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRANSLATION_REPO = "jiangzhuo9357/opus-mt-en-ru-ct2"
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    source_language: str
    target_language: str | None = None

    def __post_init__(self):
        if self.source_language not in ("en", "ru"):
            raise ValueError("Supported source languages: en, ru")
        if self.target_language is not None and (self.source_language, self.target_language) != ("en", "ru"):
            raise ValueError("Translation is only supported for en to ru")


@dataclass(frozen=True)
class InferenceResult:
    text: str
    tokens: int


def source_eos_token(config: dict) -> str | None:
    """Transformers conversions expect the tokenizer to supply source EOS."""
    return config.get("eos_token", "</s>") if config.get("add_source_eos") is False else None


class ASRModel:
    def __init__(self, profile: ModelProfile):
        from faster_whisper import WhisperModel

        self.language = profile.source_language
        self.name = (
            os.getenv("AUDITOR_ASR_MODEL_EN", os.getenv("AUDITOR_ASR_MODEL", "distil-large-v3"))
            if self.language == "en" else os.getenv("AUDITOR_ASR_MODEL_MULTILINGUAL", "large-v3-turbo")
        )
        device = os.getenv("AUDITOR_DEVICE", "cuda")
        compute_type = os.getenv("AUDITOR_COMPUTE_TYPE", "float16")
        self.asr = WhisperModel(self.name, device=device, compute_type=compute_type)
        if self.language == "ru" and not self.asr.model.is_multilingual:
            raise ValueError("Russian transcription requires a multilingual ASR model")
        self.info = {
            "name": self.name, "device": self.asr.model.device,
            "compute_type": self.asr.model.compute_type, "language": self.language,
        }
        if os.getenv("AUDITOR_WARMUP", "1") == "1":
            self.run(np.zeros(16000, dtype=np.float32))

    def run(self, samples: np.ndarray | str) -> InferenceResult:
        segments, _ = self.asr.transcribe(
            samples, language=self.language, task="transcribe", beam_size=1,
            condition_on_previous_text=False, vad_filter=False, no_speech_threshold=0.6,
        )
        segments = list(segments)
        return InferenceResult(
            " ".join(segment.text.strip() for segment in segments).strip(),
            sum(len(segment.tokens) for segment in segments),
        )


class TranslationModel:
    def __init__(self, profile: ModelProfile):
        import json
        import ctranslate2
        import sentencepiece as spm
        from huggingface_hub import snapshot_download

        model_dir = os.getenv("AUDITOR_TRANSLATION_MODEL")
        if not model_dir:
            model_dir = snapshot_download(
                repo_id=TRANSLATION_REPO,
                allow_patterns=["config.json", "model.bin", "shared_vocabulary.json", "source.spm", "target.spm"],
            )
        model_path = Path(model_dir)
        for filename in ("model.bin", "config.json", "source.spm", "target.spm"):
            if not (model_path / filename).is_file():
                raise RuntimeError(f"Translation model lacks {filename}: {model_path}")
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        self._source_eos = source_eos_token(config)
        device = os.getenv("AUDITOR_TRANSLATION_DEVICE", "cpu")
        self.translator = ctranslate2.Translator(str(model_path), device=device)
        self.source_spm = spm.SentencePieceProcessor(model_file=str(model_path / "source.spm"))
        self.target_spm = spm.SentencePieceProcessor(model_file=str(model_path / "target.spm"))
        self.info = {"name": model_dir, "device": self.translator.device, "language": "ru"}

    def run(self, text: str) -> InferenceResult:
        source = self.source_spm.encode(text, out_type=str)
        if self._source_eos and (not source or source[-1] != self._source_eos):
            source.append(self._source_eos)
        max_length = min(128, max(24, int(len(source) * 1.8) + 8))
        result = self.translator.translate_batch(
            [source], beam_size=2, max_decoding_length=max_length,
            repetition_penalty=1.12, no_repeat_ngram_size=4,
        )
        tokens = result[0].hypotheses[0]
        return InferenceResult(self.target_spm.decode(tokens).strip(), len(tokens))


def _worker_main(connection, model_type, profile):
    # No CUDA objects are created in the HTTP server process or inherited via fork.
    try:
        model = model_type(profile)
        connection.send((True, model.info))
        while True:
            payload = connection.recv()
            try:
                connection.send((True, model.run(payload)))
            except Exception as exc:
                log.exception("Model inference failed")
                connection.send((False, str(exc)))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        log.exception("Model worker failed")
        try:
            connection.send((False, str(exc)))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class ModelWorker:
    def __init__(self, model_type, profile):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.lock = threading.Lock()
        self.close_lock = threading.Lock()
        self.process = context.Process(target=_worker_main, args=(child, model_type, profile), daemon=True)
        try:
            self.process.start()
            child.close()
            self.info = self._receive(float(os.getenv("AUDITOR_MODEL_LOAD_TIMEOUT", "300")))
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            raise TimeoutError("Model worker timed out")
        try:
            success, result = self.connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError("Model worker exited") from exc
        if not success:
            raise RuntimeError(result)
        return result

    def run(self, payload, *, timeout: float | None = None):
        try:
            with self.lock:
                self.connection.send(payload)
                if timeout is None:
                    timeout = float(os.getenv("AUDITOR_INFERENCE_TIMEOUT", "30"))
                return self._receive(timeout)
        except TimeoutError:
            # A late reply must never be mistaken for the next segment's reply.
            self.close()
            raise

    def close(self):
        with self.close_lock:
            self._close()

    def _close(self):
        # Termination also interrupts in-flight inference on a disconnected client.
        if self.process.pid is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=3)
            if self.process.is_alive():
                raise RuntimeError("Could not stop model worker")
        with self.lock:
            self.connection.close()


class Models:
    def __init__(self, profile: ModelProfile):
        self.asr = self.translator = None
        try:
            self.asr = ModelWorker(ASRModel, profile)
            if profile.target_language is not None:
                self.translator = ModelWorker(TranslationModel, profile)
        except BaseException:
            self.close()
            raise

    def describe(self):
        return {"asr": self.asr.info, "translation": self.translator.info if self.translator else None}

    def transcribe(self, samples: np.ndarray) -> InferenceResult:
        return self.asr.run(samples)

    def transcribe_file(self, path: str) -> InferenceResult:
        timeout = float(os.getenv("AUDITOR_FILE_INFERENCE_TIMEOUT", "3600"))
        return self.asr.run(path, timeout=timeout)

    def translate(self, text: str) -> InferenceResult:
        if self.translator is None:
            raise RuntimeError("Translation is disabled")
        return self.translator.run(text)

    def close(self):
        errors = []
        for worker in (self.asr, self.translator):
            if worker is not None:
                try:
                    worker.close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]
