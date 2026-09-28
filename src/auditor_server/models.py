"""Local ASR and translation model adapters."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np


TRANSLATION_REPO = "jiangzhuo9357/opus-mt-en-ru-ct2"


class Models:
    def __init__(self) -> None:
        from faster_whisper import WhisperModel
        import ctranslate2
        import sentencepiece as spm
        from huggingface_hub import snapshot_download

        self._lock = threading.Lock()
        asr_name = os.getenv("AUDITOR_ASR_MODEL", "distil-large-v3")
        device = os.getenv("AUDITOR_DEVICE", "cuda")
        compute_type = os.getenv("AUDITOR_COMPUTE_TYPE", "float16")
        self.asr = WhisperModel(asr_name, device=device, compute_type=compute_type)

        model_dir = os.getenv("AUDITOR_TRANSLATION_MODEL")
        if not model_dir:
            model_dir = snapshot_download(
                repo_id=TRANSLATION_REPO,
                allow_patterns=["config.json", "model.bin", "shared_vocabulary.json", "source.spm", "target.spm"],
            )
        model_path = Path(model_dir)
        for filename in ("model.bin", "source.spm", "target.spm"):
            if not (model_path / filename).is_file():
                raise RuntimeError(f"Translation model lacks {filename}: {model_path}")

        translation_device = os.getenv("AUDITOR_TRANSLATION_DEVICE", "cpu")
        self.translator = ctranslate2.Translator(str(model_path), device=translation_device)
        self.source_spm = spm.SentencePieceProcessor(model_file=str(model_path / "source.spm"))
        self.target_spm = spm.SentencePieceProcessor(model_file=str(model_path / "target.spm"))

        # Trigger CUDA initialization before accepting the first client.
        if os.getenv("AUDITOR_WARMUP", "1") == "1":
            self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, samples: np.ndarray) -> str:
        with self._lock:
            segments, _ = self.asr.transcribe(
                samples,
                language="en",
                task="transcribe",
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
                no_speech_threshold=0.6,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()

    def translate(self, text: str) -> str:
        source = self.source_spm.encode(text, out_type=str)
        with self._lock:
            result = self.translator.translate_batch([source], beam_size=2)
        return self.target_spm.decode(result[0].hypotheses[0]).strip()
