"""PCM framing and lightweight speech activity detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import monotonic

import numpy as np


SAMPLE_RATE = 16000
FRAME_SAMPLES = 320  # 20 ms
FRAME_BYTES = FRAME_SAMPLES * 2


@dataclass(frozen=True)
class StreamConfig:
    speech_rms: float = 0.012
    silence_seconds: float = 0.7
    max_phrase_seconds: float = 6.0
    preroll_seconds: float = 0.3

    def __post_init__(self) -> None:
        if not 0 < self.speech_rms < 1:
            raise ValueError("speech_rms must be between 0 and 1")
        if not 0.1 <= self.silence_seconds <= 3:
            raise ValueError("silence_seconds must be between 0.1 and 3")
        if not 2 <= self.max_phrase_seconds <= 15:
            raise ValueError("max_phrase_seconds must be between 2 and 15")
        if not 0 <= self.preroll_seconds <= 2:
            raise ValueError("preroll_seconds must be between 0 and 2")


class AudioStream:
    def __init__(self, config: StreamConfig = StreamConfig()) -> None:
        self.config = config
        self._remainder = bytearray()
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, round(config.preroll_seconds / 0.02)))
        self._phrase: list[np.ndarray] = []
        self._silence_frames = 0
        self._pending: deque[tuple[np.ndarray, bool, float]] = deque()
        self._pending_samples = 0
        self.total_samples = 0
        self.revision = 0
        self.phrase_id = 0

    def feed(self, data: bytes) -> bool:
        self._remainder.extend(data)
        changed = False
        while len(self._remainder) >= FRAME_BYTES:
            frame = np.frombuffer(self._remainder[:FRAME_BYTES], dtype="<i2").astype(np.float32) / 32768.0
            del self._remainder[:FRAME_BYTES]
            self.total_samples += FRAME_SAMPLES
            rms = float(np.sqrt(np.mean(frame * frame)))
            speech = rms >= self.config.speech_rms
            if not self._phrase:
                if speech:
                    self.phrase_id += 1
                    self._phrase.extend(self._preroll)
                    self._phrase.append(frame)
                    self._preroll.clear()
                    self.revision += 1
                    changed = True
                else:
                    self._preroll.append(frame)
                continue

            self._phrase.append(frame)
            self.revision += 1
            changed = True
            self._silence_frames = 0 if speech else self._silence_frames + 1
            paused = self._silence_frames * 0.02 >= self.config.silence_seconds
            if paused or len(self._phrase) * 0.02 >= self.config.max_phrase_seconds:
                self._finish_phrase(paused=paused)
        return changed

    def _finish_phrase(self, *, paused: bool = False) -> None:
        if self._phrase:
            samples = np.concatenate(self._phrase)
            self._pending.append((samples, paused, monotonic()))
            self._pending_samples += len(samples)
            self._phrase = []
            self._silence_frames = 0
            self.revision += 1

    def finish(self) -> None:
        if self._remainder:
            # A partial sample cannot be decoded. The client normally sends even byte counts.
            even_bytes = len(self._remainder) & ~1
            if even_bytes and self._phrase:
                self._phrase.append(np.frombuffer(self._remainder[:even_bytes], dtype="<i2").astype(np.float32) / 32768.0)
            self._remainder.clear()
        self._finish_phrase(paused=True)

    def pop_final(self) -> np.ndarray | None:
        item = self.pop_final_item()
        return item[0] if item else None

    def pop_final_item(self) -> tuple[np.ndarray, bool, float] | None:
        if not self._pending:
            return None
        samples, paused, queued_at = self._pending.popleft()
        self._pending_samples -= len(samples)
        return samples, paused, queued_at

    def partial(self) -> tuple[int, int, np.ndarray] | None:
        if len(self._phrase) * 0.02 < 0.6:
            return None
        return self.phrase_id, self.revision, np.concatenate(self._phrase)

    @property
    def has_final(self) -> bool:
        return bool(self._pending)

    @property
    def final_count(self) -> int:
        return len(self._pending)

    @property
    def pending_seconds(self) -> float:
        return self._pending_samples / SAMPLE_RATE

    @property
    def oldest_pending_age(self) -> float:
        return monotonic() - self._pending[0][2] if self._pending else 0.0

    @property
    def active(self) -> bool:
        return bool(self._phrase)
