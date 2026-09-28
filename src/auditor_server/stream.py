"""PCM framing and lightweight speech activity detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


SAMPLE_RATE = 16000
FRAME_SAMPLES = 320  # 20 ms
FRAME_BYTES = FRAME_SAMPLES * 2


@dataclass(frozen=True)
class StreamConfig:
    speech_rms: float = 0.012
    silence_seconds: float = 0.7
    max_phrase_seconds: float = 10.0
    preroll_seconds: float = 0.3


class AudioStream:
    def __init__(self, config: StreamConfig = StreamConfig()) -> None:
        self.config = config
        self._remainder = bytearray()
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, round(config.preroll_seconds / 0.02)))
        self._phrase: list[np.ndarray] = []
        self._silence_frames = 0
        self._pending: deque[np.ndarray] = deque()
        self.revision = 0
        self.phrase_id = 0

    def feed(self, data: bytes) -> bool:
        self._remainder.extend(data)
        changed = False
        while len(self._remainder) >= FRAME_BYTES:
            frame = np.frombuffer(self._remainder[:FRAME_BYTES], dtype="<i2").astype(np.float32) / 32768.0
            del self._remainder[:FRAME_BYTES]
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
            if (self._silence_frames * 0.02 >= self.config.silence_seconds or
                    len(self._phrase) * 0.02 >= self.config.max_phrase_seconds):
                self._finish_phrase()
        return changed

    def _finish_phrase(self) -> None:
        if self._phrase:
            self._pending.append(np.concatenate(self._phrase))
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
        self._finish_phrase()

    def pop_final(self) -> np.ndarray | None:
        return self._pending.popleft() if self._pending else None

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
