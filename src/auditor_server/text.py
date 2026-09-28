"""Bounded text assembly for subtitle translation."""

from __future__ import annotations

import re


SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def normalize(text: str) -> str:
    return " ".join(text.split())


def agreed_prefix(previous: str, current: str) -> str:
    """Return words confirmed by two consecutive ASR passes."""
    old_words = previous.split()
    new_words = current.split()
    common = []
    for old, new in zip(old_words, new_words):
        if old.casefold().strip(".,!?;:") != new.casefold().strip(".,!?;:"):
            break
        common.append(new)
    # The last word is often still being spoken, even if both passes guessed it.
    return " ".join(common[:-1]) if len(common) > 1 else ""


class SentenceAssembler:
    def __init__(self, max_words: int = 32) -> None:
        self.max_words = max_words
        self.tail = ""

    def add(self, text: str) -> list[str]:
        text = normalize(text)
        if not text:
            return []
        ready = []
        # Whisper sometimes starts a new sentence after a forced audio cut,
        # leaving the previous chunk with a comma instead of a full stop.
        if self.tail.endswith(",") and text[0].isupper():
            ready.extend(self.flush())
        self.tail = normalize(f"{self.tail} {text}")
        parts = SENTENCE_END.split(self.tail)
        complete, self.tail = parts[:-1], parts[-1]
        if self.tail.endswith((".", "!", "?")):
            complete.append(self.tail)
            self.tail = ""
        for part in complete:
            ready.extend(self._bound(part))
        words = self.tail.split()
        while len(words) >= self.max_words:
            ready.append(" ".join(words[:self.max_words]))
            words = words[self.max_words:]
        self.tail = " ".join(words)
        return ready

    def flush(self) -> list[str]:
        if not self.tail:
            return []
        text, self.tail = self.tail, ""
        return self._bound(text)

    def _bound(self, text: str) -> list[str]:
        words = text.split()
        return [" ".join(words[i:i + self.max_words]) for i in range(0, len(words), self.max_words)]
