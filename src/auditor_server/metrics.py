"""Bounded session counters and cached, optional hardware telemetry."""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Timing:
    count: int = 0
    total_ms: float = 0
    last_ms: float | None = None

    def add(self, seconds):
        self.count += 1
        self.last_ms = seconds * 1000
        self.total_ms += self.last_ms

    def snapshot(self):
        return {"count": self.count, "last_ms": self.last_ms,
                "mean_ms": self.total_ms / self.count if self.count else None}


@dataclass
class SessionMetrics:
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    asr_final_tokens: int = 0
    asr_partial_tokens: int = 0
    translation_tokens: int = 0
    stable_segments: int = 0
    translation_errors: int = 0
    asr: Timing = field(default_factory=Timing)
    translation: Timing = field(default_factory=Timing)
    finalized_audio_to_asr: Timing = field(default_factory=Timing)
    stable_to_translation: Timing = field(default_factory=Timing)

    def snapshot(self):
        result = asdict(self)
        for name in ("asr", "translation", "finalized_audio_to_asr", "stable_to_translation"):
            result[name] = getattr(self, name).snapshot()
        result["asr_tokens_per_second"] = (
            (self.asr_final_tokens + self.asr_partial_tokens) / (self.asr.total_ms / 1000)
            if self.asr.total_ms else None
        )
        return result


class SystemMonitor:
    """Linux /proc and NVIDIA CLI telemetry, sampled off the event loop."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="auditor-metrics", daemon=True)
        self._latest = {"sampled_at": None, "cpu_percent": None, "ram": None,
                        "server_rss_bytes": None, "gpus": [], "errors": ["initializing"]}
        self._sampled = None
        self._cpu_previous = None

    def start(self):
        self._thread.start()

    def snapshot(self):
        return {**self._latest, "sample_age_seconds": time.monotonic() - self._sampled if self._sampled else None}

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self):
        while not self._stop.is_set():
            sampled = time.monotonic()
            self._latest = self._sample()
            self._sampled = sampled
            self._stop.wait(1)

    def _sample(self):
        result = {"sampled_at": utc_now(), "cpu_percent": None, "ram": None,
                  "server_rss_bytes": None, "gpus": [], "errors": []}
        if sys.platform == "linux":
            for key, getter in (("cpu_percent", self._cpu), ("ram", self._ram),
                                ("server_rss_bytes", self._rss)):
                try:
                    result[key] = getter()
                except (OSError, ValueError, KeyError, IndexError) as exc:
                    result["errors"].append(f"{key}: {exc}")
        else:
            result["errors"].append("CPU/RAM telemetry requires Linux /proc")
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.total,memory.used,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=True,
            )
            result["gpus"] = parse_gpus(query.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            result["errors"].append(f"GPU telemetry unavailable: {exc}")
        return result

    def _cpu(self):
        # guest and guest_nice are already included in user/nice: sum only first 8.
        values = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        total, idle = sum(values), values[3] + values[4]
        previous, self._cpu_previous = self._cpu_previous, (total, idle)
        if previous is None or total <= previous[0]:
            return None
        return max(0.0, min(100.0, 100 * (1 - (idle - previous[1]) / (total - previous[0]))))

    def _ram(self):
        values = {line.split(':')[0]: int(line.split()[1]) * 1024
                  for line in Path("/proc/meminfo").read_text().splitlines() if len(line.split()) >= 2}
        total, available = values["MemTotal"], values["MemAvailable"]
        return {"total_bytes": total, "available_bytes": available,
                "used_bytes": total - available, "percent": (total - available) / total * 100}

    def _rss(self):
        pending, visited, rss = [os.getpid()], set(), 0
        while pending:
            pid = pending.pop()
            if pid in visited:
                continue
            visited.add(pid)
            try:
                rss += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                # Children can belong to the thread that started a model worker.
                for children in Path(f"/proc/{pid}/task").glob("*/children"):
                    pending.extend(int(x) for x in children.read_text().split())
            except (FileNotFoundError, ProcessLookupError):
                continue
        return rss


def parse_gpus(output):
    def number(value):
        try:
            return float(value.strip())
        except ValueError:
            return None

    def memory(value):
        value = number(value)
        return int(value * 1024 * 1024) if value is not None else None

    result = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if len(row) != 7:
            raise ValueError("Unexpected nvidia-smi response")
        index, uuid, name, utilization, total, used, free = row
        total, used, free = memory(total), memory(used), memory(free)
        result.append({
            "index": int(index), "uuid": uuid.strip(), "name": name.strip(),
            "utilization_percent": number(utilization),
            "vram": {"total_bytes": total, "used_bytes": used, "free_bytes": free,
                     "percent": used / total * 100 if total and used is not None else None},
        })
    return result
