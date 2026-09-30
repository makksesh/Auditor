"""Serialize model replacement and single-client ownership on the event loop."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict

from .models import ModelProfile

log = logging.getLogger(__name__)


class ModelsBusy(RuntimeError):
    pass


class ModelsNotReady(RuntimeError):
    pass


class ModelManager:
    def __init__(self, factory):
        self.factory = factory
        self.models = None
        self.profile = None
        self.state = "unloaded"
        self.error = None
        self.active = False
        self.session = None
        self.last_session = None
        self.file_job = None
        self.last_file_job = None
        self.operation = None

    def snapshot(self):
        return {
            "state": self.state, "profile": asdict(self.profile) if self.profile else None,
            "models": self.models.describe() if self.models else None,
            "active_sessions": int(self.session is not None),
            "busy": self.active, "error": self.error,
        }

    def _check_idle(self):
        if self.active or (self.operation and not self.operation.done()):
            raise ModelsBusy("A session or model operation is in progress")

    def begin_load(self, profile: ModelProfile):
        self._check_idle()
        if self.state == "ready" and self.profile == profile:
            return None
        self.state = "loading"
        self.error = None
        self.operation = asyncio.create_task(self._load(profile))
        return self.operation

    async def _load(self, profile):
        try:
            await self._dispose()
            self.profile = profile
            self.models = await asyncio.to_thread(self.factory, profile)
            self.state = "ready"
        except Exception as exc:
            log.exception("Model loading failed")
            self.state = "error"
            self.error = str(exc)

    async def _dispose(self):
        if self.models is not None:
            await asyncio.to_thread(self.models.close)
            self.models = None
        self.profile = None

    def acquire(self, profile: ModelProfile):
        self._check_idle()
        if self.state != "ready" or profile != self.profile:
            raise ModelsNotReady("Prepare the matching profile with POST /v1/models/load first")
        self.active = True
        return self.models

    def reserve_file(self):
        self._check_idle()
        self.active = True

    def begin_reserved_file_load(self, profile: ModelProfile):
        if self.state == "ready" and self.profile == profile:
            return None
        self.state = "loading"
        self.error = None
        self.operation = asyncio.create_task(self._load(profile))
        return self.operation

    def prepared_file_models(self, profile: ModelProfile):
        if self.state != "ready" or self.profile != profile:
            raise ModelsNotReady(self.error or "Could not prepare the selected model")
        return self.models

    def begin_unload(self):
        self._check_idle()
        self.state = "unloading"
        self.operation = asyncio.create_task(self._unload())
        return self.operation

    async def _unload(self):
        try:
            await self._dispose()
            self.state = "unloaded"
            self.error = None
        except Exception as exc:
            log.exception("Model unloading failed")
            self.state = "error"
            self.error = str(exc)
        finally:
            self.active = False
            self.session = None
            self.file_job = None

    def release(self):
        if self.session is not None:
            self.last_session = self.session.metrics_snapshot()
        if self.file_job is not None:
            self.last_file_job = dict(self.file_job)
        self.state = "unloading"
        self.operation = asyncio.create_task(self._unload())
        return self.operation

    async def abandon_legacy_load(self, operation):
        # A cancelled old-protocol handshake must not leave newly loaded weights resident.
        await asyncio.shield(operation)
        if self.operation is operation and not self.active:
            await asyncio.shield(self.begin_unload())

    async def shutdown(self):
        if self.operation:
            await asyncio.shield(self.operation)
        await self._unload()
