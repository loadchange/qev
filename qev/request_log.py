"""Local, replayable logs for the two non-streaming model HTTP endpoints."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

_LOG = logging.getLogger(__name__)
_ENDPOINTS = {"/v1/systemone", "/v1/chat/completions"}
_IMAGE = re.compile(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)")


def _utc():
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _digest(body):
    return hashlib.sha256(body).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _strings(value, pointer=""):
    if isinstance(value, str):
        yield pointer, value
    elif isinstance(value, dict):
        for key, item in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            yield from _strings(item, f"{pointer}/{escaped}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{pointer}/{index}")


@dataclass
class _Record:
    directory: Path
    metadata: dict
    failed: bool = False


class RequestLog:
    """Write request folders and a small append-only index, never model caches.

    Construction validates local write access. Runtime logging failures are
    recorded best-effort and reported as ``False`` by ``finish``. Files contain
    raw bodies, not headers/cookies. ``saved`` means completed filesystem writes,
    not an fsync guarantee or proof that the client received the response.
    """

    def __init__(self, root: str | Path, *, model_context: dict | None = None):
        self.root = Path(root).expanduser().resolve()
        self.model_context = json.loads(json.dumps(model_context or {}, allow_nan=False))
        self._index_lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.NamedTemporaryFile(prefix=".qev-write-check-", dir=self.root):
            pass
        descriptor = os.open(self.root / "index.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        os.close(descriptor)

    @staticmethod
    def _write(path, body):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(body)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _append(self, summary):
        line = (json.dumps(summary, ensure_ascii=False, allow_nan=False) + "\n").encode()
        # Every request uses a distinct directory. Serialize each entire index
        # line across concurrent inference/IO threads sharing this logger.
        with self._index_lock, (self.root / "index.jsonl").open("ab") as stream:
            stream.write(line)

    @staticmethod
    def _failure(record, stage, exc):
        record.failed = True
        record.metadata.setdefault("log_errors", []).append({"stage": stage, "type": type(exc).__name__})
        _LOG.warning("Qev request log %s: %s failed (%s)",
                     record.metadata["request_id"], stage, type(exc).__name__)

    def begin(self, request_id, endpoint, body, started_at, *, request_complete=True):
        record = _Record(self.root / request_id, {
            "schema_version": 1, "request_id": request_id, "started_at": started_at,
            "endpoint": endpoint, "method": "POST", "state": "unfinished",
            "status_code": None, "elapsed_ms": None, "model_context": self.model_context,
            "request_complete": request_complete, "request_bytes": len(body),
            "request_sha256": _digest(body), "response_complete": False, "media": [],
        })
        try:
            record.directory.mkdir(mode=0o700)
            # Save replay bytes before decoding or extracting any attachments.
            self._write(record.directory / "request.json", body)
            self._write(record.directory / "metadata.json", _json_bytes(record.metadata))
            try:
                parsed = json.loads(body)
            except (ValueError, UnicodeError, RecursionError) as exc:
                record.metadata["request_parse_error"] = type(exc).__name__
            else:
                for pointer, value in _strings(parsed):
                    match = _IMAGE.fullmatch(value)
                    if not match:
                        continue
                    try:
                        content = base64.b64decode(match[2], validate=True)
                    except (ValueError, binascii.Error):
                        record.metadata.setdefault("media_errors", []).append({
                            "json_pointer": pointer, "error": "invalid_base64",
                        })
                        continue
                    extension = "jpg" if match[1] == "jpeg" else match[1]
                    relative = f"media/{len(record.metadata['media']):04d}.{extension}"
                    (record.directory / "media").mkdir(exist_ok=True, mode=0o700)
                    self._write(record.directory / relative, content)
                    record.metadata["media"].append({
                        "json_pointer": pointer, "path": relative, "mime": f"image/{match[1]}",
                        "bytes": len(content), "sha256": _digest(content),
                    })
            self._write(record.directory / "metadata.json", _json_bytes(record.metadata))
        except Exception as exc:  # noqa: BLE001 -- Logging cannot reject inference.
            self._failure(record, "begin", exc)
        return record

    def finish(self, record, response_body, status_code, elapsed_ms, *, state="completed",
               response_complete=True, exception_type=None):
        metadata = record.metadata
        metadata.update(finished_at=_utc(), status_code=status_code,
                        elapsed_ms=round(elapsed_ms, 3), state=state,
                        response_complete=response_complete)
        if exception_type:
            metadata["exception_type"] = exception_type
        try:
            if response_body is not None:
                self._write(record.directory / "response.json", response_body)
                metadata.update(response_bytes=len(response_body), response_sha256=_digest(response_body))
                try:
                    response = json.loads(response_body)
                except (ValueError, UnicodeError, RecursionError):
                    pass
                else:
                    if isinstance(response, dict) and isinstance(response.get("qev"), dict):
                        metadata["qev"] = response["qev"]
            metadata["log_status"] = "error" if record.failed else "saved"
            self._write(record.directory / "metadata.json", _json_bytes(metadata))
            self._append({key: metadata[key] for key in (
                "request_id", "started_at", "endpoint", "state", "status_code", "elapsed_ms", "log_status",
            )} | {"media_count": len(metadata["media"])})
        except Exception as exc:  # noqa: BLE001 -- Preserve the actual HTTP result.
            self._failure(record, "finish", exc)
            metadata["log_status"] = "error"
            try:
                self._write(record.directory / "metadata.json", _json_bytes(metadata))
            except Exception:  # noqa: BLE001, S110 -- Already warned; disk may still be unavailable.
                pass
        _LOG.info("Qev request %s %s status=%s log=%s", metadata["request_id"],
                  metadata["endpoint"], status_code, "error" if record.failed else "saved")
        return not record.failed


class RequestLogMiddleware:
    """Buffer non-streaming model responses until their local log is written.

    Install inside RequestBodyLimit. With FastAPI's ServerErrorMiddleware outside
    this middleware, an unhandled exception may have no captured HTTP response;
    record that fact and re-raise rather than inventing a 500 body or headers.
    """

    def __init__(self, app, *, request_log: RequestLog):
        self.app, self.request_log = app, request_log

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in _ENDPOINTS:
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        started_at = _utc()
        request_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex
        incoming, body = [], []
        disconnected, request_complete = False, False
        # BodyLimit has already bounded these bytes. Replay the observed messages
        # unchanged, including a disconnect; do not poll receive concurrently.
        while True:
            message = await receive()
            incoming.append(message)
            if message["type"] == "http.disconnect":
                disconnected = True
                break
            body.append(message.get("body", b""))
            if not message.get("more_body", False):
                request_complete = True
                break
        record = await run_in_threadpool(self.request_log.begin, request_id, scope["path"],
                                         b"".join(body), started_at, request_complete=request_complete)
        pending = iter(incoming)
        response_start, response_chunks = None, []
        finalized = False

        async def replay():
            nonlocal disconnected
            message = next(pending, None)
            if message is None:
                message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def capture(message):
            nonlocal response_start, finalized
            if message["type"] == "http.response.start":
                response_start = dict(message)
            elif message["type"] == "http.response.body":
                response_chunks.append(dict(message))
                if not message.get("more_body", False):
                    if response_start is None:
                        raise RuntimeError("HTTP response body arrived before response start")
                    saved = await run_in_threadpool(
                        self.request_log.finish, record,
                        b"".join(chunk.get("body", b"") for chunk in response_chunks),
                        response_start["status"], (time.perf_counter() - started) * 1000,
                    )
                    finalized = True
                    headers = [(key, value) for key, value in response_start.get("headers", [])
                               if key.lower() not in {b"x-qev-request-id", b"x-qev-log-status"}]
                    response_start["headers"] = [*headers, (b"x-qev-request-id", request_id.encode()),
                                                 (b"x-qev-log-status", b"saved" if saved else b"error")]
                    await send(response_start)
                    for chunk in response_chunks:
                        await send(chunk)
            else:
                await send(message)

        try:
            await self.app(scope, replay, capture)
        except BaseException as exc:
            if not finalized:
                state = ("cancelled" if isinstance(exc, asyncio.CancelledError) else
                         "disconnected" if disconnected or isinstance(exc, ClientDisconnect) else "exception")
                await run_in_threadpool(
                    self.request_log.finish, record,
                    b"".join(chunk.get("body", b"") for chunk in response_chunks) if response_chunks else None,
                    response_start["status"] if response_start else None,
                    (time.perf_counter() - started) * 1000,
                    state=state, response_complete=False, exception_type=type(exc).__name__,
                )
            raise
        if not finalized:
            await run_in_threadpool(self.request_log.finish, record, None,
                                    response_start["status"] if response_start else None,
                                    (time.perf_counter() - started) * 1000,
                                    state="disconnected" if disconnected else "incomplete",
                                    response_complete=False)
