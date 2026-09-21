"""Replay and failure checks using in-memory ASGI apps, without a model."""

import asyncio
import base64
import hashlib
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import ClientDisconnect
from starlette.responses import Response

from qev.request_log import RequestLog, RequestLogMiddleware
from qev.serve import RequestBodyLimit


def folders(root):
    return sorted(path for path in root.iterdir() if path.is_dir())


def metadata(directory):
    return json.loads((directory / "metadata.json").read_bytes())


async def read_body(receive):
    body = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise ClientDisconnect()
        body.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(body)


def image_url(format):
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), "red").save(stream, format)
    data = stream.getvalue()
    mime = "image/" + format.lower()
    return "data:" + mime + ";base64," + base64.b64encode(data).decode(), data


def test_original_bodies_media_hashes_context_and_unfinished_record(tmp_path):
    pictures = [image_url(format) for format in ["PNG", "JPEG", "WEBP"]]
    body = json.dumps({
        "state": {"content": [{"type": "image_url", "image_url": {"url": pictures[0][0]}}]},
        "questions": {"q": {"criteria": {"A/~": {"content": [
            {"type": "image_url", "image_url": {"url": pictures[1][0]}},
            {"type": "image_url", "image_url": {"url": pictures[2][0]}},
        ]}}}}, "remote": "https://example.invalid/image.png", "local": "file:///private/image.png",
    }, ensure_ascii=False, indent=3).encode()
    original_response = b'{ "answers": {"q": "A/~"}, "qev": {"timings_ms": {"inference": 12.5}} }'
    context = {"checkpoint": "/models/test", "backend": "stub", "config": {"temperature": 1.2}}
    log = RequestLog(tmp_path, model_context=context)
    context["config"]["temperature"] = 9  # The startup metadata is a snapshot.

    async def app(scope, receive, send):
        assert await read_body(receive) == body
        directory, = folders(tmp_path)
        assert (directory / "request.json").read_bytes() == body
        assert metadata(directory)["state"] == "unfinished"
        assert not (directory / "response.json").exists()
        await Response(original_response, media_type="application/json", headers={
            "X-Original": "kept", "X-Qev-Request-Id": "old", "X-Qev-Log-Status": "old",
        })(scope, receive, send)

    response = TestClient(RequestLogMiddleware(app, request_log=log)).post(
        "/v1/systemone", content=body,
        headers={"Authorization": "Bearer secret-test-token", "Cookie": "session=private-cookie"},
    )
    assert response.content == original_response and response.headers["X-Original"] == "kept"
    assert response.headers["X-Qev-Log-Status"] == "saved"
    directory, = folders(tmp_path)
    assert directory.name == response.headers["X-Qev-Request-Id"]
    assert (directory / "request.json").read_bytes() == body
    assert (directory / "response.json").read_bytes() == original_response
    meta = metadata(directory)
    assert meta["state"] == "completed" and meta["status_code"] == 200
    assert meta["request_complete"] and meta["response_complete"]
    assert meta["model_context"]["config"]["temperature"] == 1.2
    assert meta["qev"]["timings_ms"]["inference"] == 12.5
    assert meta["started_at"].endswith("Z") and meta["finished_at"].endswith("Z")
    assert meta["request_sha256"] == hashlib.sha256(body).hexdigest()
    assert meta["response_sha256"] == hashlib.sha256(original_response).hexdigest()
    assert [item["json_pointer"] for item in meta["media"]] == [
        "/state/content/0/image_url/url",
        "/questions/q/criteria/A~1~0/content/0/image_url/url",
        "/questions/q/criteria/A~1~0/content/1/image_url/url",
    ]
    for item, (url, original) in zip(meta["media"], pictures, strict=True):
        assert (directory / item["path"]).read_bytes() == original
        assert item["sha256"] == hashlib.sha256(original).hexdigest()
        assert item["bytes"] == len(original) and item["mime"] in url
    saved = "\n".join(path.read_text() for path in directory.glob("*.json"))
    assert "secret-test-token" not in saved and "private-cookie" not in saved
    index = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert len(index) == 1 and index[0]["request_id"] == directory.name
    assert index[0]["media_count"] == 3 and "data:image" not in json.dumps(index)


@pytest.mark.parametrize("body,status", [(b'{"model":"unknown"}', 404), (b'{"unfinished":', 422)])
def test_handled_errors_and_invalid_json_are_logged(tmp_path, body, status):
    app = FastAPI()

    @app.post("/v1/systemone")
    async def fail(body: dict):
        raise HTTPException(404, "Unknown model alias")

    app.add_middleware(RequestLogMiddleware, request_log=RequestLog(tmp_path))
    app.add_middleware(RequestBodyLimit)
    response = TestClient(app).post("/v1/systemone", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == status and response.headers["X-Qev-Log-Status"] == "saved"
    directory, = folders(tmp_path)
    assert (directory / "request.json").read_bytes() == body
    assert (directory / "response.json").read_bytes() == response.content
    assert metadata(directory)["status_code"] == status
    if status == 422:
        assert metadata(directory)["request_parse_error"] == "JSONDecodeError"


def test_unhandled_exception_is_not_swallowed_or_given_an_invented_response(tmp_path):
    app = FastAPI()

    @app.post("/v1/systemone")
    async def fail():
        raise RuntimeError("original failure")

    app.add_middleware(RequestLogMiddleware, request_log=RequestLog(tmp_path))
    with pytest.raises(RuntimeError, match="original failure"):
        TestClient(app).post("/v1/systemone", content=b'{"input":1}')
    directory, = folders(tmp_path)
    meta = metadata(directory)
    assert meta["state"] == "exception" and meta["exception_type"] == "RuntimeError"
    assert meta["status_code"] is None and not meta["response_complete"]
    assert (directory / "request.json").read_bytes() == b'{"input":1}'
    assert not (directory / "response.json").exists()
    # The existing outer error handler still supplies its ordinary 500 response.
    response = TestClient(app, raise_server_exceptions=False).post("/v1/systemone", json={})
    assert response.status_code == 500 and response.text == "Internal Server Error"


def test_concurrent_requests_keep_bodies_separate_and_do_not_cache_answers(tmp_path):
    log = RequestLog(tmp_path)

    async def app(scope, receive, send):
        body = await read_body(receive)
        await asyncio.sleep(0)
        await Response(body, media_type="application/json")(scope, receive, send)

    wrapped = RequestLogMiddleware(app, request_log=log)

    def request(index):
        body = json.dumps({"sequence": index}).encode()
        response = TestClient(wrapped).post("/v1/chat/completions", content=body)
        assert response.content == body and response.headers["X-Qev-Log-Status"] == "saved"
        return response.headers["X-Qev-Request-Id"]

    with ThreadPoolExecutor(max_workers=8) as clients:
        ids = list(clients.map(request, range(24)))
    assert len(set(ids)) == 24 and len(folders(tmp_path)) == 24
    index = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert len(index) == 24 and {item["request_id"] for item in index} == set(ids)
    for number, request_id in enumerate(ids):
        directory = tmp_path / request_id
        assert json.loads((directory / "request.json").read_bytes()) == {"sequence": number}
        assert (directory / "request.json").read_bytes() == (directory / "response.json").read_bytes()
    # Replaying an identical body still executes and creates another record.
    assert request(0) not in ids and len(folders(tmp_path)) == 25


@pytest.mark.parametrize("stage", ["request", "response", "index"])
def test_disk_errors_do_not_change_the_actual_response(tmp_path, monkeypatch, stage):
    log = RequestLog(tmp_path)
    original = log._write

    def write(path, body):
        if path.name == stage + ".json":
            raise OSError("injected disk failure")
        return original(path, body)

    monkeypatch.setattr(log, "_write", write)
    if stage == "index":
        def fail_index(summary):
            raise OSError("injected index failure")
        monkeypatch.setattr(log, "_append", fail_index)

    async def app(scope, receive, send):
        await read_body(receive)
        await Response(b'{"actual":"answer"}', media_type="application/json")(scope, receive, send)

    response = TestClient(RequestLogMiddleware(app, request_log=log)).post("/v1/systemone", json={})
    assert response.status_code == 200 and response.json() == {"actual": "answer"}
    assert response.headers["X-Qev-Log-Status"] == "error"
    directory, = folders(tmp_path)
    assert metadata(directory)["log_status"] == "error"


@pytest.mark.parametrize("method,path", [("GET", "/v1/systemone"), ("GET", "/assets/app.js"),
                                        ("GET", "/health"), ("POST", "/api/snake/games")])
def test_other_routes_are_not_logged(tmp_path, method, path):
    async def app(scope, receive, send):
        await Response(b"ok")(scope, receive, send)

    response = TestClient(RequestLogMiddleware(app, request_log=RequestLog(tmp_path))).request(method, path)
    assert response.content == b"ok" and "X-Qev-Request-Id" not in response.headers
    assert not folders(tmp_path) and not (tmp_path / "index.jsonl").read_bytes()


def test_outer_body_limit_rejects_before_logging(tmp_path):
    async def app(*args):
        raise AssertionError("Oversized input must not reach the app")

    wrapped = RequestBodyLimit(RequestLogMiddleware(app, request_log=RequestLog(tmp_path)), max_bytes=8)
    response = TestClient(wrapped).post("/v1/systemone", content=b"x" * 9)
    assert response.status_code == 413 and not folders(tmp_path)
    assert "X-Qev-Log-Status" not in response.headers


def test_startup_rejects_a_non_directory_log_root(tmp_path):
    destination = tmp_path / "not-a-directory"
    destination.write_text("existing file")
    with pytest.raises(OSError):
        RequestLog(destination)


@pytest.mark.parametrize("kind", ["cancelled", "disconnected"])
def test_cancel_and_disconnect_retain_partial_log_and_propagate(tmp_path, kind):
    signal = asyncio.CancelledError() if kind == "cancelled" else ClientDisconnect()

    async def app(scope, receive, send):
        if kind == "disconnected":
            await read_body(receive)
        raise signal

    messages = iter([{"type": "http.request", "body": b"{}", "more_body": kind == "disconnected"},
                     {"type": "http.disconnect"}])

    async def receive():
        return next(messages)

    async def send(message):
        raise AssertionError("Cancelled/disconnected calls must not invent a response")

    wrapped = RequestLogMiddleware(app, request_log=RequestLog(tmp_path))
    with pytest.raises(type(signal)):
        asyncio.run(wrapped({"type": "http", "method": "POST", "path": "/v1/systemone"}, receive, send))
    directory, = folders(tmp_path)
    assert metadata(directory)["state"] == kind
    assert not metadata(directory)["response_complete"]
    assert not (directory / "response.json").exists()


def test_body_writes_run_off_the_event_loop_and_finish_before_response_start(tmp_path, monkeypatch):
    log = RequestLog(tmp_path)
    event_thread = threading.get_ident()
    original = log._write
    write_threads = []

    def write(path, body):
        write_threads.append(threading.get_ident())
        original(path, body)

    monkeypatch.setattr(log, "_write", write)

    async def app(scope, receive, send):
        assert threading.get_ident() == event_thread
        await read_body(receive)
        await Response(b"{}")(scope, receive, send)

    async def receive():
        return {"type": "http.request", "body": b"{}"}

    async def send(message):
        if message["type"] == "http.response.start":
            directory, = folders(tmp_path)
            assert metadata(directory)["state"] == "completed"
            assert (directory / "response.json").read_bytes() == b"{}"
            assert (tmp_path / "index.jsonl").read_bytes()

    asyncio.run(RequestLogMiddleware(app, request_log=log)(
        {"type": "http", "method": "POST", "path": "/v1/systemone"}, receive, send))
    assert write_threads and event_thread not in write_threads
