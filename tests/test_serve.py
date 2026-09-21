import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from qev.api import decode_state, to_answers, to_record
from qev.serve import RequestBodyLimit, create_app


class StubAgent:
    backend = "stub"

    def __init__(self):
        self.config = {"model_name": "qev-0.8b"}

    def predict(self, state, questions, model=None):
        decode_state(state)
        record, meta = to_record({"state": state, "questions": questions})
        ps = [[1 / len(q["options"])] * len(q["options"]) for q in record["questions"]]
        return {"model": model, "answers": to_answers(ps, meta), "usage": {"input_tokens": 3, "output_tokens": 0}}

    def chat_completions(self, request):
        return {"id": "chatcmpl-test", "object": "chat.completion", "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "native response"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "qev": {"decision_adapter_enabled": False}}


@pytest.mark.parametrize("alias", ["jev-latest", "qev-latest", "qev:0.8b", "qev:0.8b-mlx"])
def test_backward_compatible_typed_aliases_and_optional_instructions(alias):
    client = TestClient(create_app(StubAgent()))
    response = client.post("/v1/systemone", json={"model": alias, "state": "hello", "questions": {"answer": {"type": "noul"}}})
    assert response.status_code == 200
    assert response.json()["answers"]["answer"]["noul"] == 0.5


def test_server_native_endpoint_and_error_contract():
    client = TestClient(create_app(StubAgent()))
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 200
    assert response.json()["qev"]["decision_adapter_enabled"] is False
    assert client.post("/v1/chat/completions", json={"model": "unknown", "messages": [{"role": "user", "content": "hi"}]}).status_code == 404
    assert client.post("/v1/systemone", content="{}", headers={"Content-Length": str(13 * 1024 * 1024)}).status_code == 413
    assert client.post("/v1/systemone", json={"state": [{"type": "image_url", "image_url": {"url": "/private/image.png"}}], "questions": {"q": {"type": "noul"}}}).status_code == 422


def test_embedded_pages_and_assets_do_not_shadow_existing_api():
    client = TestClient(create_app(StubAgent()))
    for route in ("/", "/snake", "/playground"):
        response = client.get(route)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert '/assets/app.js' in response.text
    for name in ("app.js", "snake.js", "playground.js", "choice-form.js", "style.css", "icon.svg"):
        response = client.get(f"/assets/{name}")
        assert response.status_code == 200, name
        assert response.content
    assert client.get("/assets/does-not-exist.js").status_code == 404
    assert client.get("/health").json()["backend"] == "stub"
    assert client.get("/docs").status_code == 200
    assert "/v1/systemone" in client.get("/openapi.json").json()["paths"]


def test_chunked_request_body_is_bounded_before_app_execution():
    called, sent = [], []
    async def app(*args):
        called.append(True)
    messages = iter([{"type": "http.request", "body": b"1234", "more_body": True}, {"type": "http.request", "body": b"5678", "more_body": False}])
    async def receive():
        return next(messages)
    async def send(message):
        sent.append(message)
    asyncio.run(RequestBodyLimit(app, max_bytes=5)({"type": "http", "method": "POST", "headers": []}, receive, send))
    assert not called
    assert sent[0]["status"] == 413


def test_official_typesafe_sdk_serialization_and_response_decoding():
    sdk = pytest.importorskip("typesafe_sdk")
    httpx2 = pytest.importorskip("httpx2")
    client = TestClient(create_app(StubAgent()))
    def handler(request):
        response = client.request(request.method, request.url.path, content=request.content, headers={"Content-Type": "application/json"})
        return httpx2.Response(response.status_code, json=response.json())
    with sdk.TypeSafeClient(api_key="local", base_url="http://qev.local", model="jev-latest", transport=httpx2.MockTransport(handler)) as official:
        assert official.models.list().models[0].name == "qev-0.8b"
        result = official.system_one(state={"text": "hello"}, questions={
            "binary": sdk.Noul(), "department": sdk.Choice(criteria={"a": None, "b": None}),
            "priority": sdk.Score(criteria=["low", "high"]),
        })
        assert result.nouls["binary"].noul == 0.5
        assert result.choices["department"].choice == "a"
        assert result.scores["priority"].legend == {0: "low", 1: "high"}
        assert result.usage.output_tokens == 0


@pytest.mark.parametrize("directory", [None, "custom-journal"])
def test_cli_records_requests_in_default_or_selected_directory(tmp_path, monkeypatch, directory):
    from qev.cli import main

    monkeypatch.chdir(tmp_path)
    agent = SimpleNamespace(backend="torch", config={"model_name": "qev-0.8b"},
                            predict=lambda *args, **kwargs: {"answers": {}, "qev": {"backend": "torch"}})
    monkeypatch.setattr("qev.inference.Agent", lambda *args, **kwargs: agent)
    applications = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: applications.append(app))
    arguments = ["serve", "--model", "checkpoint", "--no-warmup", "--no-wired-memory"]
    if directory:
        arguments.extend(["--request-log-dir", directory])
    main(arguments)
    with TestClient(applications[0]) as client:
        request = {"state": "record this request", "questions": {"q": {"type": "noul"}}}
        response = client.post("/v1/systemone", json=request)
    assert response.status_code == 200
    assert response.headers["x-qev-log-status"] == "saved"
    folder = tmp_path / (directory or "runs/request-logs") / response.headers["x-qev-request-id"]
    assert json.loads((folder / "request.json").read_text()) == request
    assert json.loads((folder / "response.json").read_text()) == response.json()
    context = json.loads((folder / "metadata.json").read_text())["model_context"]
    assert context["checkpoint"] == str(tmp_path / "checkpoint")
    assert context["backend"] == "torch"
    assert context["config"] == agent.config


def test_programmatic_service_does_not_record_without_explicit_logger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app(StubAgent())) as client:
        response = client.post("/v1/systemone", json={"state": "hello", "questions": {"q": {"type": "noul"}}})
    assert response.status_code == 200
    assert "x-qev-request-id" not in response.headers
    assert not (tmp_path / "runs").exists()
