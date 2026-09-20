import asyncio

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
