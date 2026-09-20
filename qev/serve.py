"""Local typed decisions and native Qwen multimodal generation.

Images are inline data URLs, and videos are explicitly sampled inline frames.
This service never opens client-provided paths or fetches client-provided URLs.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from . import __version__
from .api import ChatCompletionRequest, SystemOneRequest


class RequestBodyLimit:
    """Bound HTTP bytes before JSON parsing, including chunked request bodies."""
    def __init__(self, app, max_bytes=12 * 1024 * 1024):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        if b"content-length" in headers:
            try:
                declared = int(headers[b"content-length"])
            except ValueError:
                return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
            if declared < 0 or declared > self.max_bytes:
                return await JSONResponse({"detail": "Request body exceeds the byte limit"}, status_code=413)(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_bytes:
                return await JSONResponse({"detail": "Request body exceeds the byte limit"}, status_code=413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body, sent = b"".join(chunks), False
        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


def create_app(agent):
    app = FastAPI(title="Qev", version=__version__)
    app.add_middleware(RequestBodyLimit)
    from .demo import create_demo_router

    app.include_router(create_demo_router(agent))
    web = Path(__file__).with_name("web")
    app.mount("/assets", StaticFiles(directory=web), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/snake", include_in_schema=False)
    @app.get("/playground", include_in_schema=False)
    def playground():
        return FileResponse(web / "index.html", headers={"Cache-Control": "no-cache"})

    def known_model(name):
        return name in {"qev-latest", "jev-latest", "qev-0.8b", "qev-0.8b-mlx", "qev:0.8b", "qev:0.8b-mlx", "qev-native",
                        agent.config.get("model_name"), agent.config.get("base_model"), agent.config.get("base")}

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": agent.backend}

    @app.get("/v1/models")
    def models():
        name = agent.config.get("model_name", "qev-0.8b")
        return {"models": [{"name": name, "description": "Qwen3.5 typed decisions; multimodal decision accuracy is not yet validated", "release_date": "2026-09-20"},
                           {"name": "qev-latest", "description": f"Alias of {name}", "release_date": "2026-09-20"},
                           {"name": "qev:0.8b", "description": f"Alias of loaded checkpoint {name}", "release_date": "2026-09-20"},
                           {"name": "qev:0.8b-mlx", "description": f"Alias of loaded checkpoint {name}; aliases do not switch the runtime", "release_date": "2026-09-20"},
                           {"name": "qev-native", "description": "Native Qwen text/image/video generation with the decision adapter disabled", "release_date": "2026-09-20"}]}

    @app.post("/v1/systemone")
    def system_one(request: SystemOneRequest):
        if not known_model(request.model):
            raise HTTPException(404, "Unknown model alias")
        if len(request.questions) > 64:
            raise HTTPException(422, "At most 64 questions per request")
        try:
            return agent.predict(request.state, {k: v.model_dump() for k, v in request.questions.items()}, model=request.model)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(501, str(exc)) from exc

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        if not known_model(request.model):
            raise HTTPException(404, "Unknown model alias")
        if not callable(getattr(agent, "chat_completions", None)):
            raise HTTPException(501, "This checkpoint/backend does not provide native multimodal generation")
        try:
            return agent.chat_completions(request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(501, "This checkpoint/backend does not provide native multimodal generation") from exc

    return app
