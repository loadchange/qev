"""Talk to a running ``qev serve`` so CLI calls skip model loading.

Only the standard library is imported: a warm server answers a CLI decision in
well under a second, while loading the model in-process takes several seconds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_SERVER = "http://127.0.0.1:8008"


class ServerError(RuntimeError):
    """The server answered with an HTTP error or could not be reached."""


def server_url(explicit: str | None = None) -> str:
    return (explicit or os.environ.get("QEV_SERVER") or DEFAULT_SERVER).rstrip("/")


def health(url: str, timeout: float = 0.5) -> dict | None:
    """The server's /health document, or None when nothing usable listens."""
    try:
        with urlopen(Request(f"{url}/health"), timeout=timeout) as response:
            document = json.load(response)
    except (OSError, URLError, ValueError):
        return None
    return document if isinstance(document, dict) and document.get("status") == "ok" else None


def serves(document: dict, model_path: Path | None) -> bool:
    """Whether a healthy server runs the requested checkpoint (any, if unspecified)."""
    if model_path is None:
        return True
    checkpoint = document.get("checkpoint")
    return bool(checkpoint) and Path(checkpoint).resolve() == Path(model_path).resolve()


def post(url: str, path: str, body: dict, timeout: float = 120) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode()
    request = Request(f"{url}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        try:
            detail = json.load(error).get("detail", error.reason)
        except (ValueError, AttributeError):
            detail = error.reason
        raise ServerError(f"{url}{path} returned HTTP {error.code}: {detail}") from None
    except (OSError, URLError) as error:
        raise ServerError(f"Cannot reach {url}: {error}") from None
