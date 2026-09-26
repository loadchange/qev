"""Model registry, request building, rendering and the decide/chat/pull commands."""
import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from qev import cli, client, models
from qev.decide import build_question, build_state, media_urls, render_quiet, render_text

CHOICE = {"model": "qev-0.8b", "answers": {"answer": {
    "type": "choice", "choice": "billing", "confidence": 0.9,
    "probabilities": {"billing": 0.94, "technical": 0.04, "account": 0.02}}},
    "qev": {"backend": "mlx"}}


@pytest.fixture
def tiny_registry(tmp_path, monkeypatch):
    """A registry entry whose manifest describes three small files."""
    files = {"qev_config.json": b'{"format": "qev"}', "pointer.safetensors": b"pointer",
             "backbone/model.safetensors": b"weights", "reports/evaluation.json": b"{}"}
    source = tmp_path / "download"
    for name, data in files.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(data)
    manifest = {"files": {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in files.items()}}
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/tiny.json").write_text(json.dumps(manifest))
    for real in models._MANIFESTS.glob("*.json"):
        (tmp_path / "manifests" / real.name).write_text(real.read_text())
    monkeypatch.setattr(models, "_MANIFESTS", tmp_path / "manifests")
    spec = models.ModelSpec("tiny", "org/tiny", "0" * 40, "mlx", "test model")
    monkeypatch.setitem(models.REGISTRY, "tiny", spec)
    return source, spec


def test_registry_names_aliases_and_paths(tmp_path):
    assert models.canonical("qev-latest") == "qev-450m" and models.canonical("nope") is None
    assert models.canonical("qev-230m-mlx") == "qev-230m" and models.canonical("qev:0.8b") == "qev-0.8b"
    with pytest.raises(models.ModelNotInstalled, match="qev pull qev-450m"):
        models.resolve()
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "qev_config.json").write_text("{}")
    assert models.resolve(str(checkpoint)) == checkpoint.resolve()
    with pytest.raises(ValueError, match="Unknown model"):
        models.resolve(str(tmp_path / "missing"))
    manifest = models.REGISTRY["qev-0.8b"].manifest
    runtime = models.runtime_files(manifest)
    assert "backbone/model.safetensors" in runtime and "backbone/LICENSE" in runtime
    assert not any(name.startswith(("reports/", "README")) for name in runtime)


@pytest.mark.parametrize("link", [False, True])
def test_pull_from_directory_verifies_and_installs(tiny_registry, link):
    source, spec = tiny_registry
    path = models.pull("tiny", source=str(source), link=link, log=lambda _: None)
    assert models.is_installed(path) and models.resolve("tiny") == path
    assert (path / "backbone/model.safetensors").is_symlink() == link
    assert not (path / "reports").exists()  # optional files are not required
    marker = json.loads((path / models.MARKER).read_text())
    assert marker["source"] == str(source.resolve()) and marker["verified_files"] == 3
    row = next(row for row in models.list_models() if row["name"] == "tiny")
    assert row["installed"] and row["size_bytes"] == sum(e["size"] for e in spec.manifest["files"].values())
    assert models.remove("tiny") == path and not path.exists()


def test_pull_rejects_modified_files(tiny_registry):
    source, _ = tiny_registry
    (source / "pointer.safetensors").write_bytes(b"tampere")  # same size, different bytes
    with pytest.raises(ValueError, match="sha256 mismatch"):
        models.pull("tiny", source=str(source), log=lambda _: None)
    assert not models.installed_path("tiny").exists()


def test_pull_downloads_pinned_revision(tiny_registry, monkeypatch):
    source, spec = tiny_registry
    calls = []

    def snapshot_download(repo_id, revision, local_dir):
        calls.append((repo_id, revision))
        import shutil
        shutil.copytree(source, local_dir, dirs_exist_ok=True)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    path = models.pull("tiny", log=lambda _: None)
    assert calls == [(spec.repo_id, spec.revision)] and models.is_installed(path)
    assert models.pull("tiny", log=lambda _: None) == path and len(calls) == 1  # already installed


def test_questions_and_text_state():
    assert build_question("choice", ["billing=Invoices and refunds", "technical"], instructions="Team?") == {
        "type": "choice", "instructions": "Team?", "criteria": {"billing": "Invoices and refunds", "technical": None}}
    with pytest.raises(ValueError, match="unique"):
        build_question("choice", ["a", "a"])
    assert build_question("noul", true="refund requested") == {
        "type": "noul", "instructions": None, "criteria": {"true": "refund requested"}}
    with pytest.raises(ValueError, match="two ordered levels"):
        build_question("score", ["only"])
    with pytest.raises(ValueError, match="question type"):
        build_question(None)
    assert build_state("plain") == "plain"


def picture(path, size, fmt="PNG", color="red", orientation=None):
    image = Image.new("RGB", size, color)
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation
    image.save(path, fmt, **({"exif": exif} if orientation else {}))
    return path


def decode(url):
    header, data = url.split(",", 1)
    return header, Image.open(io.BytesIO(base64.b64decode(data)))


def test_media_is_kept_converted_rotated_and_downscaled(tmp_path):
    small = picture(tmp_path / "small.png", (40, 20))
    rotated = picture(tmp_path / "rotated.jpg", (40, 20), "JPEG", orientation=6)
    huge = picture(tmp_path / "huge.bmp", (3000, 2000), "BMP")
    notes = []
    urls = media_urls([small, rotated, huge], notes=notes)
    header, image = decode(urls[0])
    assert header == "data:image/png;base64" and image.size == (40, 20)  # original bytes
    assert decode(urls[1])[1].size == (20, 40)  # EXIF orientation applied
    header, image = decode(urls[2])
    assert header == "data:image/jpeg;base64" and image.width * image.height <= 8_000_000 // 3
    assert len(notes) == 2
    frames = media_urls([small, picture(tmp_path / "other.png", (80, 80), color="blue")], frames=True)
    assert {decode(url)[1].size for url in frames} == {(40, 20)}
    state = build_state("look", images=[small], frames=[small, small], fps=2)
    assert [item["type"] for item in state["content"]] == ["text", "image_url", "video"]
    assert state["content"][2]["fps"] == 2
    with pytest.raises(ValueError, match="At most 8"):
        build_state("x", images=[small] * 9)


def test_rendering():
    text = render_text(CHOICE, footer="qev-0.8b · mlx")
    assert text.splitlines()[0].startswith("answer: billing") and "0.940" in text and text.endswith("qev-0.8b · mlx")
    assert render_quiet(CHOICE) == "billing"
    noul = {"answers": {"a": {"type": "noul", "noul": 0.2}, "b": {"type": "noul", "noul": 0.7}}}
    assert render_quiet(noul) == "a=no\nb=yes"
    score = {"answers": {"u": {"type": "score", "score": 1.25, "confidence": 0.6, "legend": {"0": "low", "1": "mid", "2": "high"},
                               "probabilities": {"0": 0.1, "1": 0.55, "2": 0.35}}}}
    assert "1 mid" in render_text(score) and render_quiet(score) == "1.250"


def test_decide_uses_a_matching_server(monkeypatch, capsys):
    posts = []
    monkeypatch.setattr(client, "health", lambda url, timeout=0.5: {"status": "ok", "checkpoint": "/models/a"})
    monkeypatch.setattr(client, "post", lambda url, path, body, timeout=0: posts.append((url, path, body)) or CHOICE)
    monkeypatch.setitem(__import__("sys").modules, "qev.inference", None)  # must not load a model
    code = cli.main(["decide", "Card declined twice", "-i", "Which team?", "--choice", "billing", "technical", "account", "-q"])
    assert code == 0 and capsys.readouterr().out.strip() == "billing"
    url, path, body = posts[0]
    assert url == "http://127.0.0.1:9" and path == "/v1/systemone"
    assert body["questions"]["answer"]["criteria"] == {"billing": None, "technical": None, "account": None}


def test_decide_runs_locally_when_the_server_has_another_model(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "qev_config.json").write_text("{}")
    monkeypatch.setattr(client, "health", lambda url, timeout=0.5: {"status": "ok", "checkpoint": "/models/other"})
    monkeypatch.setattr(client, "post", lambda *a, **k: pytest.fail("used the wrong server"))
    requests = []

    class Agent:
        backend = "mlx"

        def __init__(self, path, **_):
            assert Path(path) == checkpoint

        def predict(self, state, questions, model=None):
            requests.append((state, questions))
            return {"model": "qev-0.8b", "answers": {"answer": {"type": "noul", "noul": 0.1}}, "qev": {"backend": "mlx"}}

    monkeypatch.setitem(__import__("sys").modules, "qev.inference", SimpleNamespace(Agent=Agent))
    code = cli.main(["decide", "No refund mentioned", "--noul", "--model", str(checkpoint), "--exit-status", "-q"])
    assert code == 1 and capsys.readouterr().out.strip() == "no"
    assert requests[0][1]["answer"]["type"] == "noul"


def test_decide_request_file_stdin_and_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(client, "health", lambda url, timeout=0.5: {"status": "ok"})
    monkeypatch.setattr(client, "post", lambda url, path, body, timeout=0: CHOICE)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"state": "s", "questions": {"answer": {"type": "choice", "criteria": {"billing": None}}}}))
    assert cli.main(["decide", "--request", str(request), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == CHOICE
    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin"))
    captured = []
    monkeypatch.setattr(client, "post", lambda url, path, body, timeout=0: captured.append(body) or CHOICE)
    assert cli.main(["decide", "-", "--choice", "billing", "-q"]) == 0 and captured[0]["state"] == "from stdin"
    assert cli.main(["decide", "text"]) == 2  # no question type
    assert "question type" in capsys.readouterr().err
    assert cli.main(["decide", "text", "--choice", "a", "--local", "--no-pull"]) == 3  # default model missing


def test_explicit_server_must_answer(monkeypatch, capsys):
    monkeypatch.setattr(client, "health", lambda url, timeout=0.5: None)
    assert cli.main(["decide", "x", "--noul", "--server", "http://127.0.0.1:9"]) == 4
    assert "No qev server" in capsys.readouterr().err


def test_chat_and_list(monkeypatch, capsys):
    monkeypatch.setattr(client, "health", lambda url, timeout=0.5: {"status": "ok"})
    sent = []
    reply = {"choices": [{"message": {"content": "Hello!\n"}}]}
    monkeypatch.setattr(client, "post", lambda url, path, body, timeout=0: sent.append((path, body)) or reply)
    assert cli.main(["chat", "Say hi", "--max-tokens", "5"]) == 0
    assert capsys.readouterr().out == "Hello!\n"
    assert sent[0][0] == "/v1/chat/completions" and sent[0][1]["messages"] == [{"role": "user", "content": "Say hi"}]
    assert cli.main(["list"]) == 0
    assert "qev-450m *" in capsys.readouterr().out


def test_health_reports_the_checkpoint():
    from fastapi.testclient import TestClient

    from qev.serve import create_app

    agent = SimpleNamespace(backend="mlx", config={"model_name": "qev-0.8b"}, path=Path("/models/qev"))
    document = TestClient(create_app(agent)).get("/health").json()
    assert document["checkpoint"] == "/models/qev" and document["model"] == "qev-0.8b" and "version" in document
    assert client.serves(document, Path("/models/qev")) and not client.serves(document, Path("/models/other"))
    assert client.serves(document, None)


def test_processor_backend_selection(monkeypatch):
    from qev.processing import processor_backend

    monkeypatch.setenv("QEV_PROCESSOR", "numpy")
    assert processor_backend() == "numpy" and processor_backend("hf") == "hf"
    monkeypatch.delenv("QEV_PROCESSOR")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert processor_backend() == "numpy"
    with pytest.raises(ValueError):
        processor_backend("gpu")


def test_doctor_json(capsys):
    assert cli.main(["doctor", "--json"]) in (0, 1)
    report = json.loads(capsys.readouterr().out)
    names = {check["name"] for check in report["checks"]}
    assert {"platform", "processor", "QEV_HOME", "model qev-450m", "server"} <= names
