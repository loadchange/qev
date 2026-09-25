"""Named Qev checkpoints under ``QEV_HOME``: resolve, download, import, verify.

A name maps to a Hugging Face repository at a pinned revision. Every runtime
file is checked against the release manifest shipped inside this package, so a
download, a partial copy or an imported directory is only used once its sizes
and SHA-256 digests match what was published.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

MARKER = ".qev-model.json"
_MANIFESTS = Path(__file__).with_name("manifests")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    revision: str
    runtime: str
    description: str

    @property
    def manifest(self) -> dict:
        return json.loads((_MANIFESTS / f"{self.name}.json").read_text())

    @property
    def size_bytes(self) -> int:
        return sum(entry["size"] for entry in self.manifest["files"].values())


REGISTRY = {
    "qev-0.8b": ModelSpec(
        "qev-0.8b", "twainsk/qev-0.8b-mlx", "0dcf82a746794e813357d19c8d94812b1e686169", "mlx",
        "Qev 0.8B v0.3.0: Qwen3.5 multimodal foundation, decision adapter and pointer head (MLX float32)"),
}
ALIASES = {"qev-latest": "qev-0.8b", "qev:0.8b": "qev-0.8b", "qev-0.8b-mlx": "qev-0.8b"}
DEFAULT_MODEL = "qev-0.8b"


class ModelNotInstalled(LookupError):
    def __init__(self, name):
        super().__init__(f"Model {name!r} is not installed. Run: qev pull {name}")
        self.name = name


def home() -> Path:
    return Path(os.environ.get("QEV_HOME", "~/.qev")).expanduser()


def models_root() -> Path:
    return home() / "models"


def canonical(name: str) -> str | None:
    name = ALIASES.get(name, name)
    return name if name in REGISTRY else None


def runtime_files(manifest: dict) -> dict:
    """Files the runtime needs; reports and top-level cards are optional."""
    optional = ("reports/", "README", "NOTICE", "LICENSE", "release_manifest.json")
    return {name: entry for name, entry in manifest["files"].items()
            if not any(name.startswith(prefix) for prefix in optional)}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path: Path, spec: ModelSpec, *, hashes=True) -> dict:
    """Compare a directory with the pinned manifest; never modifies it."""
    path = Path(path)
    expected = runtime_files(spec.manifest)
    problems = {}
    for name, entry in expected.items():
        target = path / name
        if not target.is_file():
            problems[name] = "missing"
        elif target.stat().st_size != entry["size"]:
            problems[name] = f"size {target.stat().st_size} != {entry['size']}"
        elif hashes and digest(target) != entry["sha256"]:
            problems[name] = "sha256 mismatch"
    return {"name": spec.name, "path": str(path), "files": len(expected), "problems": problems,
            "hashes_checked": hashes, "passed": not problems}


def installed_path(name: str) -> Path:
    return models_root() / name


def is_installed(path: Path) -> bool:
    return (path / MARKER).is_file() and (path / "qev_config.json").is_file()


def resolve(spec: str | None = None) -> Path:
    """A registry name or alias, or a local checkpoint directory."""
    spec = spec or os.environ.get("QEV_MODEL") or DEFAULT_MODEL
    name = canonical(spec)
    if name is not None:
        path = installed_path(name)
        if not is_installed(path):
            raise ModelNotInstalled(name)
        return path
    path = Path(spec).expanduser()
    if (path / "qev_config.json").is_file():
        return path.resolve()
    raise ValueError(f"Unknown model {spec!r}: use one of {', '.join(REGISTRY)} or a checkpoint directory")


def _write_marker(path: Path, spec: ModelSpec, source: str, report: dict):
    marker = {"name": spec.name, "repo_id": spec.repo_id, "revision": spec.revision, "runtime": spec.runtime,
              "source": source, "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "verified_files": report["files"]}
    (path / MARKER).write_text(json.dumps(marker, indent=2) + "\n")


def pull(name: str, *, source: str | None = None, link: bool = False, force: bool = False, log=print) -> Path:
    """Download (or import from ``source``) and verify one registry model."""
    canonical_name = canonical(name)
    if canonical_name is None:
        raise ValueError(f"Unknown model {name!r}: use one of {', '.join(REGISTRY)}")
    spec = REGISTRY[canonical_name]
    target = installed_path(canonical_name)
    if is_installed(target) and not force:
        log(f"{canonical_name} is already installed at {target}")
        return target
    models_root().mkdir(parents=True, exist_ok=True)
    staging = models_root() / f".{canonical_name}.partial"
    if source is None:
        from huggingface_hub import snapshot_download

        gib = spec.size_bytes / 2**30
        log(f"Downloading {spec.repo_id}@{spec.revision[:7]} ({gib:.2f} GiB) to {target}")
        snapshot_download(spec.repo_id, revision=spec.revision, local_dir=staging)
        origin = f"hf://{spec.repo_id}@{spec.revision}"
    else:
        source_path = Path(source).expanduser().resolve()
        report = verify(source_path, spec)
        if not report["passed"]:
            raise ValueError(f"{source_path} does not match {spec.repo_id}@{spec.revision[:7]}: {report['problems']}")
        shutil.rmtree(staging, ignore_errors=True)
        for relative in runtime_files(spec.manifest):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if link:
                destination.symlink_to(source_path / relative)
            else:
                shutil.copy2(source_path / relative, destination)
        origin = str(source_path)
    log("Verifying SHA-256 of every runtime file")
    report = verify(staging, spec)
    if not report["passed"]:
        raise ValueError(f"Verification failed; kept {staging} for inspection: {report['problems']}")
    _write_marker(staging, spec, origin, report)
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    log(f"Installed {canonical_name} ({report['files']} verified files)")
    return target


def list_models() -> list[dict]:
    rows = []
    for name, spec in REGISTRY.items():
        path = installed_path(name)
        marker = json.loads((path / MARKER).read_text()) if is_installed(path) else None
        rows.append({"name": name, "repo_id": spec.repo_id, "revision": spec.revision, "runtime": spec.runtime,
                     "size_bytes": spec.size_bytes, "installed": marker is not None, "path": str(path),
                     "source": marker and marker["source"], "description": spec.description,
                     "default": name == DEFAULT_MODEL})
    return rows


def remove(name: str) -> Path:
    canonical_name = canonical(name)
    if canonical_name is None:
        raise ValueError(f"Unknown model {name!r}")
    path = installed_path(canonical_name)
    if not path.exists():
        raise ModelNotInstalled(canonical_name)
    shutil.rmtree(path)
    return path
