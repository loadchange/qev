"""Create a checksummed archive only after the owned training run succeeds."""
import hashlib
import importlib.metadata
import json
import platform
import tarfile
from pathlib import Path

root = Path("/content/qev")
out = root / "models/qev-0.8b"
job = globals().get("qev_job")
if job is None or job.poll() != 0:
    raise RuntimeError("Training must complete successfully before collection")
if not (out / "evaluation.json").is_file():
    raise RuntimeError("Missing final evaluation")
if not json.loads((out / "base_integrity.json").read_text())["unchanged"]:
    raise RuntimeError("Frozen base integrity check failed")
versions = {"python": platform.python_version()}
for package in ("torch", "torchvision", "transformers", "peft", "accelerate",
                "flash-linear-attention", "kernels", "Pillow", "safetensors", "numpy"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
(out / "environment.json").write_text(json.dumps(versions, indent=2) + "\n")
hashes = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
          for path in sorted(out.rglob("*")) if path.is_file() and path.name != "SHA256SUMS.json"}
(out / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n")
archive = Path("/content/qev-trained.tar.gz")
with tarfile.open(archive, "w:gz") as tar:
    tar.add(out, arcname="models/qev-0.8b")
    for name in ("native_baseline.json", "native_validation.json", "service_torch.json", "service_torch.server.log"):
        path = root / "runs" / name
        if path.is_file():
            tar.add(path, arcname=f"runs/{name}")
    for name in ("qev-train.log", "qev-smoke.log", "qev-native.log", "qev-service.log"):
        path = Path("/content") / name
        if path.is_file():
            tar.add(path, arcname=f"runs/{name}")
print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                  "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                  "checkpoint_files": len(hashes)}))
