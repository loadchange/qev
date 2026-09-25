"""`qev doctor`: environment, model and server checks for this machine."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _sysctl(name):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=2, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect(server=None, *, verify_hashes=False):
    """Return (checks, facts); each check is (status, name, detail) with ok/warn/fail."""
    from . import __version__
    from .client import health, server_url
    from .models import REGISTRY, home, is_installed, list_models, verify

    checks, facts = [], {"qev": __version__, "python": sys.version.split()[0], "executable": sys.executable}
    system, machine = platform.system(), platform.machine()
    mac = platform.mac_ver()[0]
    facts.update(system=system, machine=machine, macos=mac or None)
    if system == "Darwin" and machine == "arm64":
        chip = _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
        memory = int(_sysctl("hw.memsize") or 0)
        facts.update(chip=chip, memory_bytes=memory)
        checks.append(("ok", "platform", f"{chip}, macOS {mac}, {memory / 2**30:.0f} GiB memory"))
        if memory and memory < 8 * 2**30:
            checks.append(("warn", "memory", "Less than 8 GiB: the float32 model needs about 4 GiB while serving"))
    else:
        checks.append(("warn", "platform", (f"{system} {machine}: the MLX runtime needs Apple Silicon; "
                                            "use the PyTorch backend elsewhere")))
    if importlib.util.find_spec("mlx") is None:
        checks.append(("fail" if system == "Darwin" else "warn", "mlx", "Not installed"))
    else:
        try:
            import mlx.core as mx

            info = (mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()) if mx.metal.is_available() else {}
            facts.update(mlx=_version("mlx"), mlx_vlm=_version("mlx-vlm"), metal=bool(info))
            detail = f"mlx {_version('mlx')}, mlx-vlm {_version('mlx-vlm')}"
            if info:
                detail += f", Metal {info.get('architecture', 'GPU')}"
            checks.append(("ok" if info else "fail", "mlx", detail if info else detail + ", Metal unavailable"))
        except Exception as error:  # noqa: BLE001 -- report any broken install
            checks.append(("fail", "mlx", f"{type(error).__name__}: {error}"))
    from .processing import processor_backend

    backend = processor_backend()
    facts.update(torch=_version("torch"), processor=backend)
    checks.append(("ok", "processor", "Transformers image/video processors (torch)" if backend == "hf"
                   else "numpy image/video processors (no torch needed)"))
    root = home()
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    facts.update(qev_home=str(root), free_bytes=free)
    writable = os.access(root, os.W_OK)
    checks.append(("ok" if writable else "fail", "QEV_HOME", f"{root} ({free / 2**30:.1f} GiB free)"
                   + ("" if writable else ", not writable")))
    for row in list_models():
        if row["installed"]:
            if verify_hashes:
                report = verify(row["path"], REGISTRY[row["name"]])
                status, detail = ("ok", "verified") if report["passed"] else ("fail", f"{len(report['problems'])} bad files")
            else:
                status, detail = ("ok", "installed") if is_installed(Path(row["path"])) else ("fail", "incomplete")
            checks.append((status, f"model {row['name']}", f"{detail}, {row['path']}"))
        else:
            need = row["size_bytes"] / 2**30
            status = "warn" if free > row["size_bytes"] * 1.1 else "fail"
            checks.append((status, f"model {row['name']}", f"not installed ({need:.2f} GiB): qev pull {row['name']}"))
    url = server_url(server)
    document = health(url)
    facts["server"] = document
    checks.append(("ok", "server", f"{url}: {document.get('model', '?')} on {document.get('backend', '?')}")
                  if document else ("warn", "server", f"none at {url}; `qev serve` or `brew services start qev` keeps the model warm"))
    return checks, facts


def main(args):
    checks, facts = collect(args.server, verify_hashes=args.verify)
    if args.json:
        print(json.dumps({"checks": [{"status": s, "name": n, "detail": d} for s, n, d in checks], "facts": facts},
                         ensure_ascii=False, indent=2))
    else:
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        marks = {"ok": ("✓", "32"), "warn": ("!", "33"), "fail": ("✗", "31")}
        for status, name, detail in checks:
            mark, code = marks[status]
            print(f"{f'\033[{code}m{mark}\033[0m' if color else mark} {name}: {detail}")
    return 1 if any(status == "fail" for status, _, _ in checks) else 0
