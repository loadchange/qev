"""Package only project source and the explicitly selected v1 training data."""
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
archive = root / "runs/qev-code.tar.gz"
archive.parent.mkdir(exist_ok=True)
with tarfile.open(archive, "w:gz") as tar:
    paths = [root / "pyproject.toml", root / "uv.lock"]
    for folder in ("qev", "scripts", "data/v1"):
        paths.extend(p for p in (root / folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix != ".pyc")
    for path in sorted(paths):
        tar.add(path, arcname=str(path.relative_to(root)))
print(archive)
print(f"{archive.stat().st_size:,} bytes")
