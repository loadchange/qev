"""Bundle the source, Snake data and the published Qev Snake adapter for Colab."""
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
archive = root / "runs/qevd-code.tar.gz"
archive.parent.mkdir(exist_ok=True)
paths = [root / "pyproject.toml", root / "scripts/benchmark_snake.py"]
for folder in ("qev", "experiments", "data/snake-v1", "models/qev-snake-0.8b"):
    paths.extend(p for p in (root / folder).rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != "episodes.jsonl")
with tarfile.open(archive, "w:gz") as tar:
    for path in sorted(paths):
        tar.add(path, arcname=str(path.relative_to(root)))
print(archive, f"{archive.stat().st_size:,} bytes", len(paths), "files")
