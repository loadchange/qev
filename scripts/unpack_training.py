"""Unpack the Colab artifact and verify every checkpoint file."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("archive")
parser.add_argument("--sha256", required=True)
args = parser.parse_args()
archive = Path(args.archive)
actual = hashlib.sha256(archive.read_bytes()).hexdigest()
if actual != args.sha256:
    raise RuntimeError(f"Archive checksum mismatch: {actual}")
root = Path(__file__).resolve().parents[1]
output = root / "models/qev-0.8b"
if output.exists():
    raise FileExistsError(f"Refusing to overwrite existing checkpoint: {output}")
with tarfile.open(archive, "r:gz") as tar:
    tar.extractall(root, filter="data")
hashes = json.loads((output / "SHA256SUMS.json").read_text())
for name, expected in hashes.items():
    path = (output / name).resolve()
    if not path.is_relative_to(output.resolve()):
        raise ValueError("Unsafe checkpoint manifest path")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Checkpoint checksum mismatch: {name}")
print(f"Verified {len(hashes)} checkpoint files: {output}")
