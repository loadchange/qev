"""Archive every arm's reports and checkpoints with a checksum manifest."""
import hashlib
import json
import tarfile
from pathlib import Path

out_root = Path(globals().get("QEVD_OUT", "/content/qevd-runs"))
files = sorted(p for p in out_root.rglob("*") if p.is_file())
extra = [p for p in (Path("/content/qevd-queue.log"), Path("/content/qevd-mm-probe.json"),
                     Path("/content/qevd-mm-probe.log")) if p.exists()]
hashes = {str(p.relative_to(out_root.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files + extra}
manifest = out_root / "SHA256SUMS.json"
manifest.write_text(json.dumps(hashes, indent=2) + "\n")
archive = Path("/content/qevd-results.tar.gz")
with tarfile.open(archive, "w:gz") as tar:
    tar.add(out_root, arcname=out_root.name)
    for path in extra:
        tar.add(path, arcname=f"{out_root.name}/{path.name}")
print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "files": len(hashes),
                  "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))
