"""Extract the locally generated project bundle into the owned session."""
import tarfile
from pathlib import Path

root = Path("/content/qev")
root.mkdir(exist_ok=True)
with tarfile.open("/content/qev-code.tar.gz", "r:gz") as tar:
    tar.extractall(root, filter="data")
print("QEV_SOURCE_READY", len(list(root.rglob("*.py"))))
