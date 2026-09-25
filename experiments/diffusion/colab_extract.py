"""Extract the uploaded bundle into /content/qev."""
import tarfile
from pathlib import Path

root = Path("/content/qev")
root.mkdir(exist_ok=True)
with tarfile.open("/content/qevd-code.tar.gz", "r:gz") as tar:
    tar.extractall(root, filter="data")
print("QEVD_SOURCE_READY", len(list(root.rglob("*.py"))), "python files")
