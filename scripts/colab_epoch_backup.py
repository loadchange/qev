"""Back up the first complete epoch while the second epoch continues training."""
import hashlib
import json
import tarfile
from pathlib import Path

root = Path("/content/qev")
checkpoint = root / "models/qev-0.8b"
config_path = checkpoint / "qev_config.json"
if not config_path.exists():
    print("NO_COMPLETED_EPOCH_YET")
else:
    config = json.loads(config_path.read_text())
    if config.get("completed_epochs") != 1:
        print("SKIP_EPOCH_BACKUP", config.get("completed_epochs"))
    elif not (checkpoint / "processor/tokenizer_config.json").exists():
        print("WAIT_FOR_PROCESSOR_SAVE")
    else:
        archive = Path("/content/qev-epoch1.tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(checkpoint, arcname="qev-epoch1")
        after = json.loads(config_path.read_text())
        if config != after:
            raise RuntimeError("Checkpoint changed during backup; use final collection instead")
        print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                          "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))
