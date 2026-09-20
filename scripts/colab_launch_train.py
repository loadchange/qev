"""Start the reproducible Qev v1 supervised run after the smoke test passes."""
import os
import subprocess
from pathlib import Path

smoke = globals().get("qev_smoke")
if smoke is None or smoke.poll() != 0:
    raise RuntimeError("Run and pass the real multimodal smoke test first")
if "QEV_NATIVE_VISION_OK" not in Path("/content/qev-smoke.log").read_text():
    raise RuntimeError("Native multimodal smoke test has not completed")
job = globals().get("qev_job")
if job is not None and job.poll() is None:
    raise RuntimeError("A Qev training process is already running")
with open("/content/qev-train.log", "w") as qev_train_log:
    qev_job = subprocess.Popen(["python", "-u", "-m", "qev.train", "--data", "data/v1",
        "--out", "models/qev-0.8b", "--epochs", "2", "--batch", "4", "--accum", "4",
        "--lr", "5e-5", "--device", "cuda"],
        cwd="/content/qev", stdout=qev_train_log, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH":"/content/qev", "HF_HUB_DISABLE_PROGRESS_BARS":"1"})
print("QEV_TRAIN_PID", qev_job.pid)
