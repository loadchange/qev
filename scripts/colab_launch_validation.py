"""Validate the final checkpoint in a background process after successful training."""
import os
import subprocess

job = globals().get("qev_job")
if job is None or job.poll() != 0:
    raise RuntimeError("Final training and evaluation must succeed before native validation")
with open("/content/qev-native.log", "w") as log:
    qev_validation = subprocess.Popen(["python", "-u", "scripts/validate_native.py",
        "--checkpoint", "models/qev-0.8b", "--output", "runs/native_validation.json"],
        cwd="/content/qev", stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH":"/content/qev", "HF_HUB_DISABLE_PROGRESS_BARS":"1"})
print("QEV_VALIDATION_PID", qev_validation.pid)
