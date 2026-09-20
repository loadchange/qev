"""Run real HTTP/SDK acceptance, using the final Torch checkpoint and CUDA."""
import os
import subprocess

job = globals().get("qev_job")
if job is None or job.poll() != 0:
    raise RuntimeError("Final training and evaluation must succeed first")
native = globals().get("qev_validation")
if native is None or native.poll() != 0:
    raise RuntimeError("Complete native preservation validation first")
with open("/content/qev-service.log", "w") as log:
    qev_service = subprocess.Popen(["python", "-u", "scripts/validate_service.py",
        "--checkpoint", "models/qev-0.8b", "--backend", "torch", "--device", "cuda",
        "--output", "runs/service_torch.json"],
        cwd="/content/qev", stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH":"/content/qev", "HF_HUB_DISABLE_PROGRESS_BARS":"1"})
print("QEV_SERVICE_PID", qev_service.pid)
