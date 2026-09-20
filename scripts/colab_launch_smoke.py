"""Launch in the persistent Colab kernel; poll via colab_poll.py."""
import os
import subprocess

with open("/content/qev-smoke.log", "w") as qev_smoke_log:
    qev_smoke = subprocess.Popen(["python", "-u", "scripts/colab_smoke.py"],
        cwd="/content/qev", stdout=qev_smoke_log, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH":"/content/qev", "HF_HUB_DISABLE_PROGRESS_BARS":"1"})
print("QEV_SMOKE_PID", qev_smoke.pid)
