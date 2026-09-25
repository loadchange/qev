"""Start the arm queue as one background process in the persistent kernel.

Arms run sequentially so each trains and is timed alone on the GPU. A failing
arm is recorded and the queue moves on. Set QEVD_ARMS / QEVD_EXTRA before
running this file to change the queue (used for smoke tests).
"""
import json
import os
import subprocess
from pathlib import Path

ARMS = [  # label, backbone, attention
    ("qwen3-causal", "qwen3", "causal"),
    ("a2d-block", "a2d-qwen3", "block"),
    ("lfm2-causal", "lfm2-vl", "causal"),
    ("qwen35-causal", "qwen3_5", "causal"),
    ("qwen3-block", "qwen3", "block"),
    ("lfm2-block", "lfm2-vl", "block"),
]
selected = globals().get("QEVD_ARMS") or [label for label, _, _ in ARMS]
extra = globals().get("QEVD_EXTRA", [])
out_root = globals().get("QEVD_OUT", "/content/qevd-runs")
job = globals().get("qevd_job")
if job is not None and job.poll() is None:
    raise RuntimeError("The QevD queue is already running")
lines = ["set -u", f"mkdir -p {out_root}"]
for label, backbone, attention in ARMS:
    if label not in selected:
        continue
    command = ["python", "-u", "experiments/diffusion/train_qevd.py", "--backbone", backbone,
               "--attention", attention, "--label", label, "--out", f"{out_root}/{label}", *extra]
    lines.append(f"echo QEVD_START {label} $(date +%s)")
    lines.append(" ".join(command) + f" > {out_root}/{label}.log 2>&1; "
                 f"echo QEVD_END {label} $? $(date +%s)")
lines.append("echo QEVD_QUEUE_DONE $(date +%s)")
Path("/content/qevd-queue.sh").write_text("\n".join(lines) + "\n")
qevd_job = subprocess.Popen(["bash", "/content/qevd-queue.sh"], cwd="/content/qev",
                            stdout=open("/content/qevd-queue.log", "w"),  # noqa: SIM115 -- owned by the job
                            stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": "/content/qev", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                                 "TOKENIZERS_PARALLELISM": "false",
                                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
print("QEVD_QUEUE_PID", qevd_job.pid, json.dumps(selected))
