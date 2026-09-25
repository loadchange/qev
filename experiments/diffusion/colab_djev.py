"""Install vLLM nightly, serve DiffusionGemma with djev-run's flags, run the reads.

Run only after the training queue on this VM has finished: vLLM nightly
replaces Colab's PyTorch. Everything runs in one background shell job.
"""
import os
import subprocess

job = globals().get("djev_job")
if job is not None and job.poll() is None:
    raise RuntimeError("The djev job is already running")
script = r"""
set -u
python -c "import vllm" 2>/dev/null || pip uninstall -y torchaudio torchvision > /content/djev-install.log 2>&1
python -c "import vllm" 2>/dev/null || pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly >> /content/djev-install.log 2>&1 || { echo DJEV_INSTALL_FAILED; exit 1; }
echo DJEV_INSTALLED $(python -c 'import vllm; print(vllm.__version__)')
export VLLM_ENABLE_V1_MULTIPROCESSING=0
vllm serve nvidia/diffusiongemma-26B-A4B-it-NVFP4 --port 8000 --served-model-name djev-dgemma \
  --trust-remote-code --enforce-eager --language-model-only --attention-backend TRITON_ATTN \
  --kv-cache-memory 2G --kv-cache-dtype bfloat16 --max-num-seqs 32 --max-model-len 4096 \
  --diffusion-config '{"canvas_length":128}' --override-generation-config '{"max_new_tokens":null}' \
  > /content/djev-server.log 2>&1 &
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8000/health > /dev/null; then echo DJEV_SERVER_READY $i; break; fi
  sleep 5
done
curl -sf http://127.0.0.1:8000/health > /dev/null || { echo DJEV_SERVER_FAILED; tail -40 /content/djev-server.log; exit 1; }
cd /content/qev && PYTHONPATH=/content/qev python -u experiments/diffusion/djev_zeroshot.py --base http://127.0.0.1:8000 \
  --output /content/qevd-runs/djev-zeroshot.json
echo DJEV_DONE $?
"""
djev_job = subprocess.Popen(["bash", "-c", script], stdout=open("/content/djev.log", "w"),  # noqa: SIM115 -- the background job owns the handle
                            stderr=subprocess.STDOUT, env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
print("DJEV_PID", djev_job.pid)
