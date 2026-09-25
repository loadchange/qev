"""Install experiment dependencies while keeping Colab's CUDA PyTorch."""
import os
import subprocess
import sys

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout)
# Colab's preinstalled torchao is incompatible with PEFT 0.21 and unused here.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers==5.17.0", "peft==0.21.0", "accelerate>=1.12", "safetensors>=0.6",
    "huggingface-hub>=0.34", "pydantic>=2.10", "numpy>=2.2", "pyarrow>=18", "Pillow>=11",
    "kernels", "flash-linear-attention"])
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
print("QEVD_SETUP_COMPLETE", flush=True)
