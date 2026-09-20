"""Install only the training dependencies, retaining Colab's CUDA PyTorch."""
import os
import subprocess
import sys

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True).stdout)
# Colab's preinstalled torchao 0.10 is incompatible with PEFT 0.21. Qev
# does not use torchao quantization; leave the CUDA PyTorch install intact.
subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers==5.17.0', 'peft==0.21.0', 'accelerate>=1.12',
    'safetensors>=0.6', 'huggingface-hub>=0.34', 'pydantic>=2.10',
    'fastapi>=0.115', 'uvicorn>=0.30', 'numpy>=2.2', 'kernels',
    'flash-linear-attention', 'Pillow>=11'])
print('QEV_SETUP_COMPLETE', flush=True)
