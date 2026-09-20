"""Install only HTTP acceptance-test clients, without replacing CUDA PyTorch."""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "typesafe-sdk==0.7.0", "httpx>=0.28"])
print("QEV_VALIDATION_CLIENTS_READY")
