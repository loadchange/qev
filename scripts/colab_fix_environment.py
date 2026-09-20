"""Remove an unused, incompatible Colab preinstall without replacing torch."""
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
print("TORCHAO_REMOVED", flush=True)
