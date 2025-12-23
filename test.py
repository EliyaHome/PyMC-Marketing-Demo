import os
import sys

# --- MAGIC FIX: Help JAX find the pip-installed NVIDIA libraries ---
# This adds the current environment's library paths to the system lookup
libs_path = os.path.join(sys.prefix, 'lib', 'python3.12', 'site-packages', 'nvidia')
if os.path.exists(libs_path):
    for root, dirs, files in os.walk(libs_path):
        for d in dirs:
            os.environ["LD_LIBRARY_PATH"] = f"{os.path.join(root, d)}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
# -------------------------------------------------------------------

import jax
print("JAX Devices:", jax.devices())