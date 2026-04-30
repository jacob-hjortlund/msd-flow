# scripts/check_jax_gpu_memory.py
import os
import subprocess

print(
    "XLA_PYTHON_CLIENT_MEM_FRACTION =", os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
)
print(
    "XLA_PYTHON_CLIENT_PREALLOCATE =", os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
)
print("XLA_PYTHON_CLIENT_ALLOCATOR =", os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"))

print("\nBefore JAX op:")
subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv",
    ],
    check=False,
)

import jax
import jax.numpy as jnp

# Trigger GPU backend / allocator.
x = jnp.zeros((1,), device=jax.devices("gpu")[0])
x.block_until_ready()

print("\nAfter tiny JAX op:")
subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv",
    ],
    check=False,
)

print("\nJAX memory_stats:")
for d in jax.devices("gpu"):
    print(d)
    stats = d.memory_stats()
    if stats is None:
        print("  memory_stats unavailable")
        continue
    for k, v in stats.items():
        if isinstance(v, int):
            print(f"  {k}: {v / 1024**3:.3f} GiB")
        else:
            print(f"  {k}: {v}")
