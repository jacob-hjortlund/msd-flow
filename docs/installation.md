# Installation

## Prerequisites

- Python ≥ 3.10
- A [TNG API key](https://www.tng-project.org/users/register/) (required to download training data)

## GPU Install (default)

Installs with `jax[cuda12]` for NVIDIA GPU acceleration:

```bash
pip install msd-flow
```

!!! note
    Requires CUDA 12 and compatible NVIDIA drivers.
    See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
    if you encounter CUDA version mismatches.

## CPU Install

For CPU-only machines, install the `cpu` extra to replace the GPU JAX build:

```bash
pip install "msd-flow[cpu]"
```

If you already have the GPU variant installed, uninstall JAX first:

```bash
pip uninstall jax jaxlib
pip install "msd-flow[cpu]"
```

## Development Install

To install from source in editable mode:

```bash
git clone <repo-url>
cd msd-flow
pip install -e ".[dev]"
```

## Environment Variable

Set your TNG API key before downloading data:

```bash
export TNG_API_KEY=your_key_here
```

Add this to your shell profile (`.bashrc`, `.zshrc`, etc.) to make it permanent.

## Building the Docs Locally

Install the `docs` extra and serve:

```bash
pip install "msd-flow[docs]"
mkdocs serve
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.
Run this command from the repository root (the directory containing `mkdocs.yml`).
