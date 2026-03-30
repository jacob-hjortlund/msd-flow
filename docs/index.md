# msd-flow

**msd-flow** is a flow matching framework for generative modelling of galaxy images.
It trains a velocity-field network to learn a probability flow from Gaussian noise
to the target image distribution, enabling unconditional and conditional image generation.

## Quick Start

**1. Install** (GPU, requires CUDA 12):

```bash
pip install msd-flow
```

**2. Set your TNG API key** (required to download training data):

```bash
export TNG_API_KEY=your_key_here
```

**3. Run training** from the repository root:

```bash
python train_model.py
```

Hydra creates a timestamped output directory under `outputs/` containing logs and the
resolved config for that run.

## Documentation

| Page | Description |
|------|-------------|
| [Installation](installation.md) | Install options for GPU/CPU and building the docs locally |
| [Configuration](configuration.md) | Understanding the Hydra config system and every config group |
| [Training](training.md) | How `train_model.py` works, CLI overrides, and loading checkpoints |
| [Implementing Metrics](metrics.md) | Adding custom batch and epoch metrics |
| [TODO](todo.md) | Planned features and contribution ideas |
