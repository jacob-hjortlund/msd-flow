# Packaging and Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the `src/` package to `msdflow/`, add `pyproject.toml` for pip installation (GPU by default, CPU optional), and create a MkDocs Material documentation site covering installation, configuration, training, and implementing metrics.

**Architecture:** Three sequential phases — (1) rename the Python package and update all references, (2) add packaging metadata, (3) write documentation. Each phase is independently committable. The rename must complete before packaging or docs are written, since both reference `msdflow`.

**Tech Stack:** Python packaging via `setuptools`/`pyproject.toml`, MkDocs with the Material theme, existing JAX/Equinox/Hydra stack.

---

## File Map

**Renamed (Task 1):**
- `src/` → `msdflow/` (all contents preserved, relative imports unchanged)
- All `from src.` imports in `*.py` → `from msdflow.`
- All `src.` strings in `configs/**/*.yaml` → `msdflow.`
- `train_model.py` — 3 import lines updated
- `.claude/CLAUDE.md` — command examples updated

**Created:**
- `pyproject.toml` (Task 2)
- `mkdocs.yml` (Task 3)
- `docs/index.md` (Task 4)
- `docs/installation.md` (Task 5)
- `docs/configuration.md` (Task 6)
- `docs/training.md` (Task 7)
- `docs/metrics.md` (Task 8)
- `docs/todo.md` (Task 9)

---

## Task 1: Rename `src/` → `msdflow/` and update all references

**Files:**
- Rename: `src/` → `msdflow/`
- Modify: all `*.py` files containing `from src.` or `import src.`
- Modify: `configs/**/*.yaml` files containing `src.`
- Modify: `.claude/CLAUDE.md`

- [ ] **Step 1: Rename the directory with git**

```bash
git mv src msdflow
```

Expected: no output. Verify with `ls` — `msdflow/` should exist, `src/` should be gone.

- [ ] **Step 2: Update all Python import statements**

```bash
find . -name "*.py" -not -path "./.git/*" | xargs sed -i \
  's/from src\./from msdflow\./g; s/import src\./import msdflow\./g'
```

Expected: no output. This updates all `from src.X` and `import src.X` in every `.py` file including `train_model.py` and all test files.

- [ ] **Step 3: Update all `_target_` strings in YAML config files**

```bash
find configs -name "*.yaml" | xargs sed -i 's/src\./msdflow\./g'
```

Expected: no output. Verify spot-check:

```bash
grep -r "src\." configs/
```

Expected: no output (all occurrences replaced).

- [ ] **Step 4: Update CLAUDE.md command examples**

Open `.claude/CLAUDE.md` and change:

```
- **Download data:** `python -m src.data.download_tng`
- **Assign splits:** `python -m src.data.split`
```

to:

```
- **Download data:** `python -m msdflow.data.download_tng`
- **Assign splits:** `python -m msdflow.data.split`
```

- [ ] **Step 5: Run the existing test suite to verify nothing broke**

```bash
pytest tests/ -x -q
```

Expected: all existing tests pass. If imports fail, check that `msdflow/` contains all the original subdirectories with their `__init__.py` files intact.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: rename src/ to msdflow/ for proper Python packaging"
```

---

## Task 2: Create `pyproject.toml`

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write `pyproject.toml`**

Create `/home/jacob/PhD/Projects/msd-flow/Code/msd-flow/pyproject.toml` with the following contents:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "msd-flow"
version = "0.1.0"
description = "Flow matching framework for galaxy image generation"
requires-python = ">=3.10"
dependencies = [
    "jax[cuda12]",
    "equinox",
    "optax",
    "diffrax",
    "hydra-core",
    "omegaconf",
    "torch",
    "torchvision",
    "numpy<2.0.0",
    "scipy",
    "pandas",
    "astropy",
    "h5py",
    "clearml",
    "tqdm",
    "fastdigest",
    "requests",
]

[project.optional-dependencies]
cpu = ["jax[cpu]"]
docs = ["mkdocs-material", "mkdocs-autorefs"]
dev = ["pytest", "pytest-cov"]

[tool.setuptools.packages.find]
where = ["."]
include = ["msdflow*"]
```

- [ ] **Step 2: Install the package in editable mode and verify the import**

```bash
pip install -e . --no-deps
python -c "import msdflow; print('OK')"
```

Expected: `OK`. The `--no-deps` flag skips re-installing dependencies (the env already has them).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add pyproject.toml with GPU-default JAX and optional cpu/docs/dev extras"
```

---

## Task 3: Create `mkdocs.yml`

**Files:**
- Create: `mkdocs.yml`

- [ ] **Step 1: Write `mkdocs.yml`**

Create `/home/jacob/PhD/Projects/msd-flow/Code/msd-flow/mkdocs.yml` with the following contents:

```yaml
site_name: msd-flow
site_description: Flow matching framework for galaxy image generation

theme:
  name: material
  features:
    - navigation.sections
    - navigation.top
    - content.code.copy

nav:
  - Home: index.md
  - Installation: installation.md
  - Configuration: configuration.md
  - Training: training.md
  - Implementing Metrics: metrics.md
  - TODO: todo.md

docs_dir: docs
```

- [ ] **Step 2: Commit**

```bash
git add mkdocs.yml
git commit -m "feat: add mkdocs.yml for MkDocs Material documentation site"
```

---

## Task 4: Create `docs/index.md`

**Files:**
- Create: `docs/index.md`

- [ ] **Step 1: Write `docs/index.md`**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/index.md
git commit -m "docs: add index.md"
```

---

## Task 5: Create `docs/installation.md`

**Files:**
- Create: `docs/installation.md`

- [ ] **Step 1: Write `docs/installation.md`**

```markdown
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

If you already have the GPU variant installed in your environment, uninstall JAX first:

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
```

- [ ] **Step 2: Commit**

```bash
git add docs/installation.md
git commit -m "docs: add installation.md"
```

---

## Task 6: Create `docs/configuration.md`

**Files:**
- Create: `docs/configuration.md`

- [ ] **Step 1: Write `docs/configuration.md`**

```markdown
# Configuration

msd-flow uses [Hydra](https://hydra.cc/) for configuration management. All training
runs are driven by a hierarchy of YAML config files under `configs/`.

## How Hydra Works

Hydra composes a single configuration object from a **defaults list** and individual
group configs. The entry point `configs/config.yaml` defines the defaults list:

```yaml
defaults:
  - data@data.download: download_tng50
  - data@data.dataset: dataset
  - data@data.dataloader: dataloader
  - model@model: ncsnpp
  - flow@flow.sample: sample
  - train@train: train
  - clearml@clearml: clearml
  - _self_
```

Each line loads a YAML file from a **config group** (e.g. `model/`) and mounts it at
a key in the final config object (e.g. `cfg.model`). The `@key` syntax controls the
mount point.

### `_target_` Instantiation

Config entries with `_target_` specify a fully-qualified Python class or function.
`hydra.utils.instantiate(cfg.some_key)` imports and calls that target with the
remaining keys as keyword arguments.

`_partial_: true` returns a `functools.partial` instead of calling it immediately —
used for functions that need additional runtime arguments (like `key` or `dataloader`).

### Custom Resolvers

Two custom resolvers are registered at startup:

- `${if_cond: <list_or_null>, <if_truthy>, <if_falsy>}` — returns `if_truthy` when
  the first argument is a non-empty list, otherwise `if_falsy`. Used to auto-configure
  `cond_dim` and `p_uncond` based on whether `metadata_columns` is set.
- `${generate_snapshot_ids: <end>, <count>}` — generates `count` evenly-spaced
  snapshot IDs ending at `end`. Used in the download config.

### CLI Overrides

Any config value can be overridden from the command line:

```bash
# Change number of training epochs
python train_model.py train.num_epochs=200

# Switch to the UNet model
python train_model.py model=unet

# Enable ClearML experiment tracking
python train_model.py clearml.enabled=true

# Override the learning rate
python train_model.py train.optimizer.learning_rate=5e-5

# Skip data download (use existing processed data)
python train_model.py data.dataset.skip_download=true

# Train at 256×256
python train_model.py image_size=256
```

### Creating a Custom Config Override

To run with a modified config without editing the defaults:

1. Copy and rename the relevant config file, e.g.
   `cp configs/model/ncsnpp.yaml configs/model/ncsnpp_small.yaml`
2. Edit the new file.
3. Pass the override on the command line:
   `python train_model.py model=ncsnpp_small`

---

## Config Groups

### `configs/config.yaml` — Top-Level Config

```yaml
seed: 42                           # Global random seed for JAX PRNG
image_size: 512                    # Spatial resolution (H and W) fed to the model
metadata_columns: null             # List of metadata column names for conditioning,
                                   # or null for unconditional training
work_dir: ${hydra:runtime.cwd}    # Resolves to the current working directory at runtime
```

`metadata_columns` is the main toggle for conditional training. Setting it to a list
(e.g. `[stellar_mass]`) enables classifier-free guidance; `null` trains unconditionally.
The `if_cond` resolver automatically sets `cond_dim` and `p_uncond` based on this value.

---

### `configs/data/download_tng50.yaml` — TNG Data Download

```yaml
_target_: msdflow.data.download_tng.download_tng_data
_partial_: true
api_key: ${oc.env:TNG_API_KEY,null}        # Read from TNG_API_KEY environment variable
version_ids: [0, 1, 2, 3]                  # TNG50 projection view IDs to download
snapshots: ${generate_snapshot_ids:72,20}  # 20 snapshot IDs ending at 72
num_files_per_view: 50                     # Images per projection view per snapshot
max_workers: 5                             # Parallel download threads
raw_dir: "${data.dataset.data_dir}/raw"    # Destination for raw FITS files
bands: ["SUBARU_HSC.I"]                    # Photometric band
batch_size: 100                            # API request batch size
```

Override `max_workers` for faster downloads:
`python train_model.py data.download.max_workers=10`

---

### `configs/data/dataset.yaml` — Dataset Split Config

```yaml
dataset_name: "TNG50"
ratios:
  train: 0.90
  val:   0.05
  test:  0.05
skip_download: false    # Set to true to skip download and reuse existing data
```

---

### `configs/data/dataloader.yaml` — Image Preprocessing Pipeline

The dataloader config assembles a multi-stage preprocessing pipeline.
Each raw FITS image passes through the following transforms in order:

**Stage 1 — Pre-arcsinh** (applied to raw surface brightness maps):

| Transform | What it does |
|-----------|-------------|
| `SurfaceBrightnessToNanomaggies` | Converts AB mag/arcsec² to linear flux (nanomaggies) |
| `ClipAndPad` | Crops or pads to `image_size × image_size` pixels |
| `PDFNorm` | Fits a per-channel percentile normalisation from the training set |

**Stage 2 — Arcsinh stretch:**

| Transform | What it does |
|-----------|-------------|
| `ArcsinhStretch` | Applies `arcsinh(flux / scale)` where `scale` is the training-set `percentile`-th percentile |

**Stage 3 — Post-arcsinh:**

| Transform | What it does |
|-----------|-------------|
| `GlobalNorm` | Linearly rescales the stretched values to `[-1, 1]` using training-set min/max |

**Augmentations** (training split only):

| Transform | What it does |
|-----------|-------------|
| `RandomHorizontalFlip` | Flip with p=0.5 |
| `RandomVerticalFlip` | Flip with p=0.5 |
| `RandomRotation90` | Rotate by 0°, 90°, 180°, or 270° uniformly |

The train DataLoader uses batch size 32 with shuffle; val/test use batch size 64 without shuffle.

---

### `configs/model/ncsnpp.yaml` — NCSN++ (default)

```yaml
_target_: msdflow.model.NCSNpp
_partial_: true
in_channels: 1
out_channels: 1
base_channels: 128
channel_multipliers: [1, 1, 1, 2, 2, 4, 4]   # Channel width at each resolution level
num_res_blocks: 2                              # Residual blocks per resolution level
attn_resolutions: [16]                         # Apply self-attention at this spatial size
dropout: 0.1
num_groups: 32                                 # GroupNorm group count
num_heads: 1                                   # Attention heads
fourier_scale: 16.0                            # Random Fourier time embedding scale
skip_rescale: true                             # Rescale skip connections
image_size: ${image_size}                      # Inherited from top-level config
cond_dim: "${if_cond: ${metadata_columns}, 1, 0}"  # Auto-set: 1 if conditioning, 0 otherwise
prediction_type: "velocity"                    # "velocity" or "image"
```

Switch to UNet: `python train_model.py model=unet`

---

### `configs/model/unet.yaml` — UNet (alternative)

```yaml
_target_: msdflow.model.UNet
_partial_: true
in_channels: 1
out_channels: 1
base_channels: 64
channel_multipliers: [1, 2, 4, 8]
num_res_blocks: 2
num_heads: 8
num_groups: 8
cond_dim: "${if_cond: ${metadata_columns}, 1, 0}"
prediction_type: "velocity"
```

---

### `configs/flow/sample.yaml` — ODE Sampler

```yaml
t0: 0.0                                      # Integration start time
t1: 1.0                                      # Integration end time (data distribution)
dt0: 0.01                                    # Initial step size (100 steps for Euler)
solver: "diffrax.Euler"                      # ODE solver (any diffrax solver class name)
stepsize_controller: "diffrax.ConstantStepSize"
stepsize_controller_cfg:
  - rtol: 0.001
  - atol: 0.000001
guidance_scale: 1.0                          # CFG guidance scale; 1.0 = no extra guidance
```

---

### `configs/train/train.yaml` — Training Hyperparameters

```yaml
_target_: msdflow.train.trainer.train
_partial_: true

loss_fn:
  _target_: msdflow.train.metrics.flow_matching_loss
  _partial_: true

batch_metrics:
  - _target_: msdflow.train.metrics.flow_matching_loss
    _partial_: true

epoch_metrics: []               # List of epoch-level metrics; empty by default

num_train_eval_batches: 0       # Train batches used for batch metrics (0 = all)
num_val_eval_batches: 0         # Val batches collected for epoch metrics (0 = all)

optimizer:
  _target_: optax.adamw
  learning_rate: 1.0e-4

coupling:
  _target_: msdflow.flow.independent_coupling   # or msdflow.flow.ot_coupling
  _partial_: true

time_sampler:
  _target_: msdflow.flow.sample_time_uniform
  _partial_: true
  t_min: 0.0
  t_max: 1.0

path_sampler:
  _target_: msdflow.flow.sample_path
  _partial_: true
  sigma_0: 0.0      # Noise on x_0 (0 = straight OT path)
  sigma_1: 0.0      # Noise on x_1

num_epochs: 100
num_steps_per_epoch: 0          # 0 = full dataloader per epoch
p_uncond: "${if_cond: ${metadata_columns}, 0.1, 1.0}"  # CFG condition drop probability

checkpoint_dir: ${work_dir}/checkpoints
val_every: 1                    # Validate every N epochs
checkpoint_every: ${train.val_every}
log_every: ${train.val_every}
ema_decay: 0.9999

sample_fn: null                 # Set to a sampling callable to enable sample generation
sample_every: 0                 # Generate samples every N epochs (0 = disabled)
num_samples: 4                  # Number of samples to generate per event
samples_dir: ${work_dir}/samples
```

---

### `configs/clearml/clearml.yaml` — Experiment Tracking

```yaml
enabled: false                          # Set to true to enable ClearML
project_name: msd-flow
task_name: train
offline_dir: ${work_dir}/.clearml_offline   # Used if the ClearML server is unreachable
```

Enable with: `python train_model.py clearml.enabled=true clearml.task_name=my_experiment`
```

- [ ] **Step 2: Commit**

```bash
git add docs/configuration.md
git commit -m "docs: add configuration.md"
```

---

## Task 7: Create `docs/training.md`

**Files:**
- Create: `docs/training.md`

- [ ] **Step 1: Write `docs/training.md`**

```markdown
# Training

## Running Training

All commands must be run from the **repository root** (the directory containing `train_model.py`):

```bash
python train_model.py
```

Hydra creates a timestamped output directory under `outputs/YYYY-MM-DD/HH-MM-SS/`
containing the fully resolved config and the run log.

---

## What `train_model.py` Does

The script executes six sequential steps, each logged to the console:

### Step 0: ClearML Setup

Initialises a ClearML experiment task when `clearml.enabled=true`. If the ClearML
server is unreachable the task falls back to offline mode, writing artefacts to
`clearml.offline_dir`. When `clearml.enabled=false` (default), this step is a no-op.

### Step 1: Dataset Resolution

Calls `resolve_dataset()`, which inspects the target data directory and handles
three cases automatically:

| Condition | Action |
|-----------|--------|
| Data does not exist | Downloads raw FITS files via the TNG API, then assigns splits |
| Data exists, splits don't match | Re-assigns train/val/test splits to existing files |
| Data exists, splits match | Skips all I/O; returns the existing path |

The resolved path is a directory containing processed `.h5` files, one per galaxy image.

### Step 2: Config Injection

Injects the resolved dataset path into `cfg.data.dataloader.data_dir` so that
all downstream DataLoader configs reference the correct directory. This is done at
runtime because the path is not known until after the download step.

### Step 3: Dataloader Initialisation

Instantiates three PyTorch `DataLoader`s (train, val, test) by composing the
preprocessing pipeline from `configs/data/dataloader.yaml`. Each image goes through:
surface brightness conversion → arcsinh stretch → global normalisation → augmentations
(train split only). See [Configuration — dataloader](configuration.md#configsdatadataloaderya-image-preprocessing-pipeline)
for the full pipeline description.

### Step 4: Seeding

Creates a JAX `PRNGKey` from `cfg.seed`. All subsequent random operations
(model initialisation, batch preparation, sampling) split off keys from this root
deterministically.

### Step 5: Model Initialisation

Instantiates the model (NCSNpp by default) using `hydra.utils.instantiate`. The model
is an Equinox module (a JAX pytree): it takes `(t, x_t, cond, cond_mask)` as inputs
and outputs a prediction of the same shape as `x_t`.

### Step 6: Training Loop

Calls `msdflow.train.trainer.train()` with the model, dataloaders, and full config.
See [The Training Loop](#the-training-loop) below.

---

## The Training Loop

The training loop (`msdflow/train/trainer.py::train`) runs for `num_epochs` epochs.

### Per-Step Logic

For each batch:

1. **Sample time** `t ∈ [0, 1)` uniformly per sample.
2. **Sample noise** `x_0 ~ N(0, I)`.
3. **Couple** `(x_0, x_1)` — by default independently; optionally via optimal transport.
4. **Interpolate** to get `x_t = x_0 + t * (x_1 - x_0)` and target velocity `u_t = x_1 - x_0`.
5. **Forward + backward pass** — compute `loss = MSE(model(t, x_t, cond, cond_mask), u_t)`,
   compute gradients, apply AdamW update.
6. **EMA update** — `ema_model = decay * ema_model + (1 - decay) * model`.

### Validation (every `val_every` epochs)

- Runs all **batch metrics** (e.g. flow matching loss) over the full val set and
  up to `num_train_eval_batches` train batches. Results logged as `val/<name>` and `train/<name>`.
- Runs all **epoch metrics** (if any) on `num_val_eval_batches` collected val batches.
  Results logged as `epoch/<name>`.

### Checkpointing (every `checkpoint_every` epochs)

Saves two `.eqx` files to `checkpoint_dir`:

```
checkpoints/
  model_epoch10_raw.eqx    # Instantaneous model weights
  model_epoch10_ema.eqx    # EMA model weights — use this for inference
```

### Logging (every `log_every` epochs)

Logs the following scalars to ClearML (if enabled) and the Python logger:

| Key | Description |
|-----|-------------|
| `train/loss` | Mean per-step training loss for the epoch |
| `val/<metric>` | Each batch metric evaluated on the val split |
| `train/<metric>` | Each batch metric evaluated on the train split |
| `epoch/<metric>` | Each epoch metric |

---

## Common CLI Overrides

```bash
# Train for longer
python train_model.py train.num_epochs=500

# Lower learning rate
python train_model.py train.optimizer.learning_rate=5e-5

# Switch to UNet model
python train_model.py model=unet

# Skip data download (reuse existing processed data)
python train_model.py data.dataset.skip_download=true

# Validate and checkpoint less frequently (saves time)
python train_model.py train.val_every=5

# Use optimal-transport coupling
python train_model.py "train.coupling._target_=msdflow.flow.ot_coupling"

# Train at lower resolution
python train_model.py image_size=256

# Enable ClearML tracking
python train_model.py clearml.enabled=true clearml.task_name=my_run

# Use logit-normal time sampling instead of uniform
python train_model.py "train.time_sampler._target_=msdflow.flow.sample_time_logit_normal"
```

---

## Loading a Checkpoint

Checkpoints are serialised Equinox pytrees. To load one you must first instantiate
a model with the **exact same architecture** as the saved checkpoint, then deserialise
the weights into it:

```python
import jax
import equinox as eqx
from msdflow.model import NCSNpp

# Instantiate with the same hyperparameters used during training
key = jax.random.PRNGKey(0)
model = NCSNpp(
    in_channels=1,
    out_channels=1,
    base_channels=128,
    channel_multipliers=[1, 1, 1, 2, 2, 4, 4],
    num_res_blocks=2,
    attn_resolutions=[16],
    dropout=0.1,
    num_groups=32,
    num_heads=1,
    fourier_scale=16.0,
    skip_rescale=True,
    image_size=512,
    cond_dim=0,
    prediction_type="velocity",
    key=key,
)

# Load the saved EMA weights
model = eqx.tree_deserialise_leaves("checkpoints/model_epoch100_ema.eqx", model)
```

The architecture parameters must match exactly — mismatched shapes will raise an error.
```

- [ ] **Step 2: Commit**

```bash
git add docs/training.md
git commit -m "docs: add training.md"
```

---

## Task 8: Create `docs/metrics.md`

**Files:**
- Create: `docs/metrics.md`

- [ ] **Step 1: Write `docs/metrics.md`**

```markdown
# Implementing Metrics

Metrics in msd-flow are plain Python callables registered in `configs/train/train.yaml`.
There are two types, differing in when and how they are called during training.

---

## Metric Types

### Batch Metrics

**Signature:** `(model, x_t, u_t, t, cond, cond_mask) -> scalar`

Called once per validation batch on **interpolated** data. Used to track training
progress and detect overfitting by comparing train vs. val values.

| Argument | Shape | Description |
|----------|-------|-------------|
| `model` | — | Current EMA model (Equinox module) |
| `x_t` | `(B, C, H, W)` | Interpolated images at time `t` |
| `u_t` | `(B, C, H, W)` | Target velocity field (`x_1 - x_0`) |
| `t` | `(B,)` | Per-sample times in `[0, 1)` |
| `cond` | `(B, cond_dim)` | Conditioning vectors; `(B, 0)` if unconditional |
| `cond_mask` | `(B,)` bool | `True` = use condition; `False` = drop (CFG) |

**Returns:** A scalar JAX array.

Batch metrics are JIT-compiled together into a single `filter_jit` step.
Each metric must have a **unique function name** — the trainer uses `fn.__name__`
as the logging key.

### Epoch Metrics

**Signature:** `(model, val_batches, key) -> scalar`

Called once per validation cycle on a fixed list of **raw** `(images, meta)` batches
collected from the val dataloader. Used for generation-based metrics (e.g. FID)
that need to sample from the model.

| Argument | Description |
|----------|-------------|
| `model` | Current EMA model |
| `val_batches` | `list[tuple[Tensor, Tensor]]` — raw `(images, meta)` PyTorch batches |
| `key` | JAX PRNG key |

**Returns:** A scalar float or JAX array.

Any additional dependencies (ODE solver, number of samples, etc.) must be baked in via
`functools.partial` or Hydra `_partial_: true` in the config.

---

## Example: Custom Batch Metric

The following adds a **mean absolute error** (MAE) metric alongside the default MSE loss.

**Step 1 — Add the function to `msdflow/train/metrics.py`:**

```python
import equinox as eqx
import jax.numpy as jnp
from msdflow.train.metrics import _to_velocity


def flow_matching_mae(
    model,
    x_t: jnp.ndarray,
    u_t: jnp.ndarray,
    t: jnp.ndarray,
    cond: jnp.ndarray,
    cond_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Mean absolute error between predicted and target velocity fields.

    Args:
        model:     Velocity-field network with a ``prediction_type`` attribute.
        x_t:       shape (B, C, H, W) — interpolated samples at time t.
        u_t:       shape (B, C, H, W) — target velocities.
        t:         shape (B,) — per-sample times in [0, 1).
        cond:      shape (B, cond_dim) — conditioning vectors.
        cond_mask: shape (B,) bool — CFG condition mask.

    Returns:
        Scalar mean absolute error.
    """
    pred = eqx.filter_vmap(model)(t, x_t, cond, cond_mask)
    v_t = _to_velocity(pred, x_t, t, model.prediction_type)
    return jnp.mean(jnp.abs(v_t - u_t))
```

**Step 2 — Register it in `configs/train/train.yaml`:**

```yaml
batch_metrics:
  - _target_: msdflow.train.metrics.flow_matching_loss
    _partial_: true
  - _target_: msdflow.train.metrics.flow_matching_mae
    _partial_: true
```

The metric will appear in the logs and ClearML as `val/flow_matching_mae` and
`train/flow_matching_mae`.

!!! warning "Unique function names required"
    Each entry in `batch_metrics` must resolve to a callable with a unique `__name__`.
    If two metrics share a name, the trainer raises a `ValueError` at startup.

---

## Example: Custom Epoch Metric

The following computes the **mean pixel-wise standard deviation** of generated samples —
a basic diversity measure that requires running the ODE sampler.

**Step 1 — Add the function to `msdflow/train/metrics.py`:**

```python
import jax
import jax.numpy as jnp
import numpy as np


def sample_diversity(
    model,
    val_batches: list,
    key: jax.Array,
    sample_fn: callable,
    num_samples: int = 16,
) -> float:
    """Mean pixel-wise standard deviation of generated samples.

    Measures how spread out the model's output distribution is.
    Higher values indicate more diverse samples.

    Args:
        model:       EMA velocity-field network.
        val_batches: Raw ``(images, meta)`` batches from the val dataloader.
                     Used to infer the image shape; values are not used.
        key:         JAX PRNG key for sampling.
        sample_fn:   Callable ``(model, key, image_shape) -> jnp.ndarray``
                     with shape ``(C, H, W)``. Bake in solver config via partial.
        num_samples: Number of images to generate.

    Returns:
        Mean pixel-wise standard deviation across all generated samples.
    """
    images, _ = val_batches[0]
    image_shape = tuple(images.shape[1:])  # (C, H, W)

    keys = jax.random.split(key, num_samples)
    samples = np.stack([np.array(sample_fn(model, k, image_shape)) for k in keys])
    return float(np.mean(np.std(samples, axis=0)))
```

**Step 2 — Register it in `configs/train/train.yaml`:**

```yaml
epoch_metrics:
  - _target_: msdflow.train.metrics.sample_diversity
    _partial_: true
    sample_fn:
      _target_: your.sampling.function   # replace with your ODE sampler callable
      _partial_: true
    num_samples: 16
```

!!! note
    `sample_fn` must be a callable with signature `(model, key, image_shape) -> array`.
    Implement your ODE sampler using `diffrax` and register it here via `_target_`.

The metric will appear in logs as `epoch/sample_diversity`.

---

## Metric Logging

All metrics are logged under fixed prefixes:

| Prefix | Source |
|--------|--------|
| `train/loss` | Per-epoch mean training loss |
| `val/<name>` | Batch metric evaluated on the val split |
| `train/<name>` | Batch metric evaluated on the train split |
| `epoch/<name>` | Epoch metric |

When ClearML is enabled (`clearml.enabled=true`), these scalars are tracked as
time series. Without ClearML they appear in the Python log output.
```

- [ ] **Step 2: Commit**

```bash
git add docs/metrics.md
git commit -m "docs: add metrics.md"
```

---

## Task 9: Create `docs/todo.md`

**Files:**
- Create: `docs/todo.md`

- [ ] **Step 1: Write `docs/todo.md`**

```markdown
# TODO

Planned features and known gaps. Contributions welcome.

- [ ] Implement [submitit](https://github.com/facebookincubator/submitit) launcher plugin for running on SLURM-based clusters
- [ ] Implement generative metrics (e.g. FID, KID)
- [ ] Implement physical property based metrics
- [ ] Implement early stopping based on chosen metric
- [ ] Implement tracking of best model based on same metric as early stopping
- [ ] Add image sampling function in default training config
- [ ] Implement inverse image transforms
```

- [ ] **Step 2: Commit**

```bash
git add docs/todo.md
git commit -m "docs: add todo.md"
```

---

## Task 10: Verify MkDocs Build

**Files:** No changes.

- [ ] **Step 1: Install the docs extra (if not already installed)**

```bash
pip install -e ".[docs]" --no-deps
pip install mkdocs-material mkdocs-autorefs
```

- [ ] **Step 2: Build the site and check for errors**

```bash
mkdocs build --strict
```

Expected: `INFO - Documentation built in X.XX seconds` with no warnings or errors.
`--strict` converts all warnings (e.g. broken links) into errors.

- [ ] **Step 3: Serve locally and verify all pages render correctly**

```bash
mkdocs serve
```

Open [http://localhost:8000](http://localhost:8000) and check:
- Home page renders with quick-start snippet
- All nav links work (Installation, Configuration, Training, Implementing Metrics, TODO)
- Code blocks have copy buttons
- No broken internal links

- [ ] **Step 4: Commit (only if mkdocs.yml or docs needed fixing)**

If any fixes were needed during verification:

```bash
git add -A
git commit -m "docs: fix mkdocs build warnings"
```
