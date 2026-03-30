# Configuration

msd-flow uses [Hydra](https://hydra.cc/) for configuration management. All training runs are driven by a hierarchy of YAML config files under `configs/`.

## How Hydra Works

Hydra composes a single configuration object from a **defaults list** and individual group configs. The entry point `configs/config.yaml` defines the defaults list:

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

Each line loads a YAML file from a **config group** (e.g. `model/`) and mounts it at a key in the final config object (e.g. `cfg.model`). The `@key` syntax controls the mount point.

### `_target_` Instantiation

Config entries with `_target_` specify a fully-qualified Python class or function. `hydra.utils.instantiate(cfg.some_key)` imports and calls that target with the remaining keys as keyword arguments.

`_partial_: true` returns a `functools.partial` instead of calling it immediately — used for functions that need additional runtime arguments (like `key` or `dataloader`).

### Custom Resolvers

Two custom resolvers are registered at startup:

- `${if_cond: <list_or_null>, <if_truthy>, <if_falsy>}` — returns `if_truthy` when the first argument is a non-empty list, otherwise `if_falsy`. Used to auto-configure `cond_dim` and `p_uncond` based on whether `metadata_columns` is set.
- `${generate_snapshot_ids: <end>, <count>}` — generates `count` evenly-spaced snapshot IDs ending at `end`. Used in the download config.

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

1. Copy and rename the relevant config file, e.g. `cp configs/model/ncsnpp.yaml configs/model/ncsnpp_small.yaml`
2. Edit the new file.
3. Pass the override on the command line: `python train_model.py model=ncsnpp_small`

## Config Groups

### `configs/config.yaml` — Top-Level Config

```yaml
seed: 42                           # Global random seed for JAX PRNG
image_size: 512                    # Spatial resolution (H and W) fed to the model
metadata_columns: null             # List of metadata column names for conditioning,
                                   # or null for unconditional training
work_dir: ${hydra:runtime.cwd}    # Resolves to the current working directory at runtime
```

`metadata_columns` is the main toggle for conditional training. Setting it to a list (e.g. `[stellar_mass]`) enables classifier-free guidance; `null` trains unconditionally. The `if_cond` resolver automatically sets `cond_dim` and `p_uncond` based on this value.

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

Override `max_workers` for faster downloads: `python train_model.py data.download.max_workers=10`

### `configs/data/dataset.yaml` — Dataset Split Config

```yaml
dataset_name: "TNG50"                              # Name tag used for directory organisation
data_dir: "${hydra:runtime.cwd}/data"              # Root directory for processed dataset files
seed: ${seed}                                      # Inherited from top-level seed for reproducibility
ratios:
  train: 0.90                                      # Fraction of data assigned to the training split
  val:   0.05                                      # Fraction assigned to the validation split
  test:  0.05                                      # Fraction assigned to the test split
skip_download: false    # Set to true to skip download and reuse existing data
```

### `configs/data/dataloader.yaml` — Image Preprocessing Pipeline

The dataloader config assembles a multi-stage preprocessing pipeline. Each raw FITS image passes through the following transforms in order:

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
activation:
  _target_: jax.nn.swish                       # Activation function (swish by default)
  _partial_: true
fourier_scale: 16.0                            # Random Fourier time embedding scale
skip_rescale: true                             # Rescale skip connections
image_size: ${image_size}                      # Inherited from top-level config
cond_dim: "${if_cond: ${metadata_columns}, 1, 0}"  # Auto-set: 1 if conditioning, 0 otherwise
prediction_type: "velocity"                    # "velocity" or "image"
```

Switch to UNet: `python train_model.py model=unet`

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
activation:
  _target_: jax.nn.silu                        # Activation function (silu by default)
  _partial_: true
cond_dim: "${if_cond: ${metadata_columns}, 1, 0}"
prediction_type: "velocity"
```

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

### `configs/clearml/clearml.yaml` — Experiment Tracking

```yaml
enabled: false                          # Set to true to enable ClearML
project_name: msd-flow
task_name: train
offline_dir: ${work_dir}/.clearml_offline   # Used if the ClearML server is unreachable
```

Enable with: `python train_model.py clearml.enabled=true clearml.task_name=my_experiment`
