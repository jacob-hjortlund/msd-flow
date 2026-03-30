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
(train split only). See [Configuration](configuration.md) for the full pipeline description.

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

The training loop (`msdflow/train/trainer.py`) runs for `num_epochs` epochs.

### Per-Step Logic

For each batch:

1. **Sample time** `t ∈ [0, 1)` uniformly per sample.
2. **Sample noise** `x_0 ~ N(0, I)`.
3. **Couple** `(x_0, x_1)` — by default independently; optionally via optimal transport.
4. **Interpolate** to get `x_t = x_0 + t * (x_1 - x_0)` and target velocity `u_t = x_1 - x_0`.
5. **Forward + backward pass** — compute `loss = MSE(model(t, x_t, cond, cond_mask), u_t)`, compute gradients, apply AdamW update.
6. **EMA update** — `ema_model = decay * ema_model + (1 - decay) * model`.

### Validation (every `val_every` epochs)

- Runs all **batch metrics** over the full val set and up to `num_train_eval_batches` train batches. Results logged as `val/<name>` and `train/<name>`.
- Runs all **epoch metrics** (if any) on `num_val_eval_batches` collected val batches. Results logged as `epoch/<name>`.

### Checkpointing (every `checkpoint_every` epochs)

Saves two `.eqx` files to `checkpoint_dir`:

```
checkpoints/
  model_epoch10_raw.eqx    # Instantaneous model weights
  model_epoch10_ema.eqx    # EMA model weights — use this for inference
```

### Logging (every `log_every` epochs)

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
