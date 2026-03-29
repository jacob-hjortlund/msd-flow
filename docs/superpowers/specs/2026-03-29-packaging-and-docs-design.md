# Packaging and Documentation Design

**Date:** 2026-03-29
**Branch:** docs
**Status:** Approved

---

## Overview

Add proper Python packaging via `pyproject.toml` and a MkDocs-based documentation site to the `msd-flow` repository. This enables users to install the package with `pip` and access local documentation covering configuration, the training entry point, and extending the metrics system.

---

## 1. Package Rename: `src` → `msdflow`

The current importable module is named `src`, which is a non-conventional Python package name. It will be renamed to `msdflow` to match standard Python packaging practices.

**Actions:**
- Rename `src/` directory to `msdflow/`
- Update all `from src.` and `import src.` references across the codebase (~30+ files including source files and tests)
- Update any string references to the module name (e.g., in config `_target_` fields: `src.model.NCSNpp` → `msdflow.model.NCSNpp`)
- Update `CLAUDE.md` common commands if they reference `src`

---

## 2. pyproject.toml

A single `pyproject.toml` at the repository root, using `setuptools` as the build backend with a `src`-style layout pointing to the renamed `msdflow/` directory.

### Package Metadata

```toml
[project]
name = "msd-flow"
version = "0.1.0"
description = "Flow matching framework for galaxy image generation"
requires-python = ">=3.10"
```

### Core Dependencies

Extracted from `env.yml`, pinned loosely (no JAX — that is an optional dep):

- `equinox`
- `optax`
- `diffrax`
- `hydra-core`
- `omegaconf`
- `torch`
- `torchvision`
- `numpy<2.0.0`
- `scipy`
- `pandas`
- `astropy`
- `h5py`
- `clearml`
- `tqdm`
- `fastdigest`
- `caustics`
- `requests`

### Optional Dependency Groups

| Group | Contents | Install command |
|-------|----------|-----------------|
| `cpu` | `jax[cpu]` | `pip install msd-flow[cpu]` |
| `gpu` | `jax[cuda12]` | `pip install msd-flow[gpu]` |
| `docs` | `mkdocs-material`, `mkdocs-autorefs` | `pip install msd-flow[docs]` |
| `dev` | `pytest`, `pytest-cov` | `pip install msd-flow[dev]` |

JAX is kept out of core deps because the CPU and GPU variants are mutually exclusive pip extras. Users must install one of `[cpu]` or `[gpu]`.

### Entry Point

```toml
[project.scripts]
msd-flow-train = "msdflow.train_model:main"
```

This exposes `train_model.py` as a CLI entry point (requires moving or exposing it via the package).

Actually — `train_model.py` lives at the repo root, not inside `msdflow/`. It should stay there for Hydra's working-directory behavior. No entry point script is needed; users run it via `python train_model.py` as documented.

### Build System

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["msdflow*"]
```

---

## 3. Documentation Structure

**Format:** MkDocs with the Material theme.
**Location:** `docs/` directory (shared with existing `docs/superpowers/` content — MkDocs will only serve the pages listed in `mkdocs.yml`).

### `mkdocs.yml`

```yaml
site_name: msd-flow
theme:
  name: material
  features:
    - navigation.sections
    - content.code.copy

nav:
  - Home: index.md
  - Installation: installation.md
  - Configuration: configuration.md
  - Training: training.md
  - Implementing Metrics: metrics.md
  - TODO: todo.md
```

### Page Content Outline

**`docs/index.md` — Overview**
- What msd-flow is (flow matching for galaxy images)
- Quick-start snippet (install + run training)
- Link to each doc section

**`docs/installation.md` — Installation**
- Prerequisites (Python ≥ 3.10, conda optional)
- Install with CPU JAX: `pip install msd-flow[cpu]`
- Install with GPU JAX: `pip install msd-flow[gpu]`
- Install docs dependencies: `pip install msd-flow[docs]`
- How to build and serve docs locally: `mkdocs serve`
- Environment variable setup (`TNG_API_KEY`)

**`docs/configuration.md` — Configuration**
- Hydra config system overview (defaults list, overrides, `_target_` instantiation)
- Annotated walkthrough of every config group:
  - `config.yaml` (top-level keys: seed, image_size, metadata_columns, work_dir)
  - `data/download_tng50.yaml`
  - `data/dataset.yaml`
  - `data/dataloader.yaml` (image preprocessing pipeline)
  - `model/ncsnpp.yaml` and `model/unet.yaml`
  - `flow/sample.yaml`
  - `train/train.yaml`
  - `clearml/clearml.yaml`
- How to override from CLI: `python train_model.py train.num_epochs=200 model=unet`
- How to create a custom config group override

**`docs/training.md` — Training Entry Point**
- What `train_model.py` does, step by step (dataset resolution → dataloader → seed → model → training loop)
- How the training loop works (EMA, validation, checkpointing, ClearML logging)
- Common CLI override examples
- Checkpoint format and how to resume/load a model

**`docs/metrics.md` — Implementing New Metrics**
- The two metric types:
  - **Batch metrics** `(model, x_t, u_t, t, cond, cond_mask) -> scalar` — called each validation batch
  - **Epoch metrics** `(model, val_batches, key) -> scalar` — called once per validation cycle over raw data
- Annotated example: implementing a custom batch metric
- Annotated example: implementing an epoch metric (e.g., FID-style metric that needs to generate samples)
- How to register a metric in `configs/train/train.yaml` via `_target_` and `_partial_`
- Notes on dependencies (bake them in via `_partial_: true` in config)

**`docs/todo.md` — TODO**
- Implement submitit-launcher plugin for running on SLURM-based clusters
- Implement generative metrics
- Implement physical property based metrics
- Implement early stopping based on chosen metric
- Implement tracking of best model based on same metric as early stopping
- Add image sampling function in default training config
- Implement inverse image transforms

---

## 4. File Changes Summary

| Action | File(s) |
|--------|---------|
| Rename directory | `src/` → `msdflow/` |
| Update imports | All `*.py` files referencing `from src.` or `import src.` |
| Update `_target_` strings | All `configs/**/*.yaml` with `src.` prefixes |
| Create | `pyproject.toml` |
| Create | `mkdocs.yml` |
| Create | `docs/index.md` |
| Create | `docs/installation.md` |
| Create | `docs/configuration.md` |
| Create | `docs/training.md` |
| Create | `docs/metrics.md` |
| Create | `docs/todo.md` |
| Update | `.claude/CLAUDE.md` (import paths in examples) |

---

## 5. Out of Scope

- Hosting documentation (GitHub Pages, ReadTheDocs) — not currently available
- Auto-generated API reference (no `mkdocstrings` integration)
- Restructuring the Hydra config layout
- Any changes to training logic, model architecture, or data pipeline
