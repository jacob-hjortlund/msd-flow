"""Tests for resumable training checkpoint helpers."""

import json
import os
import signal
from pathlib import Path

import equinox as eqx
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import optax
import pytest
from omegaconf import OmegaConf

from msdflow.train.checkpointing import (
    compute_config_hash,
    checkpoint_run_dir,
    checkpoint_filename_stem,
    TrainingCheckpoint,
    latest_pointer_path,
    build_checkpoint_metadata,
    discover_latest_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
    SigtermFlag,
    validate_checkpoint_metadata,
)
from msdflow.train.trainer import make_train_state


def test_train_config_contains_resume_defaults():
    """Hydra train config should expose resume defaults."""
    cfg = OmegaConf.load("configs/train/train.yaml")

    assert cfg.resume.restart is False
    assert cfg.resume.auto is True
    assert cfg.resume.hash is None
    assert "clearml" in list(cfg.resume.hash_exclude)
    assert "train.resume" in list(cfg.resume.hash_exclude)
    assert "train.num_epochs" in list(cfg.resume.hash_exclude)
    assert "train.time_loss_diagnostic" in list(cfg.resume.hash_exclude)
    assert cfg.resume.latest_filename == "latest.json"
    assert cfg.resume.save_on_sigterm is True


def test_train_config_enables_time_loss_diagnostic_by_default():
    """Default train config should enable validation time-binned loss logging."""
    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()), version_base=None
    ):
        cfg = compose(
            config_name="config", overrides=["train.num_train_eval_batches=7"]
        )

    assert cfg.train.trainer.time_loss_diagnostic.enabled is True
    assert cfg.train.trainer.time_loss_diagnostic.split == "val"
    assert cfg.train.trainer.time_loss_diagnostic.num_bins == 20
    assert cfg.train.trainer.num_train_eval_batches == 7
    assert cfg.train.trainer.time_loss_diagnostic.num_batches == 7
    assert cfg.train.trainer.time_loss_diagnostic.log_heatmap is True


def test_train_config_exposes_clr_defaults():
    """Default train config should expose opt-in CLR controls as no-ops."""
    cfg = OmegaConf.load("configs/train/train.yaml")

    assert cfg.loss_fn.project_velocity is False
    assert cfg.batch_metrics[0].project_velocity is False
    assert cfg.x0_mode == "gaussian"
    assert cfg.project_velocity is False
    assert cfg["_epoch_metrics_dict"].fid_metric.generate_fn.x0_mode == "gaussian"
    assert cfg["_epoch_metrics_dict"].fid_metric.generate_fn.project_velocity is False
    assert cfg.sample_fn.x0_mode == "gaussian"
    assert cfg.sample_fn.project_velocity is False


def test_run_sh_uses_stable_checkpoint_root():
    """NERSC run script should not put resumable checkpoints in per-job RUN_DIR."""
    text = open("run.sh").read()

    assert "#SBATCH --signal=TERM@600" in text
    assert 'export CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"' in text
    assert '"$CHECKPOINT_ROOT"' in text
    assert '"train.checkpoint_dir=${CHECKPOINT_ROOT}"' in text
    assert '"train.resume.restart=false"' in text
    assert '"train.resume.save_on_sigterm=true"' in text


def test_compute_config_hash_ignores_excluded_paths():
    """Excluded config paths must not affect the stable hash."""
    cfg_a = OmegaConf.create(
        {
            "clearml": {"task_name": "job-a"},
            "train": {
                "num_epochs": 100,
                "resume": {"restart": False},
                "optimizer": {"learning_rate": 1.0e-4},
            },
            "model": {"base_channels": 64},
        }
    )
    cfg_b = OmegaConf.create(
        {
            "clearml": {"task_name": "job-b"},
            "train": {
                "num_epochs": 200,
                "resume": {"restart": True},
                "optimizer": {"learning_rate": 1.0e-4},
            },
            "model": {"base_channels": 64},
        }
    )

    hash_a, payload_a = compute_config_hash(
        cfg_a,
        exclude_paths=["clearml", "train.resume", "train.num_epochs"],
    )
    hash_b, payload_b = compute_config_hash(
        cfg_b,
        exclude_paths=["clearml", "train.resume", "train.num_epochs"],
    )

    assert hash_a == hash_b
    assert payload_a == payload_b
    assert "clearml" not in payload_a
    assert "resume" not in payload_a["train"]
    assert "num_epochs" not in payload_a["train"]


def test_composed_config_hash_ignores_time_loss_diagnostic_defaults():
    """Configured hash excludes should keep diagnostics observational."""
    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()),
        version_base=None,
    ):
        cfg_a = compose(
            config_name="config",
            overrides=["work_dir=/tmp/msdflow-test"],
        )
        cfg_b = compose(
            config_name="config",
            overrides=["work_dir=/tmp/msdflow-test"],
        )

    cfg_b.train.time_loss_diagnostic.enabled = (
        not cfg_a.train.time_loss_diagnostic.enabled
    )
    cfg_b.train.time_loss_diagnostic.num_bins = (
        cfg_a.train.time_loss_diagnostic.num_bins + 1
    )
    cfg_b.train.time_loss_diagnostic.log_heatmap = (
        not cfg_a.train.time_loss_diagnostic.log_heatmap
    )

    hash_a, payload_a = compute_config_hash(
        cfg_a,
        exclude_paths=list(cfg_a.train.resume.hash_exclude),
    )
    hash_b, payload_b = compute_config_hash(
        cfg_b,
        exclude_paths=list(cfg_b.train.resume.hash_exclude),
    )

    assert hash_a == hash_b
    assert payload_a == payload_b
    assert "time_loss_diagnostic" not in payload_a["train"]


def test_composed_config_hash_changes_for_clr_defaults():
    """CLR defaults should be compatibility-relevant in stable hashes."""
    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()),
        version_base=None,
    ):
        cfg_with_defaults = compose(
            config_name="config",
            overrides=["work_dir=/tmp/msdflow-test"],
        )
        cfg_without_defaults = compose(
            config_name="config",
            overrides=["work_dir=/tmp/msdflow-test"],
        )

    without_payload = OmegaConf.to_container(cfg_without_defaults, resolve=True)
    without_train = without_payload["train"]
    without_train["loss_fn"].pop("project_velocity")
    without_train["batch_metrics"][0].pop("project_velocity")
    without_train["_epoch_metrics_dict"]["fid_metric"]["generate_fn"].pop("x0_mode")
    without_train["_epoch_metrics_dict"]["fid_metric"]["generate_fn"].pop(
        "project_velocity"
    )
    without_train["epoch_metrics"][0]["generate_fn"].pop("x0_mode")
    without_train["epoch_metrics"][0]["generate_fn"].pop("project_velocity")
    without_train.pop("x0_mode")
    without_train.pop("project_velocity")
    without_train["sample_fn"].pop("x0_mode")
    without_train["sample_fn"].pop("project_velocity")
    cfg_without_defaults = OmegaConf.create(
        without_payload,
        flags={"allow_objects": True},
    )

    hash_with, payload_with = compute_config_hash(
        cfg_with_defaults,
        exclude_paths=list(cfg_with_defaults.train.resume.hash_exclude),
    )
    hash_without, payload_without = compute_config_hash(
        cfg_without_defaults,
        exclude_paths=list(cfg_without_defaults.train.resume.hash_exclude),
    )

    assert hash_with != hash_without
    assert payload_with != payload_without
    assert payload_with["train"]["x0_mode"] == "gaussian"
    assert payload_with["train"]["project_velocity"] is False
    assert payload_with["train"]["loss_fn"]["project_velocity"] is False
    assert payload_with["train"]["batch_metrics"][0]["project_velocity"] is False
    assert payload_with["train"]["sample_fn"]["x0_mode"] == "gaussian"
    assert payload_with["train"]["sample_fn"]["project_velocity"] is False
    assert (
        payload_with["train"]["epoch_metrics"][0]["generate_fn"]["x0_mode"]
        == "gaussian"
    )
    assert (
        payload_with["train"]["epoch_metrics"][0]["generate_fn"]["project_velocity"]
        is False
    )


def test_composed_config_hash_changes_for_clr_opt_in():
    """CLR opt-ins should remain compatibility-relevant in stable hashes."""
    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()),
        version_base=None,
    ):
        cfg_default = compose(
            config_name="config",
            overrides=["work_dir=/tmp/msdflow-test"],
        )
        cfg_clr = compose(
            config_name="config",
            overrides=[
                "work_dir=/tmp/msdflow-test",
                "train.x0_mode=clr",
                "train.project_velocity=true",
            ],
        )

    hash_default, payload_default = compute_config_hash(
        cfg_default,
        exclude_paths=list(cfg_default.train.resume.hash_exclude),
    )
    hash_clr, payload_clr = compute_config_hash(
        cfg_clr,
        exclude_paths=list(cfg_clr.train.resume.hash_exclude),
    )

    assert hash_default != hash_clr
    assert payload_default["train"]["x0_mode"] == "gaussian"
    assert payload_default["train"]["project_velocity"] is False
    assert payload_clr["train"]["x0_mode"] == "clr"
    assert payload_clr["train"]["project_velocity"] is True
    assert payload_clr["train"]["loss_fn"]["project_velocity"] is True
    assert payload_clr["train"]["batch_metrics"][0]["project_velocity"] is True
    assert payload_clr["train"]["epoch_metrics"][0]["generate_fn"]["x0_mode"] == "clr"
    assert (
        payload_clr["train"]["epoch_metrics"][0]["generate_fn"]["project_velocity"]
        is True
    )
    assert payload_clr["train"]["sample_fn"]["x0_mode"] == "clr"
    assert payload_clr["train"]["sample_fn"]["project_velocity"] is True


def test_compute_config_hash_changes_for_model_fields():
    """Model compatibility fields must affect the stable hash."""
    cfg_a = OmegaConf.create({"model": {"base_channels": 64}, "train": {}})
    cfg_b = OmegaConf.create({"model": {"base_channels": 128}, "train": {}})

    hash_a, _ = compute_config_hash(cfg_a, exclude_paths=[])
    hash_b, _ = compute_config_hash(cfg_b, exclude_paths=[])

    assert hash_a != hash_b


def test_compute_config_hash_accepts_length_keyword():
    """Hash length should be configurable with the planned keyword."""
    cfg = OmegaConf.create({"model": {"base_channels": 64}, "train": {}})

    stable_hash, _ = compute_config_hash(cfg, exclude_paths=[], length=8)

    assert len(stable_hash) == 8


def test_checkpoint_run_dir_joins_root_and_hash(tmp_path):
    """Hash-specific checkpoint directory should be deterministic."""
    assert checkpoint_run_dir(str(tmp_path), "abc123") == str(tmp_path / "abc123")


def test_checkpoint_run_dir_accepts_root_keyword(tmp_path):
    """Hash-specific checkpoint directory should accept the planned keyword."""
    result = checkpoint_run_dir(root=str(tmp_path), stable_hash="abc123")

    assert result == str(tmp_path / "abc123")


def test_latest_pointer_path_uses_configured_filename(tmp_path):
    """Latest pointer path should live inside the hash-specific directory."""
    run_dir = tmp_path / "hash"

    result = latest_pointer_path(str(run_dir), "latest.json")

    assert result == str(run_dir / "latest.json")


def test_discover_latest_checkpoint_returns_none_when_restart_true(tmp_path):
    """restart=true must force a fresh run even when a pointer exists."""
    run_dir = tmp_path / "hash"
    run_dir.mkdir()
    (run_dir / "latest.json").write_text(json.dumps({"metadata_path": "x.json"}))

    result = discover_latest_checkpoint(
        str(run_dir),
        latest_filename="latest.json",
        restart=True,
    )

    assert result is None


def test_discover_latest_checkpoint_returns_none_when_pointer_absent(tmp_path):
    """restart=false should return None when no latest pointer exists."""
    run_dir = tmp_path / "hash"
    run_dir.mkdir()

    result = discover_latest_checkpoint(
        str(run_dir),
        latest_filename="latest.json",
        restart=False,
    )

    assert result is None


@pytest.mark.parametrize("payload_path", ["", None])
def test_discover_latest_checkpoint_rejects_invalid_payload_path(
    tmp_path,
    payload_path,
):
    """Metadata payload_path must be a non-empty string path."""
    run_dir = tmp_path / "hash"
    run_dir.mkdir()
    metadata_path = run_dir / "checkpoint_epoch0001_step0000.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stable_hash": "hash",
                "checkpoint_kind": "periodic",
                "epoch": 1,
                "completed_microsteps": 0,
                "payload_path": payload_path,
                "clearml_task_id": "task-1",
            }
        )
    )
    (run_dir / "latest.json").write_text(
        json.dumps({"metadata_path": str(metadata_path)})
    )

    with pytest.raises(ValueError, match="payload_path"):
        discover_latest_checkpoint(
            str(run_dir),
            latest_filename="latest.json",
            restart=False,
        )


def test_discover_latest_checkpoint_loads_pointer_metadata(tmp_path):
    """restart=false should load metadata referenced by latest.json."""
    run_dir = tmp_path / "hash"
    run_dir.mkdir()
    metadata_path = run_dir / "checkpoint_epoch0001_step0000.json"
    payload_path = run_dir / "checkpoint_epoch0001_step0000.eqx"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stable_hash": "hash",
                "checkpoint_kind": "periodic",
                "epoch": 1,
                "completed_microsteps": 0,
                "payload_path": str(payload_path),
                "clearml_task_id": "task-1",
            }
        )
    )
    (run_dir / "latest.json").write_text(
        json.dumps({"metadata_path": str(metadata_path)})
    )

    result = discover_latest_checkpoint(
        str(run_dir),
        latest_filename="latest.json",
        restart=False,
    )

    assert result is not None
    assert result["metadata_path"] == str(metadata_path)
    assert result["payload_path"] == str(payload_path)
    assert result["clearml_task_id"] == "task-1"


def _linear_checkpoint_payload():
    """Build a small checkpoint payload for serialization tests."""
    model = eqx.nn.Linear(2, 1, key=jax.random.PRNGKey(0))
    optimizer = optax.sgd(0.1)
    state = make_train_state(model, optimizer)
    return TrainingCheckpoint(
        state=state,
        ema_model=model,
        ema_initialized=True,
        key=jax.random.PRNGKey(1),
        sampling_key=jax.random.PRNGKey(2),
        epoch=3,
        completed_microsteps=5,
        epoch_loss=7.5,
        best_metric_value=0.25,
        best_epoch=2,
        patience_counter=1,
        total_epoch_time=11.0,
        total_train_time=9.0,
        total_val_time=3.0,
        val_runs=2,
        val_time=1.5,
        val_metrics={"flow_matching_loss": 0.25},
        train_metrics={"flow_matching_loss": 0.5},
        epoch_metric_results={"fid_zoobot": 10.0},
    )


def test_training_checkpoint_payload_holds_full_state():
    """TrainingCheckpoint should hold all train-resume state fields."""
    checkpoint = _linear_checkpoint_payload()

    assert checkpoint.state.model is checkpoint.ema_model
    assert checkpoint.ema_initialized is True
    assert jnp.array_equal(checkpoint.key, jax.random.PRNGKey(1))
    assert jnp.array_equal(checkpoint.sampling_key, jax.random.PRNGKey(2))
    assert checkpoint.epoch == 3
    assert checkpoint.completed_microsteps == 5
    assert checkpoint.epoch_loss == 7.5
    assert checkpoint.best_metric_value == 0.25
    assert checkpoint.best_epoch == 2
    assert checkpoint.patience_counter == 1
    assert checkpoint.total_epoch_time == 11.0
    assert checkpoint.total_train_time == 9.0
    assert checkpoint.total_val_time == 3.0
    assert checkpoint.val_runs == 2
    assert checkpoint.val_time == 1.5
    assert checkpoint.val_metrics == {"flow_matching_loss": 0.25}
    assert checkpoint.train_metrics == {"flow_matching_loss": 0.5}
    assert checkpoint.epoch_metric_results == {"fid_zoobot": 10.0}


def test_checkpoint_filename_stem_uses_epoch_and_microstep():
    """Checkpoint filenames should use one-based epoch and padded microsteps."""
    assert (
        checkpoint_filename_stem(epoch=6, completed_microsteps=42)
        == "checkpoint_epoch0007_step0042"
    )


def test_build_checkpoint_metadata_returns_json_safe_required_fields(tmp_path):
    """Metadata should include all required JSON-safe checkpoint fields."""
    payload_path = tmp_path / "checkpoint_epoch0007_step0042.eqx"
    metadata = build_checkpoint_metadata(
        stable_hash="abc123",
        checkpoint_kind="periodic",
        epoch=6,
        completed_microsteps=42,
        payload_path=str(payload_path),
        grad_accum_steps=2,
        microsteps_per_epoch=64,
        monitor="flow_matching_loss",
        monitor_mode="min",
        clearml_task_id="task-1",
        source_checkpoint_path="/source/checkpoint.eqx",
        hash_payload={"train": {"optimizer": "sgd"}},
        ema_initialized=True,
        best_metric_value=float("nan"),
        best_epoch=5,
    )

    json.dumps(metadata)
    assert metadata["schema_version"] == 1
    assert metadata["stable_hash"] == "abc123"
    assert metadata["checkpoint_kind"] == "periodic"
    assert metadata["epoch"] == 6
    assert metadata["completed_microsteps"] == 42
    assert metadata["payload_path"] == str(payload_path)
    assert metadata["grad_accum_steps"] == 2
    assert metadata["microsteps_per_epoch"] == 64
    assert metadata["monitor"] == "flow_matching_loss"
    assert metadata["monitor_mode"] == "min"
    assert metadata["clearml_task_id"] == "task-1"
    assert metadata["source_checkpoint_path"] == "/source/checkpoint.eqx"
    assert metadata["hash_payload"] == {"train": {"optimizer": "sgd"}}
    assert metadata["ema_initialized"] is True
    assert metadata["best_metric_value"] == "nan"
    assert metadata["best_epoch"] == 5
    assert isinstance(metadata["saved_at_unix"], float)


def test_build_checkpoint_metadata_accepts_checkpoint_payload_context(tmp_path):
    """Metadata builder should accept checkpoint payload context directly."""
    checkpoint = _linear_checkpoint_payload()
    payload_path = tmp_path / "checkpoint_epoch0004_step0005.eqx"

    metadata = build_checkpoint_metadata(
        stable_hash="abc123",
        checkpoint_kind="manual",
        checkpoint=checkpoint,
        payload_path=str(payload_path),
        grad_accum_steps=1,
        microsteps_per_epoch=8,
        monitor="flow_matching_loss",
        monitor_mode="min",
        clearml_task_id=None,
        source_checkpoint_path=None,
        hash_payload={"train": {"optimizer": "sgd"}},
    )

    assert metadata["epoch"] == checkpoint.epoch
    assert metadata["completed_microsteps"] == checkpoint.completed_microsteps
    assert metadata["ema_initialized"] == checkpoint.ema_initialized
    assert metadata["best_metric_value"] == checkpoint.best_metric_value
    assert metadata["best_epoch"] == checkpoint.best_epoch


def test_validate_checkpoint_metadata_rejects_wrong_stable_hash(tmp_path):
    """Checkpoint metadata should reject incompatible stable hashes."""
    metadata = build_checkpoint_metadata(
        stable_hash="abc123",
        checkpoint_kind="periodic",
        epoch=0,
        completed_microsteps=0,
        payload_path=str(tmp_path / "checkpoint.eqx"),
        grad_accum_steps=1,
        microsteps_per_epoch=8,
        monitor="flow_matching_loss",
        monitor_mode="min",
        clearml_task_id=None,
        source_checkpoint_path=None,
        hash_payload={"train": {"optimizer": "sgd"}},
        ema_initialized=True,
        best_metric_value=0.25,
        best_epoch=0,
    )

    with pytest.raises(ValueError, match="stable hash"):
        validate_checkpoint_metadata(
            metadata,
            stable_hash="different",
            monitor="flow_matching_loss",
            monitor_mode="min",
            allow_hash_override=False,
        )


def test_validate_checkpoint_metadata_accepts_expected_hash_alias():
    """Validation should support the approved-plan expected_hash keyword."""
    metadata = {
        "schema_version": 1,
        "stable_hash": "abc123",
        "checkpoint_kind": "periodic",
        "epoch": 1,
        "completed_microsteps": 0,
        "payload_path": "/tmp/checkpoint.eqx",
        "monitor": "flow_matching_loss",
        "monitor_mode": "min",
    }

    with pytest.raises(ValueError, match="stable hash"):
        validate_checkpoint_metadata(
            metadata,
            expected_hash="different",
            monitor="flow_matching_loss",
            monitor_mode="min",
            microsteps_per_epoch=8,
            allow_hash_override=False,
        )


def test_save_training_checkpoint_writes_payload_metadata_and_latest(tmp_path):
    """Saving should write payload, metadata, and latest pointer atomically."""
    run_dir = tmp_path / "checkpoints"
    checkpoint = _linear_checkpoint_payload()

    metadata = save_training_checkpoint(
        run_dir=str(run_dir),
        checkpoint=checkpoint,
        stable_hash="abc123",
        checkpoint_kind="periodic",
        grad_accum_steps=1,
        microsteps_per_epoch=8,
        monitor="flow_matching_loss",
        monitor_mode="min",
        clearml_task_id="task-1",
        latest_filename="latest.json",
        source_checkpoint_path=None,
        hash_payload={"train": {"optimizer": "sgd"}},
    )

    assert os.path.exists(metadata["payload_path"])
    assert os.path.exists(metadata["metadata_path"])
    assert json.loads((run_dir / "latest.json").read_text()) == {
        "metadata_path": metadata["metadata_path"]
    }
    assert metadata["payload_path"].endswith("checkpoint_epoch0004_step0005.eqx")
    assert metadata["metadata_path"].endswith("checkpoint_epoch0004_step0005.json")


def test_save_with_relative_run_dir_discovers_absolute_metadata(tmp_path, monkeypatch):
    """Relative run directories should save self-contained absolute paths."""
    monkeypatch.chdir(tmp_path)
    run_dir = "relative_ckpts"
    checkpoint = _linear_checkpoint_payload()

    metadata = save_training_checkpoint(
        run_dir=run_dir,
        checkpoint=checkpoint,
        stable_hash="abc123",
        checkpoint_kind="periodic",
        grad_accum_steps=1,
        microsteps_per_epoch=8,
        monitor="flow_matching_loss",
        monitor_mode="min",
        clearml_task_id="task-1",
        latest_filename="latest.json",
        source_checkpoint_path=None,
        hash_payload={"train": {"optimizer": "sgd"}},
    )

    discovered = discover_latest_checkpoint(
        run_dir,
        latest_filename="latest.json",
        restart=False,
    )
    pointer = json.loads((tmp_path / run_dir / "latest.json").read_text())

    assert discovered is not None
    assert os.path.isabs(metadata["metadata_path"])
    assert os.path.isabs(metadata["payload_path"])
    assert pointer == {"metadata_path": metadata["metadata_path"]}
    assert discovered["metadata_path"] == metadata["metadata_path"]
    assert discovered["payload_path"] == metadata["payload_path"]
    assert not (
        tmp_path / run_dir / run_dir / "checkpoint_epoch0004_step0005.json"
    ).exists()


@pytest.mark.parametrize("latest_filename", ["../outside.json", "absolute"])
def test_save_training_checkpoint_rejects_unsafe_latest_filename(
    tmp_path,
    latest_filename,
):
    """Latest pointer filename must stay inside the checkpoint run directory."""
    run_dir = tmp_path / "checkpoints"
    if latest_filename == "absolute":
        outside = tmp_path / "outside_absolute.json"
        latest_filename = str(outside)
    else:
        outside = tmp_path / "outside.json"

    with pytest.raises(ValueError, match="latest_filename"):
        save_training_checkpoint(
            run_dir=str(run_dir),
            checkpoint=_linear_checkpoint_payload(),
            stable_hash="abc123",
            checkpoint_kind="periodic",
            grad_accum_steps=1,
            microsteps_per_epoch=8,
            monitor="flow_matching_loss",
            monitor_mode="min",
            clearml_task_id="task-1",
            latest_filename=latest_filename,
            source_checkpoint_path=None,
            hash_payload={"train": {"optimizer": "sgd"}},
        )

    assert not outside.exists()


def test_save_training_checkpoint_invalid_kind_does_not_replace_payload(tmp_path):
    """Invalid metadata should be rejected before payload replacement."""
    run_dir = tmp_path / "checkpoints"
    run_dir.mkdir()
    payload_path = run_dir / "checkpoint_epoch0004_step0005.eqx"
    payload_path.write_bytes(b"existing payload")

    with pytest.raises(ValueError, match="checkpoint kind"):
        save_training_checkpoint(
            run_dir=str(run_dir),
            checkpoint=_linear_checkpoint_payload(),
            stable_hash="abc123",
            checkpoint_kind="invalid",
            grad_accum_steps=1,
            microsteps_per_epoch=8,
            monitor="flow_matching_loss",
            monitor_mode="min",
            clearml_task_id="task-1",
            latest_filename="latest.json",
            source_checkpoint_path=None,
            hash_payload={"train": {"optimizer": "sgd"}},
        )

    assert payload_path.read_bytes() == b"existing payload"
    assert not (run_dir / "checkpoint_epoch0004_step0005.json").exists()
    assert not (run_dir / "latest.json").exists()


def test_validate_checkpoint_metadata_rejects_invalid_checkpoint_kind(tmp_path):
    """Metadata validation should reject unsupported checkpoint kinds."""
    metadata = build_checkpoint_metadata(
        stable_hash="abc123",
        checkpoint_kind="periodic",
        epoch=0,
        completed_microsteps=0,
        payload_path=str(tmp_path / "checkpoint.eqx"),
        grad_accum_steps=1,
        microsteps_per_epoch=8,
        monitor="flow_matching_loss",
        monitor_mode="min",
        clearml_task_id=None,
        source_checkpoint_path=None,
        hash_payload={"train": {"optimizer": "sgd"}},
        ema_initialized=True,
        best_metric_value=0.25,
        best_epoch=0,
    )
    metadata["checkpoint_kind"] = "invalid"

    with pytest.raises(ValueError, match="checkpoint kind"):
        validate_checkpoint_metadata(
            metadata,
            stable_hash="abc123",
            monitor="flow_matching_loss",
            monitor_mode="min",
            allow_hash_override=False,
        )


def test_load_training_checkpoint_round_trips_full_state(tmp_path):
    """Loading should restore a serialized checkpoint against a like tree."""
    run_dir = tmp_path / "checkpoints"
    checkpoint = _linear_checkpoint_payload()
    metadata = save_training_checkpoint(
        run_dir=str(run_dir),
        checkpoint=checkpoint,
        stable_hash="abc123",
        checkpoint_kind="periodic",
        grad_accum_steps=1,
        microsteps_per_epoch=8,
        monitor="flow_matching_loss",
        monitor_mode="min",
        clearml_task_id="task-1",
        latest_filename="latest.json",
        source_checkpoint_path=None,
        hash_payload={"train": {"optimizer": "sgd"}},
    )
    like = _linear_checkpoint_payload()

    restored = load_training_checkpoint(metadata["payload_path"], like)
    pointer = json.loads((run_dir / "latest.json").read_text())

    assert os.path.exists(metadata["payload_path"])
    assert os.path.exists(metadata["metadata_path"])
    assert pointer == {"metadata_path": metadata["metadata_path"]}
    assert restored.ema_initialized == checkpoint.ema_initialized
    assert restored.epoch == checkpoint.epoch
    assert restored.completed_microsteps == checkpoint.completed_microsteps
    assert restored.epoch_loss == checkpoint.epoch_loss
    assert restored.best_metric_value == checkpoint.best_metric_value
    assert restored.best_epoch == checkpoint.best_epoch
    assert restored.patience_counter == checkpoint.patience_counter
    assert restored.total_epoch_time == checkpoint.total_epoch_time
    assert restored.total_train_time == checkpoint.total_train_time
    assert restored.total_val_time == checkpoint.total_val_time
    assert restored.val_runs == checkpoint.val_runs
    assert restored.val_time == checkpoint.val_time
    assert restored.val_metrics == checkpoint.val_metrics
    assert restored.train_metrics == checkpoint.train_metrics
    assert restored.epoch_metric_results == checkpoint.epoch_metric_results
    assert jnp.array_equal(restored.state.model.weight, checkpoint.state.model.weight)
    assert jnp.array_equal(restored.ema_model.weight, checkpoint.ema_model.weight)
    assert jnp.array_equal(restored.key, checkpoint.key)
    assert jnp.array_equal(restored.sampling_key, checkpoint.sampling_key)


def test_sigterm_flag_sets_requested_without_exiting():
    """SIGTERM handler should set a flag and return."""
    flag = SigtermFlag(enabled=True)

    with flag:
        flag.handle(signal.SIGTERM, None)
        assert flag.requested is True


def test_sigterm_flag_enabled_installs_and_restores_handler(monkeypatch):
    """Enabled SIGTERM flag should install and restore the process handler."""
    getsignal_calls = []
    signal_calls = []

    def previous_handler(signum, frame):
        """Stand in for the previously installed SIGTERM handler.

        Args:
            signum: Signal number received by the process.
            frame: Current execution frame supplied by the signal module.
        """

    def fake_getsignal(signum):
        """Record signal lookup and return the previous handler.

        Args:
            signum: Signal number whose handler is being requested.

        Returns:
            The test previous handler.
        """
        getsignal_calls.append(signum)
        return previous_handler

    def fake_signal(signum, handler):
        """Record signal handler installation and restoration.

        Args:
            signum: Signal number whose handler is being changed.
            handler: Handler installed for the signal.

        Returns:
            Default signal handler placeholder.
        """
        signal_calls.append((signum, handler))
        return signal.SIG_DFL

    monkeypatch.setattr(signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(signal, "signal", fake_signal)
    flag = SigtermFlag(enabled=True)

    with flag as active_flag:
        assert active_flag is flag
        assert getsignal_calls == [signal.SIGTERM]
        assert signal_calls == [(signal.SIGTERM, flag.handle)]

    assert getsignal_calls == [signal.SIGTERM]
    assert signal_calls == [
        (signal.SIGTERM, flag.handle),
        (signal.SIGTERM, previous_handler),
    ]


def test_sigterm_flag_disabled_does_not_install_handler(monkeypatch):
    """Disabled SIGTERM flag should not call signal.signal."""
    calls = []

    def fake_signal(signum, handler):
        calls.append((signum, handler))
        return signal.SIG_DFL

    monkeypatch.setattr(signal, "signal", fake_signal)
    flag = SigtermFlag(enabled=False)

    with flag:
        assert flag.requested is False

    assert calls == []
