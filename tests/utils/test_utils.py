"""Tests for src.utils.utils."""

import pytest

from src.utils.utils import resolve_import


def test_resolve_import_standard_library():
    """Verify a standard-library function can be resolved."""
    func = resolve_import("os.path.join")
    import os.path
    assert func is os.path.join


def test_resolve_import_jax_activation():
    """Verify a JAX activation function can be resolved."""
    func = resolve_import("jax.nn.silu")
    import jax.nn
    assert func is jax.nn.silu


def test_resolve_import_invalid_module_raises():
    """Verify ImportError is raised for a nonexistent module."""
    with pytest.raises((ImportError, ModuleNotFoundError)):
        resolve_import("nonexistent_module_xyz.foo")


def test_resolve_import_invalid_attr_raises():
    """Verify AttributeError is raised for a nonexistent attribute."""
    with pytest.raises(AttributeError):
        resolve_import("os.path.nonexistent_attr_xyz")


def test_resolve_import_returns_callable():
    """Verify the resolved object is callable."""
    func = resolve_import("math.sqrt")
    assert callable(func)
