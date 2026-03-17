import importlib
from typing import Callable


def resolve_import(import_str: str) -> Callable:
    """
    Resolve a dot-separated import string, e.g. 'jax.nn.silu'

    Args:
        import_str (str): import string

    Returns:
        Callable: import function
    """

    module_path, attr = import_str.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), attr)
