import jax.numpy as jnp
from omegaconf import OmegaConf

_SUPPORTED_JNP_DTYPES = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
}


def register_all_resolvers():
    """Registers all custom Hydra resolvers for the project."""
    if not OmegaConf.has_resolver("if_cond"):

        def if_cond(metadata_columns, true_val, false_val):
            if metadata_columns is not None:
                return true_val
            else:
                return false_val

        OmegaConf.register_new_resolver(
            "if_cond",
            if_cond,
        )

    if not OmegaConf.has_resolver("generate_snapshot_ids"):

        def generate_snapshot_ids(start, count):
            return [start + i for i in range(count)]

        OmegaConf.register_new_resolver(
            "generate_snapshot_ids",
            generate_snapshot_ids,
        )

    if not OmegaConf.has_resolver("jnp_dtype"):

        def jnp_dtype(name):
            if name not in _SUPPORTED_JNP_DTYPES:
                raise ValueError(
                    f"Unsupported jnp_dtype {name!r}. "
                    f"Supported: {sorted(_SUPPORTED_JNP_DTYPES)}."
                )
            return _SUPPORTED_JNP_DTYPES[name]

        OmegaConf.register_new_resolver(
            "jnp_dtype",
            jnp_dtype,
        )
