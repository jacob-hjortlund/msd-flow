from omegaconf import OmegaConf


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
