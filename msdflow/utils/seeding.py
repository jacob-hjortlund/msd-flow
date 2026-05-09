import torch
import random

import numpy as np
import jax.random as jr


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return jr.PRNGKey(seed)
