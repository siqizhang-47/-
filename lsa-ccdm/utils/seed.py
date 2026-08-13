"""Single global random seed for the whole pipeline.

Requirement: exactly ONE seed is used everywhere (data pipeline, backbone
training, deployment sampling, adapters, tests). Call ``set_seed`` once at
process start; never seed anything else anywhere in the repo.
"""
import os
import random

import numpy as np
import torch

GLOBAL_SEED = 2026


def set_seed(seed: int = GLOBAL_SEED):
    """Set the single global seed for python / numpy / torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
