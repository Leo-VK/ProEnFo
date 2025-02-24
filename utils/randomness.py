import random

import numpy as np
import torch


def set_random_seeds(seed: int = 0):
    """Set all relevant random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
