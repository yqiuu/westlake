import math

import torch

from .constants import FACTOR_VIB_FREQ


def compute_vibration_frequency(E_d, mass, meta_params):
    return torch.sqrt(FACTOR_VIB_FREQ*meta_params.site_density*E_d/mass)
