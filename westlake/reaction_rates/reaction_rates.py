import torch
from torch import nn


class ReactionRate(nn.Module):
    """Base class for all modules to compute reaction rates.

    This is essentially a `nn.Module` with an input to specify the required
    properties for computing the reaction rates.

    Args:
        required_props (list | None): A list of property names for computing
            the reaction rates. Use None when no property is needed. Defaults
            to None.
    """
    def __init__(self, required_props=None):
        super().__init__()
        if required_props is None:
            required_props = []
        self._required_props = required_props

    @property
    def required_props(self):
        return list(self._required_props)


class NoReaction(ReactionRate):
    def forward(self, params_med, params_reac, **params_extra):
        return torch.zeros_like(params_reac["alpha"])
