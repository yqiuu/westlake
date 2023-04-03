from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class KeyTensor:
    """A tensor whose last dimension can be accessed by a list of keys."""
    def __init__(self, keys, data):
        self._keys = tuple(keys)
        self._lookup = {key: idx for idx, key in enumerate(keys)}
        self._data = data

    def get(self, keys=None):
        """Access the data by keys."""
        if keys is None:
            return self._data

        if isinstance(keys, str):
            return self._data[..., self._lookup[keys]]

        inds = [self._lookup[key] for key in keys]
        return self._data[..., inds]

    def new(self, data):
        """Create a new key tensor using the same keys but new data."""
        return KeyTensor(self._keys, data)

    def register_buffer(self, module, name):
        name_ = f"{name}_"
        module.register_buffer(name_, self._data)
        setattr(module, name, KeyTensor(self._lookup, getattr(module, name_)))


class LinearInterp(nn.Module):
    """Linear interpolation of a key tensor.

    Args:
        x_node (tensor): (N,). x data.
        y_node (KeyTensor): (N, X). y_data.
    """
    def __init__(self, x_node, y_node):
        super(LinearInterp, self).__init__()
        self.register_buffer("x_node", x_node)
        y_node.register_buffer(self, "y_node")

    def forward(self, x_in):
        # x_in (B,)
        x_in = x_in.ravel()

        x_node = self.x_node
        n_bin = x_node.shape[0] - 1
        inds_1 = torch.searchsorted(x_node, x_in.contiguous())
        inds_1[inds_1 > n_bin] = n_bin
        inds_0 = inds_1 - 1
        inds_0[inds_0 < 0] = 1

        xn_0 = x_node[inds_0]
        xn_1 = x_node[inds_1]
        x_local = (x_in - xn_0)/(xn_1 - xn_0)
        x_local = x_local[:, None]

        yn_0 = self.y_node.get()[inds_0]
        yn_1 = self.y_node.get()[inds_1]
        y_out = yn_0*(1 - x_local) + yn_1*x_local
        y_out = self.y_node.new(y_out)

        return y_out


def data_frame_to_key_tensor(df, **kwargs):
    """Create a key tensor from a pandas dataframe."""
    return KeyTensor(df.columns.values.astype(str), torch.tensor(df.values, **kwargs))
