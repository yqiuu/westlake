import numpy as np
import torch
from torch import nn


class LinearInterpolation(nn.Module):
    """Linear interpolation of a key tensor.

    Input tensor should be (N,) or (N, 1).

    Args:
        x_node (tensor): (N,). x data.
        y_node (tensor): (N, X). y_data.
        boundary (str): Extrapolation method.
            - 'fixed': Use values at bounds.
            - 'extrapolate': Linear extrapolation.
    """
    def __init__(self, x_node, y_node, boundary="fixed"):
        super(LinearInterpolation, self).__init__()
        self.register_buffer("x_node", x_node)
        self.register_buffer("y_node", y_node)
        options = ("fixed", "extrapolate")
        if boundary not in options:
            msg = ", ".join([f"'{opn}'" for opn in options])
            raise ValueError(f"Choose boundary from ({msg}).")

    def forward(self, x_in, *args):
        # x_in (B,)
        x_in = x_in.ravel()
        x_in = x_in.clamp_min(self.x_node[0])
        x_in = x_in.clamp_max(self.x_node[-1])

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

        yn_0 = self.y_node[inds_0]
        yn_1 = self.y_node[inds_1]
        y_out = yn_0*(1 - x_local) + yn_1*x_local
        return y_out


class TensorDict(nn.Module):
    """A module that returns a dict of tensors."""
    def __init__(self, names, tensors):
        super(TensorDict, self).__init__()
        for key, val in zip(names, tensors):
            self.register_buffer(key, val)
        self.names = names

    def add(self, key, tensor):
        self.register_buffer(key, tensor)
        self.names.append(key)

    def forward(self, *args, **kwargs):
        return {key: getattr(self, key) for key in self.names}

    def indexing(self, inds):
        return TensorDict(self.names, [val[inds].clone() for val in self().values()])


def data_frame_to_tensor_dict(df):
    """Create a TensorDict from a pandas dataframe."""
    names = list(df.columns.values)
    tensors = []
    dtype_float = torch.get_default_dtype()
    for key in names:
        arr = df[key].values
        if np.issubdtype(arr.dtype, np.floating):
            dtype = dtype_float
        else:
            dtype = None
        tensors.append(torch.tensor(arr, dtype=dtype))
    return TensorDict(names, tensors)