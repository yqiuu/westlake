import torch
from torch import nn


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


class TensorDict(nn.Module):
    """A module that returns a dict of tensors."""
    def __init__(self, names, tensors):
        super(TensorDict, self).__init__()
        for key, val in zip(names, tensors):
            self.register_buffer(key, val)
        self.names = names

    def forward(self, *args, **kwargs):
        return {key: getattr(self, key) for key in self.names}


def data_frame_to_tensor_dict(df, **kwargs):
    """Create a dict of tensors from a pandas dataframe."""
    names = df.columns.values
    return TensorDict(names, [torch.tensor(df[key].values, **kwargs) for key in names])