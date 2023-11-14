import pickle
from os import path

import torch
from torch import nn

from .utils import LinearInterpolation


def load_H2_shielding_data():
    fname = path.join(path.dirname(path.abspath(__file__)), "data", "H2_shielding.pickle")
    return pickle.load(open(fname, "rb"))


class H2Shielding(nn.Module):
    def __init__(self, idx_H2, meta_params):
        super().__init__()
        data = load_H2_shielding_data()
        x_H2 = torch.as_tensor(data["x_H2"], dtype=torch.get_default_dtype())
        factor = torch.as_tensor(data["factor"], dtype=torch.get_default_dtype())
        factor = 2.54e-11*factor[:, None]
        self.interp = LinearInterpolation(x_H2, factor)
        self.register_buffer("den_Av_ratio_0", torch.tensor(meta_params.den_Av_ratio_0))
        self.idx_H2 = slice(idx_H2, idx_H2 + 1)

    def forward(self, params_med, params_reac, y_in, **kwargs):
        y_in = torch.atleast_2d(y_in)
        den_H2 = params_med["Av"]*self.den_Av_ratio_0*y_in[:, self.idx_H2]
        return self.interp(den_H2)