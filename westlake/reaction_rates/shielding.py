import pickle
from pathlib import Path

import torch
from torch import nn

from ..utils import LinearInterpolation


def load_H2_shielding_data():
    fname = Path(__file__).parent.parent/Path("data")/Path("H2_shielding_Lee+1996.pickle")
    return pickle.load(open(fname, "rb"))


def load_CO_shielding_data():
    fname = Path(__file__).parent.parent/Path("data")/Path("CO_shielding_Lee+1996.pickle")
    return pickle.load(open(fname, "rb"))


class H2Shielding_Lee1996(nn.Module):
    def __init__(self, idx_H2, config):
        super().__init__()
        data = load_H2_shielding_data()
        x_H2 = torch.as_tensor(data["x_H2"], dtype=torch.get_default_dtype())
        factor = torch.as_tensor(data["factor"], dtype=torch.get_default_dtype())
        factor = 2.54e-11*factor[:, None]
        self.interp = LinearInterpolation(x_H2, factor)
        self.register_buffer("uv_flux", torch.tensor(config.uv_flux))
        self.register_buffer("den_Av_ratio", torch.tensor(config.den_Av_ratio))
        self.idx_H2 = slice(idx_H2, idx_H2 + 1)

    def forward(self, params_med, params_reac, y_in, **kwargs):
        y_in = torch.atleast_2d(y_in)
        den_H2 = params_med["Av"]*self.den_Av_ratio*y_in[:, self.idx_H2]
        return self.uv_flux*self.interp(den_H2)


class COShielding_Lee1996(nn.Module):
    def __init__(self, idx_CO, idx_H2, config):
        super().__init__()
        data = load_CO_shielding_data()
        self.interp_CO = self._create_interp(data, "CO")
        self.interp_H2 = self._create_interp(data, "H2")
        self.interp_Av = self._create_interp(data, "Av")
        self.register_buffer("uv_flux", torch.tensor(config.uv_flux))
        self.register_buffer("den_Av_ratio", torch.tensor(config.den_Av_ratio))
        self.idx_CO = slice(idx_CO, idx_CO + 1)
        self.idx_H2 = slice(idx_H2, idx_H2 + 1)

    def _create_interp(self, data, name):
        x_node = torch.as_tensor(data[f"x_{name}"], dtype=torch.get_default_dtype())
        y_node = torch.as_tensor(data[f"theta_{name}"], dtype=torch.get_default_dtype())
        return LinearInterpolation(x_node, y_node[:, None])

    def forward(self, params_med, params_reac, y_in, **kwargs):
        y_in = torch.atleast_2d(y_in)
        factor = params_med["Av"]*self.den_Av_ratio
        den_CO = factor*y_in[:, self.idx_CO]
        den_H2 = factor*y_in[:, self.idx_H2]
        return 1.03e-10 * self.interp_CO(den_CO) \
            * self.interp_H2(den_H2) \
            * self.interp_Av(params_med["Av"]) \
            * self.uv_flux