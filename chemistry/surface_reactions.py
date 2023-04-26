import math

import torch
from torch import nn

from .constants import FACTOR_VIB_FREQ


class ThermalEvaporation(nn.Module):
    def __init__(self, meta_params):
        super(ThermalEvaporation, self).__init__()
        self.register_buffer("T_dust_0", torch.tensor(meta_params.T_dust_0))

    def forward(self, params_env, params_reac, T_gas, mask_T, E_d, freq_vib, **kwargs):
        rate = compute_evaporation_rate(params_reac["alpha"], E_d, freq_vib, self.T_dust_0)
        return rate


class CosmicRayEvaporation(nn.Module):
    def __init__(self, meta_params):
        super().__init__()
        prefactor = (meta_params.rate_cr_ion + meta_params.rate_x_ion)/1.3e-17 \
            *(meta_params.rate_fe_ion*meta_params.tau_cr_peak)
        self.register_buffer("prefactor", torch.tensor(prefactor))
        self.register_buffer("T_grain_cr_peak", torch.tensor(meta_params.T_grain_cr_peak))

    def forward(self, params_env, params_reac, T_gas, mask_T, E_d, freq_vib, **kwargs):
        prefactor = self.prefactor*params_reac["alpha"]
        rate = compute_evaporation_rate(prefactor, E_d, freq_vib, self.T_grain_cr_peak)
        return rate


class DummyZero(nn.Module):
    def forward(self, params_env, params_reac, *args, **kwargs):
        return torch.zeros_like(params_reac["alpha"])


def compute_vibration_frequency(ma, E_d, meta_params):
    return torch.sqrt(FACTOR_VIB_FREQ*meta_params.site_density*E_d/ma)


def compute_evaporation_rate(factor, E_d, freq_vib, T_dust):
    return factor*freq_vib*torch.exp(-E_d/T_dust)


#def compute_thermal_evapor_rate(factor, E_d, freq_vib, meta_params):
#    return factor*freq_vib*torch.exp(-E_d/meta_params.T_dust_0)


#def compute_cr_evapor_rate(factor, E_d, freq_vid,

#def thermal_evaporation(

#        rate = df_sub["A"].values*df_tmp["vib_ freq"].values*np.exp(-df_tmp["E_d"].astype('f4').values/meta_params.dust_temp_0)
