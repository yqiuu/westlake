import math

import torch
from torch import nn

from .constants import FACTOR_VIB_FREQ


class ThermalEvaporation(nn.Module):
    def forward(self, rmat_1st, meta_params):
        factor = rmat_1st.params_reac.get("factor")
        E_d, freq_vid = rmat_1st.params_spec.get(("E_d", "freq_vib"))[rmat_1st.spec_r]
        rate = compute_evaporation_rate(factor, E_d, freq_vib, meta_params.T_dust_0)
        return rate

class CosmicRayEvaporation(nn.Module):
    def forward(self, rmat_1st, meta_params):
        factor = rmat_1st.params_reac.get("factor") \
            * (meta_params.rate_cr_ion + meta_params.rate_x_ion) / 1.3e-7 \
            * (meta_params.rate_fe_ion*meta_params.tau_cr_peak)
        E_d, freq_vid = rmat_1st.params_spec.get(("E_d", "freq_vib"))[rmat_1st.spec_r]
        rate = compute_evaporation_rate(factor, E_d, freq_vib, meta_params.T_dust_cr_peak)
        return rate

def compute_vibration_frequency(E_d, mass, meta_params):
    return torch.sqrt(FACTOR_VIB_FREQ*meta_params.site_density*E_d/mass)


def compute_evaporation_rate(factor, E_d, freq_vib, T_dust):
    return factor*freq_vib*torch.exp(-E_d/T_dust)


def compute_thermal_evapor_rate(factor, E_d, freq_vib, meta_params):
    return factor*freq_vib*torch.exp(-E_d/meta_params.T_dust_0)


#def compute_cr_evapor_rate(factor, E_d, freq_vid, 

#def thermal_evaporation(

#        rate = df_sub["A"].values*df_tmp["vib_ freq"].values*np.exp(-df_tmp["E_d"].astype('f4').values/meta_params.dust_temp_0)
