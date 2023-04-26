import math
from dataclasses import dataclass

from .constants import M_ATOM


@dataclass(frozen=True)
class MetaParameters:
    """Meta parameters.

    Args:
        T_dust_0 (float): Initial dust temperature [K].
        site_density (float): Site density on one grain [cm^-2].
        grain_density (float): Grain mass density [g cm^-3].
        grain_radius (float): Grain radius [cm].
        dtg_mass_ratio_0 (float): Initial dust to gas mass ratio.

        rate_cr_ion (float): Cosmic ray ionisation rate [s^-1].
        rate_x_ion: (float): X-ray ionisation rate [s^-1].
        rate_fe_ion: (float): Fe-ion-grain encounter [s^-1].
        tau_cr_peak (float): Duration of peak grain temperature [s^-1].
        T_grain_cr_peak (float): Peak grain temperature due to cosmic ray heating [K].
        sticking_coeff_neutral (float):
        sticking_coeff_positive (float):
        sticking_coeff_negative (float):
    """
    # Grain parameters
    T_dust_0: float = 10.
    site_density: float = 1.5e15
    grain_density: float = 3.
    grain_radius: float = 1e-5
    dtg_mass_ratio_0: float = 1e-2

    rate_cr_ion: float = 1.3e-17
    rate_x_ion: float = 0.
    rate_fe_ion: float = 3e-14
    tau_cr_peak: float = 1e-5
    T_grain_cr_peak: float = 70.
    sticking_coeff_neutral: float = 1.
    sticking_coeff_positive: float = 0.
    sticking_coeff_negative: float = 0.

    @property
    def grain_mass(self):
        """Grain mass [ma]."""
        return 4*math.pi/3*self.grain_radius**3*self.grain_density/M_ATOM

    @property
    def dtg_num_ratio_0(self):
        """Initial dust to gas number ratio."""
        return self.dtg_mass_ratio_0/self.grain_mass


def modified_dtg_mass_ratio_0(ab_He, dtg_mass_ratio_0=None):
    """Modify the initial DTG mass ration using the He abundance."""
    if dtg_mass_ratio_0 is None:
        dtg_mass_ratio_0 = MetaParameters.dtg_mass_ratio_0
    return dtg_mass_ratio_0*(1 + 4*ab_He)