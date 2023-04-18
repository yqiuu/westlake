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
        T_cr_peak (float): Peak grain temperature due to cosmic ray heating [K].
    """
    # Grain parameters
    T_dust_0: float = 10.
    site_density: float = 1.5e15
    grain_density: float = 3.
    grain_radius: float = 1e-5
    dtg_mass_ratio_0: float = 1e-2

    rate_cr_ion: float = 1.3e-17
    rate_x_ion: float = 0.
    rate_fe_ion: float = 1e-14
    tau_cr_peak: float = 1e-5
    T_peak_cr: float = 70.

    @property
    def grain_mass(self):
        """Grain mass [ma]."""
        return 4*math.pi/3*self.grain_radius**3*self.grain_density/M_ATOM

    @property
    def grain_ab_0(self):
        """Initial grain abundance."""
        return self.dtg_mass_ratio_0/self.grain_mass
