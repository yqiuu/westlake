from dataclasses import dataclass


@dataclass
class MetaParameters:
    """Meta parameters.

    Args:
        T_dust_0 (float): Initial dust temperature [K].
        site_density (float): Site density on one grain [cm^-2].

        rate_cr_ion (float): Cosmic ray ionisation rate [s^-1].
        rate_x_ion: (float): X-ray ionisation rate [s^-1].
        rate_fe_ion: (float): Fe-ion-grain encounter [s^-1].
        tau_cr_peak (float): Duration of peak grain temperature [s^-1].
        T_cr_peak (float): Peak grain temperature due to cosmic ray heating [K].
    """
    # Grain parameters
    T_dust_0: float = 10.
    site_density: float = 1.5e15

    rate_cr_ion: float = 1.3e-17
    rate_x_ion: float = 0.
    rate_fe_ion: float = 1e-14
    tau_cr_peak: float = 1e-5
    T_peak_cr: float = 70.
