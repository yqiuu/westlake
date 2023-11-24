import math
from dataclasses import dataclass

from astropy import units

from .constants import M_ATOM


@dataclass(frozen=True)
class Config:
    """Config.

    Args:
        model (str):
            - 'simple': Gas phase model.
            - 'two phase': Two phase model.
            - 'three phase': Three phase model.

        site_density (float): Site density on one grain [cm^-2].
        grain_density (float): Grain mass density [g cm^-3].
        grain_radius (float): Grain radius [cm].
        dtg_mass_ratio (float): Initial dust to gas mass ratio.
        diffusion_barrier_thickness (float): Diffusion_barrier_thickness [cm].
        chemical_barrier_thickness (float): Grain reaction activation energy
            barrier width [cm].
        rate_cr_ion (float): Cosmic ray ionisation rate [s^-1].
        rate_x_ion: (float): X-ray ionisation rate [s^-1].
        rate_fe_ion: (float): Fe-ion-grain encounter [s^-1].
        tau_cr_peak (float): Duration of peak grain temperature [s^-1].
        T_grain_cr_peak (float): Peak grain temperature due to cosmic ray
            heating [K].
        sticking_coeff_neutral (float):
        sticking_coeff_positive (float):
        sticking_coeff_negative (float):
        vib_to_dissip_freq_ratio (float): The ratio of the surface-molecule bond
            frequency to the frequency at which energy is lost to the grain
            surface (Garrod el al. 2007).
        surf_diff_to_deso_ratio (float): Factor to convert desorption energy
            to diffusion energy for surface species.
        mant_diff_to_deso_ratio (float): Factor to convert desorption energy
            to diffusion energy for mantle species.
        num_active_layers (float): Number of active layers.
        uv_flux (float): UV flux.

        use_photodesorption (bool): If True, enable photodesorption.

        den_Av_ratio (float): Density to Av ratio to compute self-shielding.
        H2_shielding (str): Set "Lee+1996" to turn on H2 shielding. Set None to
            turn off.
        CO_shielding (str): Set "Lee+1996" to turn on CO shielding. Set None to
            turn off.

        method (str): ODE solver.
        rtol (float): Relative tolerance.
        atol (float): Ababsolute tolerance.
        ab_0_min (float): Minimum initial abundances.
        use_auto_jac (bool): If True, use `jacrev` in `torch` to compute
            jacobian.

        to_second (float): A unit factor that converts the desired unit to
            second. The default value converts year to second.

        special_species (tuple): Species whose abundance is irrelevant in the
            simulation, e.g. CR, CRP and Photon.
    """
    model: str

    # Grain parameters
    site_density: float = 1.5e15
    grain_density: float = 3.
    grain_radius: float = 1e-5
    dtg_mass_ratio: float = 1e-2
    diffusion_barrier_thickness: float = 1e-8
    chemical_barrier_thickness: float = 1e-8
    rate_cr_ion: float = 1.3e-17
    rate_x_ion: float = 0.
    rate_fe_ion: float = 3e-14
    tau_cr_peak: float = 1e-5
    T_grain_cr_peak: float = 70.
    sticking_coeff_neutral: float = 1.
    sticking_coeff_positive: float = 0.
    sticking_coeff_negative: float = 0.
    vib_to_dissip_freq_ratio: float = 1e-2
    surf_diff_to_deso_ratio: float = 0.4
    mant_diff_to_deso_ratio: float = 0.8
    num_active_layers: float = 2.
    uv_flux: float = 1.

    # Switches
    use_photodesorption: bool = False

    # Sheilding
    den_Av_ratio: float = 1./(5.34e-22)
    H2_shielding: str = None # "Lee+1996"
    CO_shielding: str = None # "Lee+1996"

    # Physcial parameters
    Av: float = 10.
    den_gas: float = 2.e+4
    T_gas: float = 10.
    T_dust: float = 10

    # Numerics
    t_start: float = 0.
    t_end: float = 1.e+6
    solver: str = "LSODA"
    rtol: float = 1e-4
    atol: float = 1e-20
    ab_0_min: float = 0.
    use_auto_jac: bool = False

    #
    to_second: float = units.year.to(units.second)

    #
    special_species: tuple = ("CR", "CRP", "Photon")


    @property
    def grain_mass(self):
        """Grain mass [ma]."""
        return 4*math.pi/3*self.grain_radius**3*self.grain_density/M_ATOM

    @property
    def dtg_num_ratio(self):
        """Initial dust to gas number ratio."""
        return self.dtg_mass_ratio/self.grain_mass

    @property
    def num_sites_per_grain(self):
        """Number of sites per grain."""
        return 4.*math.pi*self.grain_radius**2*self.site_density


def fixed_dtg_mass_ratio(ab_He, dtg_mass_ratio=None):
    """Fix the initial DTG mass ratio using the He abundance."""
    if dtg_mass_ratio is None:
        dtg_mass_ratio = Config.dtg_mass_ratio
    return dtg_mass_ratio*(1 + 4*ab_He)
