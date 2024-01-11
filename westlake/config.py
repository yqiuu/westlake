import math
from dataclasses import dataclass
from pathlib import Path

from astropy import units
from docstring_parser import parse

from .constants import M_ATOM


@dataclass(frozen=True)
class Config:
    """Config.

    Args:
        model (str): Model name.
            - 'simple': Gas phase model.
            - 'two phase': Two phase model.
            - 'three phase': Three phase model.
        t_start (float): Starting time.
        t_end(float): Ending time.
        save_rate_coeffs (bool): Switch to save rate coefficents.

        site_density (float): Site density on one grain [cm^-2].
        grain_density (float): Grain mass density [g cm^-3].
        grain_radius (float): Grain radius [cm].
        dtg_mass_ratio (float): Dust-to-gas mass ratio.
        diffusion_barrier_thickness (float): Diffusion barrier thickness [cm].
        chemical_barrier_thickness (float): Grain reaction activation energy
            barrier width [cm].
        factor_cr_peak (float): Prefactor to compute comsmic ray
            evaporation [s^-1].
        T_grain_cr_peak (float): Peak grain temperature due to cosmic ray
            heating [K].
        sticking_coeff_neutral (float): Sticking coefficient of neutral species.
        sticking_coeff_positive (float): Sticking coefficient of positive
            species.
        sticking_coeff_negative (float): Sticking coefficient of negative
            species.
        vib_to_dissip_freq_ratio (float): The ratio of the surface-molecule bond
            frequency to the frequency at which energy is lost to the grain
            surface (Garrod el al. 2007).
        surf_diff_to_deso_ratio (float): Factor to convert desorption energy
            to diffusion energy for surface species.
        mant_diff_to_deso_ratio (float): Factor to convert desorption energy
            to diffusion energy for mantle species.
        num_active_layers (float): Number of active layers.
        uv_flux (float): UV flux.
        use_competition (bool): Switch of competition mechanism

        den_Av_ratio (float): Density to Av ratio to compute self-shielding.
        H2_shielding (str): Switch of H2 sheilding (None, Lee+1996).
        CO_shielding (str): Switch of CO shielding (None, Lee+1996).

        Av (float): Visual extinction.
        den_gas (float): Gas density.
        T_gas (float): Gas temperature.
        T_dust (float): Dust temperature.
        zeta_cr (float): Cosmic ray ionisation rate [s^-1].
        zeta_xr (float): X-ray ionisation rate [s^-1].

        use_scipy_solver (bool): Switch to use a scipy ODE sovler.
        method (str): Name of the scipy ODE solver. This only works when
            `use_scipy_solver=True`; otherwise, the code uses a BDF solver
            implmented using `torch`.
        rtol (float): Relative tolerance.
        atol (float): Ababsolute tolerance.
        ab_0_min (float): Minimum initial abundances. This should not be zero
            for the three model.
        use_auto_jac (bool): If True, use `jacrev` in `torch` to compute
            jacobian, which is slow but accurate.
        to_second (float): A unit factor that converts the desired unit to
            second. The default value converts year to second.
        special_species (tuple): Species whose abundance is irrelevant in the
            simulation, e.g. CR, CRP and Photon.
    """
    model: str = "two phase"
    t_start: float = 0.
    t_end: float = 1.e+6
    save_rate_coeffs: bool = True

    # Grain parameters
    site_density: float = 1.5e15
    grain_density: float = 3.
    grain_radius: float = 1e-5
    dtg_mass_ratio: float = 1e-2
    diffusion_barrier_thickness: float = 1e-8
    chemical_barrier_thickness: float = 1e-8
    factor_cr_peak: float = 3e-19
    T_grain_cr_peak: float = 70.
    sticking_coeff_neutral: float = 1.
    sticking_coeff_positive: float = 0.
    sticking_coeff_negative: float = 0.
    vib_to_dissip_freq_ratio: float = 1e-2
    surf_diff_to_deso_ratio: float = 0.4
    mant_diff_to_deso_ratio: float = 0.8
    num_active_layers: float = 2.
    uv_flux: float = 1.

    #
    use_competition: bool = False

    # Sheilding
    den_Av_ratio: float = 1./(5.34e-22)
    H2_shielding: str = None # "Lee+1996"
    CO_shielding: str = None # "Lee+1996"

    # Physcial parameters
    Av: float = 10.
    den_gas: float = 2.e+4
    T_gas: float = 10.
    T_dust: float = 10
    zeta_cr: float = 1.3e-17
    zeta_xr: float = 0.

    # Numerics
    use_scipy_solver: bool = False
    method: str = "BDF"
    rtol: float = 1e-4
    atol: float = 1e-25
    ab_0_min: float = 1e-40
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

    @property
    def layer_factor(self):
        """Layer factor (num_layer = ab*layer_factor)."""
        return 1./(self.dtg_num_ratio*self.num_sites_per_grain)


def fixed_dtg_mass_ratio(ab_He, dtg_mass_ratio=None):
    """Fix the initial DTG mass ratio using the He abundance."""
    if dtg_mass_ratio is None:
        dtg_mass_ratio = Config.dtg_mass_ratio
    return dtg_mass_ratio*(1 + 4*ab_He)


def save_config_template(fname):
    def create_description(text):
        text = text.replace("None", "null")
        text = text.replace("\n", " ")
        text = text.split(" ")
        text_ret = "# "
        len_curr = 0
        for word in text:
            len_new = len_curr + len(word) + 1
            if len_new > 80 or word.startswith("-"):
                text_ret += "\n# "
                len_curr = 0
            text_ret += f"{word} "
            len_curr += len(word) + 1
        text_ret += "\n"
        return text_ret

    exclude_list = [
        "to_second",
        "special_species",
    ]
    with open(fname, "w") as fp:
        for var in parse(Config.__doc__).params:
            if var.arg_name in exclude_list:
                continue

            fp.write(create_description(var.description))
            val = getattr(Config, var.arg_name)
            if val is None:
                text = "{}: null".format(var.arg_name, val)
            elif var.type_name == "float":
                if abs(val) >= 1e-3 and abs(val) < 1e4:
                    text = "{}: {:.2f}".format(var.arg_name, val)
                else:
                    text = "{}: {:.2e}".format(var.arg_name, val)
            else:
                text = "{}: {}".format(var.arg_name, val)
            text += "\n\n"
            fp.write(text)