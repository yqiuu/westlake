__all__ = [
    "M_ATOM",
    "K_B",
    "H_BAR",
    "FACTOR_VIB_FREQ",
]


import math

from astropy import constants


# Atomic mass [g]
M_ATOM = 1.66053892e-24
# Boltzmann constant [g cm^2 s^-2 K^-1]
K_B = constants.k_B.cgs.value
# Reduced Planck constant [g cm^2 s^-1]
H_BAR  = constants.hbar.cgs.value
# Prefactor to compute the vibration frequency [cm^2 s^-2]
FACTOR_VIB_FREQ = 2*K_B/(math.pi*math.pi*M_ATOM)
