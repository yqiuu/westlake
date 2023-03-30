__all__ = [
    "M_ATOM",
    "FACTOR_VIB_FREQ",
]


import math

from astropy import constants


# Atomic mass [g]
M_ATOM = 1.66053892e-24 
# Prefactor to compute the vibration frequency [cm^2 s^-2]
FACTOR_VIB_FREQ = 2*constants.k_B.cgs.value/(math.pi*math.pi*M_ATOM)
