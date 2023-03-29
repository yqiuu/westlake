import torch
from torch import nn
from torch.nn import functional as F


class CosmicRayIonization(nn.Module):
    """Cosmic-ray ionization.

    Args:
        zeta (float): H2 cosmic-ray ionization rate [s^-1].
    """
    def __init__(self, zeta):
        super(CosmicRayIonization, self).__init__()
        self.register_buffer("zeta", torch.tensor(zeta, dtype=torch.float32))

    def forward(self, params):
        """
        Args:
            params (tensor): (R, 5). Parameters.

                - Minimum reaction temperature.
                - Maximum reaction temperature.
                - Alpha.
                - Beta.
                - Gamma.

        Returns:
            tensor: (R,). Reaction rate.
        """
        rate = params[:, 2]*self.zeta
        return rate


class ModifiedArrhenius(nn.Module):
    def forward(self, temp, params):
        """
        Args:
            temp (tensor): (B,). Temperature [K]. 
            params (tensor): (R, 5). Parameters.

                - Minimum reaction temperature.
                - Maximum reaction temperature.
                - Alpha.
                - Beta.
                - Gamma.

        Returns:
            tensor: (B, R). Reaction rate.
        """
        temp = temp[:, None]
        temp_min, temp_max, alpha, beta, gamma = params.T

        # TODO: Check how to compute the rate if the temperature is beyond the
        # range.
        t_width = temp_max - temp_min
        temp = (temp - temp_min)/t_width
        temp = F.hardtanh(temp, min_val=0.)
        temp = temp_min + (temp_max - temp_min)*temp

        rate = alpha*(temp/300.)**beta*torch.exp(-gamma/temp)
        return rate
