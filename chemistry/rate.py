import torch
from torch import nn
from torch.nn import functional as F


class ModifiedArrhenius(nn.Module):
    def forward(self, temp, params):
        """
        Args:
            temp (tensor): (B,). Temperature [K]. 
            params (tensor): (N, 5). Parameters.

                - Minimum reaction temperature.
                - Maximum reaction temperature.
                - Alpha.
                - Beta.
                - Gamma.

        Returns:
            (tensor): (B, N). Reaction rate.
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
