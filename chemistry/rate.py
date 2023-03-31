import torch
from torch import nn
from torch.nn import functional as F


class GasPhaseRate1st(nn.Module):
    def __init__(self, formula, params_reac, params_env, meta_params):
        super(GasPhaseRate1st, self).__init__()

        lookup = {"CR ionization": 0, "photodissociation": 1}
        mask = F.one_hot(torch.tensor([lookup[name] for name in formula]), len(lookup))
        mask = mask.type(torch.float32)
        self.register_buffer("mask", mask) # (R, F)

        self.formula_list = nn.ModuleList([
            CosmicRayIonization(meta_params.rate_cr_ion),
            Photodissociation()
        ])

        params_reac.register_buffer(self, "params_reac")
        params_env.register_buffer(self, "params_env")

    def forward(self, t_in):
        rate = torch.zeros_like(self.mask)
        for i_f, formula in enumerate(self.formula_list):
            rate[:, i_f] = formula(self.params_reac, self.params_env)
        rate = torch.sum(rate*self.mask, dim=-1)
        return rate


class CosmicRayIonization(nn.Module):
    """Cosmic-ray ionization.

    Args:
        rate_cr_ion (float): H2 cosmic-ray ionization rate [s^-1].
    """
    def __init__(self, rate_cr_ion):
        super(CosmicRayIonization, self).__init__()
        self.register_buffer("rate_cr_ion", torch.tensor(rate_cr_ion, dtype=torch.float32))

    def forward(self, params_reac, params_env):
        """
        Args:
            params_reac (KeyTensor): (R, 5). Reaction parameters.
            params_env (KeyTensor): (3,). Environment parameters.

        Returns:
            tensor: (R,). Reaction rate.
        """
        rate = params_reac.get("alpha")*self.rate_cr_ion
        return rate


class Photodissociation(nn.Module):
    def forward(self, params_reac, params_env):
        """
        Args:
            params_reac (KeyTensor): (R, 5). Reaction parameters.
            params_env (KeyTensor): (3,). Environment parameters.

        Returns:
            tensor: (R,). Reaction rate.
        """
        alpha = params_reac.get("alpha")
        gamma = params_reac.get("gamma")
        Av = params_env.get("Av")

        rate = alpha*torch.exp(-gamma*Av)
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


class GasGrainReaction(nn.Module):
    def forward(self, params, temp):
        """
        Args:
            params (tensor): (R, 5). Parameters.

                - Minimum reaction temperature.
                - Maximum reaction temperature.
                - Alpha.
                - Beta.
                - Gamma.

            temp (tensor): (B,). Temperature [K].

        Returns:
            tensor: (B, R). Reaction rate.
        """
        temp = temp[:, None]
        temp_min, temp_max, alpha, beta, _ = params.T
        rate = alpha*(temp/300.)**beta
        return rate
